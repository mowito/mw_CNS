"""Hybrid (2 1/2 D) safe velocity control -- paper section G, Eq. 24-25.

    v = -lambda * J^-1 e                                                  (24)

The paper defers the Jacobian to
  [13] Malis, Chaumette & Boudet, "2 1/2 D visual servoing", T-RA 15(2), 1999,
which IS now available at /home/mowito/Downloads/Malis2-1-2DTRA99.pdf, so the
interaction matrix below is transcribed from it rather than reconstructed.

Why hybrid at all: pure PBVS (Eq. 5) takes the shortest Cartesian path and can
rotate the scene out of the field of view -- the "feature loss problem". The
paper measures it (Table I row 7): always-PBVS drops SR to 18/20 vs 20/20 for
hybrid->PBVS, and "the failed cases are with large initial viewpoint deviation
inducing feature loss problem".

WHAT MALIS ACTUALLY SPECIFIES
-----------------------------
Reference point P on the target, extended image coordinates (Malis Eq. 14):

    m_e = [x, y, log(Z)],     x = X/Z,  y = Y/Z

Task function (Eq. 19) and its interaction matrix (Eq. 21):

    e = [ m_e - m_e* ; theta*u ]          (6-vector)
    L = [ (1/d*) L_v   L_vw ]
        [   0            L_w ]

    (1/d*) L_v = (1/Z) A,   A    = [[-1,0, x],[0,-1, y],[0,0,-1]]   (Eq. 16)
    L_vw       = [[x*y, -(1+x^2), y], [1+y^2, -x*y, -x], [-y, x, 0]] (Eq. 18)

L is UPPER TRIANGULAR, and Malis's central result (Eq. 33, Sec. IV-B) is that it
therefore has *no singularity in the whole task space* -- that is the "safe" in
"Safe Velocity Control". So the control law needs no damped pseudo-inverse; the
inverse is closed form. A is an involution up to sign:

    A^-1 = [[-1,0,-x],[0,-1,-y],[0,0,-1]]

and Malis Eq. 24 shows L_w^-1 (theta*u) = theta*u exactly, so the rotational loop
is pure theta-u with no Jacobian at all. Control law (Eq. 23) reduces to:

    omega = -lambda * theta*u
    v     = -lambda * Z_hat * A^-1 * [ (m_e - m_e*) - L_vw * theta*u ]

MALIS Eq. 23 IS *NOT* PAPER Eq. 24 -- READ THIS BEFORE CHOOSING A LAW
----------------------------------------------------------------------
An earlier revision of this file claimed they were the same law with the gravity
centre as Malis's reference point. **That was wrong, and the mistake was
measurable.** The paper writes its error vector explicitly:

    e = [ d t_hat_c ; cX_hat_g - dX_hat_g ; theta_hat*u_hat_z ]      (paper Eq. 24)

i.e. TRANSLATION is driven by the Cartesian estimate and the image gravity-centre
error drives ROTATION (+ theta*u_z in-plane). Malis Eq. 23 is the opposite split:
translation from (extended) image coordinates, rotation from full theta*u. The
paper cites [13] for the Jacobian *blocks*, not for the composition of e.

With GROUND-TRUTH inputs both laws converge (the DAgger expert ran the Malis form
and reached ~2 mm). With NETWORK estimates they are not interchangeable: far from
the goal, feature loss degrades the dual-softmax matching, so a law whose
TRANSLATION follows the image error chases phantom gravity centres. Measured
2026-07-30, closed loop on network estimates, Malis form: ep0 diverged
497 mm -> 10.3 m; ep2 spent 1493/1500 steps in hybrid mode because the camera
never approached the goal and matching never recovered. The paper's composition
bounds the damage: translation follows t_hat_c recovered from the policy's own
velocity (per-bin dir cos 0.77-0.92), while the noisy image term steers only
rotation -- exactly the dof that keeps features in view.

Use `hybrid_velocity_eq24` for deployment and rollouts. `hybrid_velocity` (the
Malis form) is kept as the faithful transcription of [13] for reference and for
`tests/test_malis_hybrid.py`.

The gravity centres (Eq. 25) and the switching criterion
||cX_g - dX_g|| > 0.1*sqrt(N16) are exact per the paper. Gravity centre as the
tracked point is Malis Sec. IV-B's own suggestion ("the center of gravity of the
target in the image... would increase the probability that the target remains in
the camera field of view"), which Eq. 25 estimates from dual-softmax confidences.

SCOPE: the hybrid law drives the ROLLOUT/deployment only. PBVS is what is
"directly supervised" (paper Sec. G) -- this is not the regression target.
"""
import numpy as np

# Patch grid -> pixel scale for the canonical camera (fx=fy=512, 512x512, 32x32 patches)
PATCH_PX = 16.0

CANONICAL_K = dict(fx=512.0, fy=512.0, cx=256.0, cy=256.0)


def _skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def patch_to_normalized(Xg_patch, fx=512.0, fy=512.0, cx=256.0, cy=256.0,
                        patch_px=PATCH_PX):
    """Patch-coordinate gravity centre -> normalized image coords (x = (u-cx)/fx).

    gravity_centers() returns patch units; Malis's interaction matrix is defined
    on normalized coordinates, so this conversion is required and was an easy
    thing to get silently wrong.
    """
    u = (np.asarray(Xg_patch, float) + 0.5) * patch_px    # patch centre in px
    return np.array([(u[0] - cx) / fx, (u[1] - cy) / fy])


def A_matrix(x, y):
    """Malis Eq. 16 numerator: (1/d*) L_v = (1/Z) A."""
    return np.array([[-1.0, 0.0, x],
                     [0.0, -1.0, y],
                     [0.0, 0.0, -1.0]])


def A_inverse(x, y):
    """Closed-form inverse of A_matrix -- A is an involution up to the sign of x,y."""
    return np.array([[-1.0, 0.0, -x],
                     [0.0, -1.0, -y],
                     [0.0, 0.0, -1.0]])


def L_vw(x, y):
    """Malis Eq. 18: rotational block of the extended-image-coordinate Jacobian."""
    return np.array([[x * y, -(1.0 + x * x), y],
                     [1.0 + y * y, -x * y, -x],
                     [-y, x, 0.0]])


def L_omega(theta_u):
    """Malis Eq. 11: L_w(u,theta) = I - (theta/2)[u]x + (1 - sinc(theta)/sinc^2(theta/2))[u]x^2

    Provided for completeness / testing -- the control law never needs it, because
    Malis Eq. 24 proves L_w^-1 (theta*u) = theta*u.
    """
    tu = np.asarray(theta_u, float)
    theta = np.linalg.norm(tu)
    if theta < 1e-9:
        return np.eye(3)
    u = tu / theta
    ux = _skew(u)
    sinc = np.sin(theta) / theta
    sinc_half = np.sin(theta / 2.0) / (theta / 2.0)
    return np.eye(3) - (theta / 2.0) * ux + (1.0 - sinc / (sinc_half ** 2)) * (ux @ ux)


def log_depth_ratio(x, y, R_dc, t_dc, Z_c=1.0):
    """Third component of (m_e - m_e*): log(rho_2) = log(Z_c / Z_d), Malis Eq. 9/14.

    The reference point sits at cP = Z_c * [x, y, 1] in the current camera frame;
    its depth in the desired frame follows from the estimated relative pose.

    R_dc, t_dc : rotation/translation taking a point from the CURRENT camera frame
                 to the DESIRED camera frame (dP = R_dc cP + t_dc).
    """
    cP = float(Z_c) * np.array([x, y, 1.0])
    Z_d = float((np.asarray(R_dc, float) @ cP + np.asarray(t_dc, float))[2])
    # Behind-camera / degenerate estimates would make the log blow up; the caller
    # is near a pose where the hybrid law is meaningless anyway, so clamp.
    if Z_d <= 1e-6 or Z_c <= 1e-6:
        return 0.0
    return float(np.log(Z_c / Z_d))


def hybrid_velocity(theta_u, cXg_patch, dXg_patch, R_dc=None, t_dc=None,
                    Z=1.0, lam=1.0, intrinsic=None):
    """Malis Eq. 23 / paper Eq. 24. Returns a 6-D camera-frame twist [v; omega].

    theta_u   : (3,) estimated rotation error as a rotation vector (theta*u),
                from d_c R_hat
    cXg_patch : (2,) current image gravity centre, patch units (Eq. 25)
    dXg_patch : (2,) desired image gravity centre, patch units (Eq. 25)
    R_dc,t_dc : estimated relative pose current->desired. Used ONLY for the
                log-depth-ratio component of the error. If omitted, that
                component is dropped to 0 (pure image + rotation control), which
                is what you want when no depth estimate is trustworthy.
    Z         : depth estimate of the reference point (d_hat*); paper canonical 1 m
    lam       : control gain
    """
    k = dict(CANONICAL_K)
    if intrinsic:
        k.update(intrinsic)
    xc = patch_to_normalized(cXg_patch, **k)
    xd = patch_to_normalized(dXg_patch, **k)
    tu = np.asarray(theta_u, float)

    # e = [ m_e - m_e* ; theta*u ], evaluated at the CURRENT gravity centre.
    if R_dc is not None and t_dc is not None:
        dlog = log_depth_ratio(xc[0], xc[1], R_dc, t_dc, Z_c=Z)
    else:
        dlog = 0.0
    e_img = np.array([xc[0] - xd[0], xc[1] - xd[1], dlog])

    # Malis Eq. 23, using L_w^-1 (theta*u) = theta*u (Eq. 24).
    x, y = float(xc[0]), float(xc[1])
    omega = -lam * tu
    v = -lam * float(Z) * (A_inverse(x, y) @ (e_img - L_vw(x, y) @ tu))
    return np.concatenate([v, omega])


def interaction_matrix(x, y, Z, theta_u):
    """Full Malis Eq. 21 L, for tests: verifies the closed-form inverse above."""
    L = np.zeros((6, 6))
    L[0:3, 0:3] = A_matrix(x, y) / float(Z)
    L[0:3, 3:6] = L_vw(x, y)
    L[3:6, 3:6] = L_omega(theta_u)
    return L


def gravity_center_gap(cXg_patch, dXg_patch):
    return float(np.linalg.norm(np.asarray(cXg_patch, float) - np.asarray(dXg_patch, float)))


def should_use_hybrid(cXg_patch, dXg_patch, n16=1024):
    """Paper section G: use hybrid while ||cX_g - dX_g|| > 0.1*sqrt(N16),
    then switch to PBVS for final precision. For H16=W16=32, N16=1024 so the
    threshold is 3.2 patch units (~51 px)."""
    return gravity_center_gap(cXg_patch, dXg_patch) > 0.1 * np.sqrt(n16)


# ---------------------------------------------------------------------------
# Paper Eq. 24 -- THE deployment law. NOT the same as Malis Eq. 23 above.
# ---------------------------------------------------------------------------
def point_interaction(x, y, Z):
    """Standard 2x6 interaction matrix for a normalized image point at depth Z
    (Malis Eq. 16+18 rows, the classic L(x, Z)): xdot = L [v; omega]."""
    return np.array([
        [-1.0 / Z, 0.0, x / Z, x * y, -(1.0 + x * x), y],
        [0.0, -1.0 / Z, y / Z, 1.0 + y * y, -x * y, -x],
    ])


def hybrid_velocity_eq24(t_c, theta_u, cXg_patch, dXg_patch, Z=1.0, lam=1.0,
                         intrinsic=None):
    """Paper Eq. 24 verbatim:  v = -lam J^-1 e,  e = [ t_c ; cXg - dXg ; theta_u_z ].

    This is NOT Malis Eq. 23, although the paper cites [13] for the Jacobian
    blocks. Malis's 2.5D drives TRANSLATION from (extended) image coordinates and
    rotation from theta*u. The paper's e-vector does the opposite split:
    TRANSLATION from the Cartesian estimate t_c and ROTATION from the image
    gravity-centre error (+ theta*u_z for in-plane alignment).

    Why that matters at deployment (measured, 2026-07-30): with NETWORK estimates
    the two laws are not equivalent. Far from the goal the dual-softmax matching
    is degraded by feature loss, so an image-driven translation chases phantom
    gravity centres -- running Malis Eq. 23 off network estimates diverged to
    10.3 m on ep0 and stayed in hybrid mode 1493/1500 steps on ep2, because the
    camera never approached the goal and matching never recovered. The paper's
    split bounds the damage: translation follows t_hat_c recovered from the
    policy's own velocity (dir cos 0.77-0.92 across bins), and the noisy image
    term steers only rotation, which is exactly the dof that keeps features in
    the field of view. Malis Eq. 23 with GROUND-TRUTH inputs remains fine (the
    DAgger expert used it and converged); it is the estimated-input regime where
    the paper's structure is the right one.

    t_c       : (3,) position of the DESIRED camera origin in the CURRENT frame,
                real units ( = -R_dc^T t_dc )
    theta_u   : (3,) rotation error current->desired as a rotation vector
    cXg/dXg   : (2,) image gravity centres, patch units (Eq. 25)
    Z         : depth estimate of the gravity-centre point (~ scene scale s)

    Returns a 6-D camera-frame twist [v; omega], real units.
    """
    k = dict(CANONICAL_K)
    if intrinsic:
        k.update(intrinsic)
    xc = patch_to_normalized(cXg_patch, **k)
    xd = patch_to_normalized(dXg_patch, **k)
    t_c = np.asarray(t_c, float)

    e = np.concatenate([t_c, xc - xd, [float(theta_u[2])]])

    J = np.zeros((6, 6))
    # d/dt of the desired-origin position p carried in the moving camera frame:
    # for a static point, pdot = -v - omega x p = -v + [p]x omega.
    J[0:3, 0:3] = -np.eye(3)
    J[0:3, 3:6] = _skew(t_c)
    # image gravity centre treated as a point feature at depth Z
    J[3:5, :] = point_interaction(xc[0], xc[1], Z)
    # theta-u interaction is ~I for the omega block (Malis Eq. 24), so the
    # in-plane row picks out omega_z.
    J[5, 5] = 1.0

    # J is invertible whenever Z > 0: rows 0-2 are rank 3 in v, and the
    # remaining 3x3 omega block [L_omega-part; e_z] degenerates only when the
    # gravity centre sits at infinity. Solve directly; fall back to damped LS
    # near numerical singularity rather than damping unconditionally.
    try:
        return -lam * np.linalg.solve(J, e)
    except np.linalg.LinAlgError:
        JtJ = J.T @ J + 1e-6 * np.eye(6)
        return -lam * np.linalg.solve(JtJ, J.T @ e)
