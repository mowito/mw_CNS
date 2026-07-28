"""Hybrid (2.5D-style) safe velocity control -- paper section G, Eq. 24-25.

    v = -lambda * J^-1 e,    e = [ d t_c ; cX_g - dX_g ; theta*u_z ]      (24)

Purpose: pure PBVS (Eq. 5) takes the shortest Cartesian path and can rotate the
scene out of the field of view -- the "feature loss problem". The paper measures
this directly (Table I row 7): always-PBVS drops SR to 18/20 vs 20/20 for
hybrid->PBVS, and "the failed cases are with large initial viewpoint deviation
inducing feature loss problem". Our own closed-loop eval starts at median
TE 1.26m / RE 99 deg, squarely in that regime.

The scheme drives TRANSLATION from the Cartesian error (straight-line path) but
ROTATION from the image gravity-centre error (keeps the matched features
centred), plus theta*u_z for in-plane alignment. Per the paper it is used while
||cX_g - dX_g|| > 0.1*sqrt(N16) and then hands over to PBVS, "which is directly
supervised", for final precision.

IMPORTANT SCOPE NOTE: the paper defers the exact Jacobian to
  [13] Malis, Chaumette & Boudet, "2 1/2 D visual servoing", T-RA 15(2), 1999.
That paper is not available locally, so the interaction matrix below is BUILT
FROM STANDARD VISUAL-SERVO BLOCKS (point-feature interaction matrix + theta-u
rotation block), not transcribed from [13]. It follows the same structure and
achieves the same stated goal, but the numerical conditioning may differ from
the original. The gravity centres (Eq. 25) and the switching criterion ARE exact
per the paper. Cross-check against ViSP's vpServo (built at
/home/mowito/mw_ws/install/VISP) before trusting this for real hardware.
"""
import numpy as np

# Patch grid -> pixel scale for the canonical camera (fx=fy=512, 512x512, 32x32 patches)
PATCH_PX = 16.0


def _skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def patch_to_normalized(Xg_patch, fx=512.0, fy=512.0, cx=256.0, cy=256.0,
                        patch_px=PATCH_PX):
    """Patch-coordinate gravity centre -> normalized image coords (x = (u-cx)/fx).

    gravity_centers() returns patch units; the point-feature interaction matrix
    is defined on normalized coordinates, so this conversion is required and was
    an easy thing to get silently wrong.
    """
    u = (np.asarray(Xg_patch, float) + 0.5) * patch_px    # patch centre in px
    return np.array([(u[0] - cx) / fx, (u[1] - cy) / fy])


def point_interaction(x, y, Z):
    """Standard 2x6 interaction matrix for a normalized image point at depth Z:
    xdot = L [v; omega]."""
    return np.array([
        [-1.0 / Z, 0.0, x / Z, x * y, -(1.0 + x * x), y],
        [0.0, -1.0 / Z, y / Z, 1.0 + y * y, -x * y, -x],
    ])


def hybrid_velocity(t_c, theta_u, cXg_patch, dXg_patch, Z=1.0, lam=1.0,
                    intrinsic=None, damping=1e-4):
    """Eq. 24. Returns a 6-D camera-frame twist [v; omega].

    t_c       : (3,) estimated translation current->desired, current camera frame
    theta_u   : (3,) estimated rotation error as a rotation vector (theta*u)
    cXg_patch : (2,) current image gravity centre, patch units
    dXg_patch : (2,) desired image gravity centre, patch units
    Z         : scene depth estimate (d*); the paper's canonical scale is 1m
    """
    k = dict(fx=512.0, fy=512.0, cx=256.0, cy=256.0)
    if intrinsic:
        k.update(intrinsic)
    xc = patch_to_normalized(cXg_patch, **k)
    xd = patch_to_normalized(dXg_patch, **k)

    # e = [ t_c (3) ; image gravity-centre error (2) ; theta*u_z (1) ]
    e = np.concatenate([np.asarray(t_c, float), xc - xd, [float(theta_u[2])]])

    J = np.zeros((6, 6))
    # d/dt of a translation error carried in the moving camera frame:
    #   edot = -v + [e_t]_x omega
    J[0:3, 0:3] = -np.eye(3)
    J[0:3, 3:6] = _skew(np.asarray(t_c, float))
    # image point block, evaluated at the CURRENT gravity centre
    J[3:5, :] = point_interaction(xc[0], xc[1], Z)
    # in-plane rotation: theta-u interaction matrix is ~I for the omega block,
    # so the z row picks out omega_z.
    J[5, 5] = 1.0

    # Damped least squares: J can be ill-conditioned when the gravity centres
    # coincide or the translation error is near zero.
    JtJ = J.T @ J + damping * np.eye(6)
    v = -lam * np.linalg.solve(JtJ, J.T @ e)
    return v


def gravity_center_gap(cXg_patch, dXg_patch):
    return float(np.linalg.norm(np.asarray(cXg_patch, float) - np.asarray(dXg_patch, float)))


def should_use_hybrid(cXg_patch, dXg_patch, n16=1024):
    """Paper section G: use hybrid while ||cX_g - dX_g|| > 0.1*sqrt(N16),
    then switch to PBVS for final precision. For H16=W16=32, N16=1024 so the
    threshold is 3.2 patch units (~51 px)."""
    return gravity_center_gap(cXg_patch, dXg_patch) > 0.1 * np.sqrt(n16)
