"""CNSv2 velocity denormalization (paper Eq. 18-21).

The controller is trained in a normalized "unit world" (canonical intrinsic,
Z~_d = 1) and predicts a normalized velocity v~ = [c v~_c ; c w~_c]. At inference
we recover a real-world velocity:

  Eq. 18   {d_c R^, d_c t^} <- PBVS^-1(c v~_c, c w~_c)      (reverse of Eq. 5)
  Eq. 19-20  c v^_c = s . c v~_c,   c w^_c = c w~_c,   s = d* (real scene scale)
  Eq. 21   real-pinhole case: rebuild E with the real intrinsic scaling and
           re-decompose (SVD, Eq. 3-4) to get the corrected velocity.

For the common case (real intrinsic == canonical), only the scale step (Eq. 20)
is needed -- that is what training uses via NeuralController.postprocess. The
intrinsic-adaptation path (Eq. 21) matters only when the real camera's focal
length / principal point differ from the canonical training setup.
"""

import numpy as np


def denormalize_scale(v_norm, d_star: float):
    """Eq. 19-20. v_norm: (6,) [v; w] normalized. Returns real-world (6,)."""
    v = np.asarray(v_norm, dtype=np.float64).copy()
    v[:3] *= d_star
    return v


def _skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]], dtype=np.float64)


def _rotvec_to_R(rv):
    theta = np.linalg.norm(rv)
    if theta < 1e-9:
        return np.eye(3)
    k = rv / theta
    K = _skew(k)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def pbvs_inverse(v_norm, lam: float = 1.0):
    """Eq. 18: recover relative pose {d_c R, d_c t} from a PBVS velocity screw.

    Reverse of Eq. 5 (c v_c = -lam R^T t, c w_c = -lam theta u):
      theta u = -w / lam       ->  d_c R = Rodrigues(theta u)
      t = -R v / lam           (since v = -lam R^T t)
    Returns (R (3,3), t (3,)).
    """
    v = np.asarray(v_norm, dtype=np.float64)
    trans, omega = v[:3], v[3:]
    rotvec = -omega / lam
    R = _rotvec_to_R(rotvec)
    t = -(R @ trans) / lam
    return R, t


def denormalize_intrinsic(v_norm, K_real, K_canonical, d_star: float, lam: float = 1.0):
    """Eq. 21 path. Adapt a normalized velocity to a real pinhole whose intrinsic
    differs from the canonical training one, then re-apply real scale.

    Approach (per paper Eq. 3-4, 18-21): recover the relative pose implied by the
    normalized velocity (Eq. 18), rebuild the essential matrix E = [t]x R, rescale
    it into the real-camera normalized coordinates via the intrinsic ratio, and
    re-decompose E by SVD to obtain the corrected rotation/translation direction;
    finally scale translation by d* (Eq. 20).
    """
    K_real = np.asarray(K_real, dtype=np.float64)
    K_can = np.asarray(K_canonical, dtype=np.float64)
    R, t = pbvs_inverse(v_norm, lam)

    # essential matrix in canonical normalized coords
    E = _skew(t) @ R
    # map into real normalized coords: x_real = S x_can with S = K_real^-1 K_can
    S = np.linalg.inv(K_real) @ K_can
    E_real = S.T @ E @ S

    # re-decompose (Eq. 3-4): E = U diag(1,1,0) V^T
    U, _, Vt = np.linalg.svd(E_real)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    # Two-way rotation ambiguity in E-decomposition; pick the branch closest to
    # the Eq.18 pose (also makes this degrade to plain scale when K==canonical).
    cand = [U @ W @ Vt, U @ W.T @ Vt]
    R_real = min(cand, key=lambda Rc: np.linalg.norm(Rc - R))
    t_real = U[:, 2]                       # translation direction (up to sign/scale)
    if np.dot(t_real, t) < 0:              # resolve sign against original direction
        t_real = -t_real
    t_real = t_real * (np.linalg.norm(t) + 1e-12)

    # re-encode as a PBVS velocity (forward of Eq. 5, lam absorbed), then scale
    rotvec = _R_to_rotvec(R_real)
    w = -lam * rotvec
    vtr = -lam * (R_real.T @ t_real)
    v = np.concatenate([vtr, w])
    return denormalize_scale(v, d_star)


def _R_to_rotvec(R):
    tr = np.clip((np.trace(R) - 1) / 2, -1, 1)
    theta = np.arccos(tr)
    if theta < 1e-9:
        return np.zeros(3)
    rv = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return rv / (2 * np.sin(theta)) * theta
