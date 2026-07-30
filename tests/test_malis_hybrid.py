"""Verify the closed-form Malis Eq. 23 control law against the numeric inverse
of the full Eq. 21 interaction matrix.

The point of transcribing Malis rather than reconstructing a Jacobian is that his
L is upper triangular and has no singularity in the whole task space (Eq. 33), so
`hybrid_velocity` can use a closed-form inverse instead of a damped pseudo-inverse.
This test is what makes that claim checkable.

    python3 tests/test_malis_hybrid.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cns.models.hybrid_control import (
    A_matrix, A_inverse, L_vw, L_omega, interaction_matrix,
    hybrid_velocity, patch_to_normalized, should_use_hybrid,
)

rng = np.random.RandomState(0)


def test_A_inverse():
    for _ in range(200):
        x, y = rng.uniform(-1, 1, 2)
        A = A_matrix(x, y)
        assert np.allclose(A @ A_inverse(x, y), np.eye(3), atol=1e-12)
    print("ok  A_inverse is the exact inverse of A (Malis Eq. 16)")


def test_L_omega_identity():
    """Malis Eq. 24: L_w^-1 (theta*u) = theta*u, for any rotation vector."""
    worst = 0.0
    for _ in range(500):
        tu = rng.uniform(-1, 1, 3)
        tu = tu / np.linalg.norm(tu) * rng.uniform(1e-3, 3.0)   # |theta| < pi
        Lw = L_omega(tu)
        worst = max(worst, np.abs(np.linalg.solve(Lw, tu) - tu).max())
    assert worst < 1e-8, worst
    print(f"ok  L_omega^-1 (theta u) == theta u  (max err {worst:.2e})")


def test_control_law_matches_numeric_inverse():
    """v = -lam L^-1 e computed numerically must equal the closed-form path."""
    worst = 0.0
    for _ in range(500):
        x, y = rng.uniform(-0.4, 0.4, 2)
        Z = rng.uniform(0.4, 2.0)
        tu = rng.uniform(-1, 1, 3)
        tu = tu / np.linalg.norm(tu) * rng.uniform(1e-3, 2.5)
        e_img = rng.uniform(-0.3, 0.3, 3)
        lam = rng.uniform(0.2, 2.0)

        # Reference: full Eq. 21 matrix, numerically inverted.
        L = interaction_matrix(x, y, Z, tu)
        e = np.concatenate([e_img, tu])
        v_ref = -lam * np.linalg.solve(L, e)

        # Closed form used by hybrid_velocity (Eq. 23 + Eq. 24).
        omega = -lam * tu
        v = -lam * Z * (A_inverse(x, y) @ (e_img - L_vw(x, y) @ tu))
        v_closed = np.concatenate([v, omega])

        worst = max(worst, np.abs(v_ref - v_closed).max())
    assert worst < 1e-8, worst
    print(f"ok  closed form == -lam L^-1 e  (max err {worst:.2e})")


def test_no_singularity_in_task_space():
    """Malis Eq. 33: det(L) != 0 everywhere in the workspace. Sweep it."""
    worst_cond = 0.0
    for _ in range(2000):
        x, y = rng.uniform(-1.0, 1.0, 2)
        Z = rng.uniform(0.2, 5.0)
        tu = rng.uniform(-1, 1, 3)
        tu = tu / np.linalg.norm(tu) * rng.uniform(1e-4, 3.10)
        L = interaction_matrix(x, y, Z, tu)
        assert abs(np.linalg.det(L)) > 1e-12, (x, y, Z, tu)
        worst_cond = max(worst_cond, np.linalg.cond(L))
    print(f"ok  det(L) != 0 over 2000 samples (worst cond {worst_cond:.1f})")


def test_zero_error_gives_zero_velocity():
    """At the goal the twist must vanish -- a bias here stalls the closed loop."""
    g = np.array([16.0, 16.0])
    v = hybrid_velocity(np.zeros(3), g, g, R_dc=np.eye(3), t_dc=np.zeros(3), Z=1.0)
    assert np.abs(v).max() < 1e-12, v
    print("ok  zero pose error -> zero twist")


def test_pure_translation_direction():
    """Camera offset in +x should command motion that reduces the image error."""
    # Desired gravity centre at image centre, current one to its right.
    dXg = np.array([15.5, 15.5])          # -> normalized (0,0) for canonical K
    cXg = np.array([19.5, 15.5])
    assert np.allclose(patch_to_normalized(dXg), [0.0, 0.0])
    v = hybrid_velocity(np.zeros(3), cXg, dXg, Z=1.0, lam=1.0)
    # x error is positive; with A^-1 = -I on the (x,y) block the commanded v_x is
    # positive, translating the camera so the feature moves back toward centre.
    assert v[0] > 0 and abs(v[1]) < 1e-12, v
    assert np.abs(v[3:]).max() < 1e-12, v
    print(f"ok  pure image-x error -> v = {np.round(v, 4)}")


def test_switch_threshold():
    """Paper Sec. G: hybrid while ||cXg - dXg|| > 0.1*sqrt(N16); N16=1024 -> 3.2."""
    base = np.array([16.0, 16.0])
    assert should_use_hybrid(base + np.array([3.3, 0.0]), base)
    assert not should_use_hybrid(base + np.array([3.1, 0.0]), base)
    print("ok  switch threshold is 3.2 patch units at N16=1024")



def test_eq24_ground_truth_convergence():
    """Paper Eq. 24 closed loop with GROUND-TRUTH inputs must converge from
    CNS-v1 initial poses -- pose-space only, the law's own sanity gate."""
    from scipy.spatial.transform import Rotation as SR
    from cns.models.hybrid_control import hybrid_velocity_eq24
    from cns.sim.pose_sampling import sample_desired, sample_initial

    def gravity_patch(wcT, wP, fx=512.0, cx=256.0):
        Rw, t = wcT[:3, :3], wcT[:3, 3]
        pc = (wP - t) @ Rw
        u = fx * pc[:, 0] / pc[:, 2] + cx
        v = fx * pc[:, 1] / pc[:, 2] + cx
        return np.array([u.mean(), v.mean()]) / 16.0 - 0.5

    rng2 = np.random.RandomState(7)
    dt, ok = 1.0 / 50.0, 0
    for ep in range(20):
        wP = rng2.uniform(-0.15, 0.15, (4, 3)); wP[:, 2] = np.abs(wP[:, 2]) * 0.3
        tar = sample_desired(rng2, wP.mean(0))
        cur = sample_initial(rng2, wP.mean(0))
        for k in range(1500):
            tcT = np.linalg.inv(tar) @ cur
            R_dc, t_dc = tcT[:3, :3], tcT[:3, 3]
            te = np.linalg.norm(t_dc)
            re = np.degrees(np.linalg.norm(SR.from_matrix(R_dc).as_rotvec()))
            if te < 0.002 and re < 1.0:
                break
            th_u = SR.from_matrix(R_dc).as_rotvec()
            t_c = -R_dc.T @ t_dc
            vel = hybrid_velocity_eq24(t_c, th_u, gravity_patch(cur, wP),
                                       gravity_patch(tar, wP), Z=1.0, lam=1.0)
            dR = SR.from_rotvec(vel[3:] * dt).as_matrix()
            dT = np.eye(4); dT[:3, :3] = dR; dT[:3, 3] = vel[:3] * dt
            cur = cur @ dT
        ok += (te < 0.002 and re < 1.0)
    assert ok == 20, f"eq24 GT closed loop: {ok}/20 converged"
    print(f"ok  paper Eq. 24 law converges 20/20 with ground-truth inputs")


if __name__ == "__main__":
    test_A_inverse()
    test_L_omega_identity()
    test_control_law_matches_numeric_inverse()
    test_no_singularity_in_task_space()
    test_zero_error_gives_zero_velocity()
    test_pure_translation_direction()
    test_switch_threshold()
    test_eq24_ground_truth_convergence()
    print("\nALL MALIS HYBRID TESTS PASSED")
