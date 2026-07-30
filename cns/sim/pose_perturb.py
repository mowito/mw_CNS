"""Near-goal camera-pose perturbation for servo training data.

WHY THIS EXISTS
---------------
bproc_gen originally drew the desired pose and every current pose independently
from the same upper-hemisphere distribution, so a pair was almost never close
together: only ~5% of samples had ||vel_si|| < 0.5. A closed loop, though, spends
its final (decisive) steps entirely in that near-goal regime, and a policy
trained without it has no signal there -- measured translation direction cosine
0.51 (chance) below ||vel_si||=0.5 versus 0.90-0.97 further out. The servo then
contracts early and stalls or diverges near the goal; one episode was handed a
53mm error and pushed it to 1442mm.

This module generates current poses AS PERTURBATIONS of the desired pose so the
error spectrum the policy actually traverses is covered.

Two details that matter:
  * LOG-UNIFORM magnitude. Uniform sampling in [0, 0.3m] still puts almost
    nothing below 10mm; log-uniform gives even coverage per decade, which is
    what an exponentially-contracting controller needs.
  * CAMERA-FRAME perturbation (tar @ dT, right-multiply). This matches
    eval_servo.integrate(), which advances the camera as `wcT @ dT`, so training
    states are drawn from the same manifold the closed loop visits.

The translation floor is deliberately near the 5mm success gate rather than at
0: as ||vel_si|| -> 0 the regression target sigma_inv(y) = 1 + log(y) diverges
(y=1e-4 gives -8.2), which would let a handful of near-zero samples dominate the
L1 magnitude loss.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

# Defaults span the convergence funnel up to where the independent-sampling
# distribution already has coverage.
#
# TE_MIN IS ONE CORRESPONDENCE PATCH, NOT THE SUCCESS GATE. It used to be 0.003 m,
# chosen to sit "just under the 5mm success gate" -- but the coarse grid is
# 512/32 = 16 px, so at the working depth of ~0.7 m one patch is
# 16 * 0.7 / 512 = 21.9 mm. A 3 mm perturbation is 1/7 of a patch: the two views
# are nearly identical in feature space (measured paired cos(Fc,Fd) = 0.93 in the
# [0,0.05) ||vel_si|| bin) so the label is unlearnable, AND sigma_inv(y)=1+log(y)
# reaches -4.15 there against a median target of 1.49, so those samples dominated
# the L1 magnitude loss. That is the recorded cause of the failed run in
# PAPER_FAITHFUL_5090.md Sec. 8 ("1400 scenes, +near-goal (TE_MIN 3 mm),
# val l_dir 0.6748 -- failed"), and Sec. 6.3 says outright: do not set TE_MIN
# below ~1 patch at your working depth.
#
# Sub-millimetre final accuracy does NOT come from resolving sub-patch errors in
# one frame; it comes from closed-loop integration over the paper's 1500 steps.
TE_MIN, TE_MAX = 0.025, 0.30      # metres (0.025 ~= 1 patch at d* ~ 0.7 m)
RE_MIN, RE_MAX = 0.5, 30.0        # degrees


def sample_error_magnitudes(rng, te_min=TE_MIN, te_max=TE_MAX,
                            re_min=RE_MIN, re_max=RE_MAX):
    """Log-uniform (translation_m, rotation_deg) error magnitudes."""
    te = 10.0 ** rng.uniform(np.log10(te_min), np.log10(te_max))
    re = 10.0 ** rng.uniform(np.log10(re_min), np.log10(re_max))
    return float(te), float(re)


def _unit(rng):
    v = rng.normal(size=3)
    n = np.linalg.norm(v)
    return v / (n + 1e-12) if n > 1e-12 else np.array([1.0, 0.0, 0.0])


def perturb_pose(tar_wcT, rng, **kw):
    """A current pose near `tar_wcT` (OpenCV cam-to-world).

    Perturbs in the CAMERA frame so the result lies on the same manifold
    eval_servo.integrate() moves along. Returns (wcT, te_m, re_deg) -- the
    magnitudes are returned so callers can log/verify the realised spectrum.
    """
    te, re = sample_error_magnitudes(rng, **kw)
    dT = np.eye(4)
    dT[:3, :3] = R.from_rotvec(np.radians(re) * _unit(rng)).as_matrix()
    dT[:3, 3] = te * _unit(rng)
    return tar_wcT @ dT, te, re
