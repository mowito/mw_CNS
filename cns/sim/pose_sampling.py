"""CNS-v1-faithful camera pose sampling for CNSv2 data generation.

Wraps CNS v1's own `cns.sim.sampling.sample_camera_pose` with the concrete
parameters CNS v1 uses for its SIM training environment
(`cns/sim/environment.py:52-66`), because our bproc_gen sampler diverged from it
in three ways that matter:

  1. ELEVATION was symmetric 20-75 deg for both poses. CNS v1 makes the DESIRED
     pose near top-down (phi 70-90) and the INITIAL pose wide (phi 30-90). That
     asymmetry *is* the task: servo from an oblique view back to a canonical
     near-overhead view. Sampling both identically poses a different, less
     well-conditioned problem.
  2. NO IN-PLANE ROTATION. look_at_cv pins camera up to world +z, so roll was
     always canonical. CNS v1 perturbs the initial pose by up to drz=60 deg
     about the camera z axis (desired: 15 deg). The paper's real-world table
     calls out "large initial in-plane rotations" as exactly where competing
     methods fail -- a model that never sees roll cannot handle it.
  3. DISTANCE was scene-relative (0.35-3.0m from a frame-fill heuristic). CNS v1
     uses an absolute r in [0.5, 0.9] m, consistent with the paper's canonical
     d* = 1m scene scale.

Paper section H: "one uniformly samples current and desired pose pairs for
rendering in the upper hemisphere offline" -- phi > 0 throughout, which both the
CNS v1 ranges and this module respect.
"""
import numpy as np

from cns.sim.sampling import sample_camera_pose

# --- CNS v1 sim training env values (cns/sim/environment.py:52-66) ------------
DESIRED = dict(r_min=0.5, r_max=0.9, phi_min=70, phi_max=90,
               drz_max=15, dry_max=5, drx_max=5)
INITIAL = dict(r_min=0.5, r_max=0.9, phi_min=30, phi_max=90,
               drz_max=60, dry_max=10, drx_max=10)
# CNS v1 control/eval constants, for reference by callers
DT = 1.0 / 50.0        # 50 Hz -- our eval had been using 0.15s, 7.5x coarser
DIST_EPS = 0.002       # m
ANGLE_EPS = 1.0        # deg

# CNS v1's sampler builds poses about the world origin; scenes are built around
# a scene centre, so callers offset the translation.
_CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])


def _sample(cfg, rng):
    """CNS v1 sample_camera_pose uses the global numpy RNG; seed it from ours so
    generation stays reproducible from a single --seed."""
    st = np.random.get_state()
    np.random.seed(int(rng.randint(0, 2**31 - 1)))
    try:
        return sample_camera_pose(**cfg)
    finally:
        np.random.set_state(st)


def sample_desired(rng, scene_center=None):
    """Near-overhead canonical view (phi 70-90, roll +/-15 deg)."""
    T = _sample(DESIRED, rng)
    if scene_center is not None:
        T[:3, 3] += np.asarray(scene_center, float)
    return T


def sample_initial(rng, scene_center=None):
    """Wide oblique view (phi 30-90, roll up to +/-60 deg)."""
    T = _sample(INITIAL, rng)
    if scene_center is not None:
        T[:3, 3] += np.asarray(scene_center, float)
    return T
