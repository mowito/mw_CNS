"""PyBullet-first scene renderer for the CNSv2 correctness gate.

Reuses CNS's Camera (cns/utils/perception.py) but renders headless (EGL if
available, else CPU TINY renderer). Objects are pybullet_data primitives as
stand-ins for GSO meshes -- swap the object pool for GSO once it finishes
downloading (see CNSv2_5090_SETUP.md Sec 5). Camera is the CNSv2 canonical
intrinsic: fx=fy=512, cx=cy=256, 512x512.

Conventions (match cns/sim + supervisor):
  wcT = camera-to-world (camera pose in world);  cwT = inv(wcT) = render extrinsic.
  OpenCV camera axes: +x right, +y down, +z forward (toward the scene).
"""

import numpy as np
import pybullet as p
import pybullet_data

from cns.utils.perception import CameraIntrinsic, Camera

# pybullet_data URDFs that carry a renderable visual mesh (probed at init).
_CANDIDATE_URDFS = [
    "duck_vhacd.urdf", "teddy_vhacd.urdf", "r2d2.urdf", "soccerball.urdf",
    "lego/lego.urdf", "cube_small.urdf", "sphere_small.urdf", "jenga/jenga.urdf",
]


def look_at(eye, center, up_world=(0, 0, 1)):
    """Build wcT (camera-to-world), OpenCV convention (+z toward center)."""
    eye = np.asarray(eye, float); center = np.asarray(center, float)
    z = center - eye
    z /= (np.linalg.norm(z) + 1e-9)
    up = np.asarray(up_world, float)
    if abs(np.dot(z, up)) > 0.98:                # avoid degeneracy near vertical
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z); x /= (np.linalg.norm(x) + 1e-9)   # right
    y = np.cross(z, x)                                     # down
    wcT = np.eye(4)
    wcT[:3, :3] = np.stack([x, y, z], axis=1)
    wcT[:3, 3] = eye
    return wcT


class PyBulletScene:
    def __init__(self, intrinsic=None, seed=0, near=0.05, far=6.0):
        self.intrinsic = intrinsic or CameraIntrinsic(512, 512, 512, 512, 256, 256)
        self.camera = Camera(self.intrinsic, near=near, far=far)
        self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        self.renderer = self._pick_renderer()
        self.rng = np.random.RandomState(seed)
        self.urdfs = self._probe_urdfs()
        self.body_ids = []
        self.scene_center = np.zeros(3)
        self.wP = None

    def _pick_renderer(self):
        try:
            p.loadPlugin("eglRendererPlugin", physicsClientId=self.cid)
            return p.ER_BULLET_HARDWARE_OPENGL
        except Exception:
            return p.ER_TINY_RENDERER

    def _probe_urdfs(self):
        ok = []
        for u in _CANDIDATE_URDFS:
            try:
                bid = p.loadURDF(u, physicsClientId=self.cid)
                p.removeBody(bid, physicsClientId=self.cid)
                ok.append(u)
            except Exception:
                pass
        if not ok:
            raise RuntimeError("no renderable pybullet_data URDFs found")
        return ok

    def reset(self, n_objs=None):
        """Scatter 1-6 randomly-posed, randomly-colored objects on a ground plane."""
        p.resetSimulation(physicsClientId=self.cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.loadURDF("plane.urdf", physicsClientId=self.cid)
        n = n_objs or int(self.rng.randint(1, 7))
        self.body_ids = []
        centers = []
        for _ in range(n):
            u = self.urdfs[self.rng.randint(len(self.urdfs))]
            xy = self.rng.uniform(-0.18, 0.18, size=2)       # tighter cluster
            z = self.rng.uniform(0.02, 0.12)
            scale = float(self.rng.uniform(0.8, 2.0))
            yaw = float(self.rng.uniform(0, 2 * np.pi))
            quat = p.getQuaternionFromEuler([0, 0, yaw])
            bid = p.loadURDF(u, [xy[0], xy[1], z], quat,
                             globalScaling=scale, physicsClientId=self.cid)
            # domain-randomize color (proxy for texture/material randomization)
            rgba = list(self.rng.uniform(0.2, 1.0, size=3)) + [1.0]
            for link in range(-1, p.getNumJoints(bid, physicsClientId=self.cid)):
                p.changeVisualShape(bid, link, rgbaColor=rgba, physicsClientId=self.cid)
            self.body_ids.append(bid)
            centers.append([xy[0], xy[1], z])
        self.wP = np.asarray(centers, float)                 # scene points for PBVS centroid
        # scene bounding sphere (from object AABBs) drives adaptive camera framing
        lo = np.array([p.getAABB(b, physicsClientId=self.cid)[0] for b in self.body_ids]).min(0)
        hi = np.array([p.getAABB(b, physicsClientId=self.cid)[1] for b in self.body_ids]).max(0)
        self.scene_center = 0.5 * (lo + hi)
        self.scene_radius = float(0.5 * np.linalg.norm(hi - lo)) + 1e-3
        return self

    def sample_camera_pose(self, fill=(0.45, 0.75), jitter=0.05):
        """Upper-hemisphere pose framing the scene to a target frame-fill fraction.
        A sphere of radius R at distance r projects to pixel radius ~ fx*R/r; we
        set r so that ~= fill*(W/2), i.e. r = 2R/fill (fx=W here)."""
        phi_fill = float(self.rng.uniform(*fill))
        r = float(np.clip(2.0 * self.scene_radius / phi_fill, 0.35, 3.0))
        theta = self.rng.uniform(0, 2 * np.pi)            # azimuth
        phi = self.rng.uniform(np.radians(20), np.radians(75))   # elevation
        eye = self.scene_center + r * np.array(
            [np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)])
        target = self.scene_center + self.rng.uniform(-jitter, jitter, size=3)
        return look_at(eye, target)

    def render(self, wcT):
        """wcT (camera-to-world) -> RGB uint8 [H,W,3]."""
        cwT = np.linalg.inv(wcT)
        gl_view = cwT.copy(); gl_view[2, :] *= -1
        gl_view = gl_view.flatten(order="F")
        res = p.getCameraImage(
            width=self.intrinsic.width, height=self.intrinsic.height,
            viewMatrix=gl_view, projectionMatrix=self.camera.gl_proj_matrix,
            renderer=self.renderer, physicsClientId=self.cid)
        H, W = self.intrinsic.height, self.intrinsic.width
        rgb = np.asarray(res[2], dtype=np.uint8).reshape(H, W, -1)[:, :, :3]
        return np.ascontiguousarray(rgb)

    def close(self):
        try:
            p.disconnect(physicsClientId=self.cid)
        except Exception:
            pass
