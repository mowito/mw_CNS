"""PyBullet-first data generation for CNSv2 (offline supervised samples).

Each sample = (current image I_c, desired image I_d, GT normalized velocity
vel_si, scene scale tPo_norm), where the GT comes from the reused CNS PBVS
supervisor. Objects are pybullet_data primitives (stand-ins for GSO); the
per-scene desired image is rendered once and reused across several current
views to amortize the CPU renderer.

Closed-loop DAgger resampling is a later layer; this offline set is enough for
the "loss converges" correctness gate (CNSv2_5090_SETUP.md Sec 6.1, check 1).
"""

import numpy as np
import torch

from cns.render.pybullet_scene import PyBulletScene
from cns.sim.supervisor import supervisor_vel, Policy


def generate_samples(n_samples, seed=0, currents_per_scene=4,
                     policy=Policy.PBVS_Straight):
    sc = PyBulletScene(seed=seed)
    Ics, Ids, vsis, tpos = [], [], [], []
    dummy, dz = np.zeros((1, 2)), np.ones(1)
    while len(Ics) < n_samples:
        sc.reset()
        tar = sc.sample_camera_pose()
        Id = sc.render(tar)
        for _ in range(currents_per_scene):
            if len(Ics) >= n_samples:
                break
            cur = sc.sample_camera_pose()
            Ic = sc.render(cur)
            _, (tpo, vsi) = supervisor_vel(
                policy, dummy, dz, dummy, dz, sc.intrinsic, cur, tar, sc.wP)
            Ics.append(Ic); Ids.append(Id); vsis.append(vsi); tpos.append(tpo)
    sc.close()

    def imgs(lst):
        arr = np.stack(lst)                          # [N,H,W,3] uint8
        return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().float() / 255.0

    return {
        "Ic": imgs(Ics),
        "Id": imgs(Ids),
        "vel_si": torch.tensor(np.stack(vsis), dtype=torch.float32),   # [N,6]
        "tPo_norm": torch.tensor(np.array(tpos), dtype=torch.float32), # [N]
    }


def load_bproc_samples(scene_dir, policy=Policy.PBVS_Straight, intrinsic=None):
    """Load BlenderProc-rendered scenes (from bproc_gen.py) into the same sample
    format as generate_samples(). Each scene .npz has images[M+1,H,W,3] (view 0 =
    desired), poses[M+1] (wcT, OpenCV cam-to-world), wP[.,3] (object centers)."""
    import os, glob
    from cns.utils.perception import CameraIntrinsic
    intr = intrinsic or CameraIntrinsic(512, 512, 512, 512, 256, 256)
    Ics, Ids, vsis, tpos = [], [], [], []
    dummy, dz = np.zeros((1, 2)), np.ones(1)
    for f in sorted(glob.glob(os.path.join(scene_dir, "scene_*.npz"))):
        z = np.load(f)
        imgs, poses, wP = z["images"], z["poses"], z["wP"]
        Id_img, tar = imgs[0], poses[0]
        for i in range(1, imgs.shape[0]):
            cur = poses[i]
            _, (tpo, vsi) = supervisor_vel(
                policy, dummy, dz, dummy, dz, intr, cur, tar, wP)
            Ics.append(imgs[i]); Ids.append(Id_img); vsis.append(vsi); tpos.append(tpo)

    def imgs_t(lst):
        arr = np.stack(lst)
        return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().float() / 255.0

    return {
        "Ic": imgs_t(Ics), "Id": imgs_t(Ids),
        "vel_si": torch.tensor(np.stack(vsis), dtype=torch.float32),
        "tPo_norm": torch.tensor(np.array(tpos), dtype=torch.float32),
    }


@torch.no_grad()
def precompute_backbone_features(net, images, device, batch=8, store_dtype=torch.float16):
    """Run the frozen ViT once over [N,3,H,W]; return CPU features [N,H16,W16,C]."""
    feats = []
    net.eval()
    for i in range(0, images.shape[0], batch):
        chunk = images[i:i + batch].to(device)
        f = net.backbone(chunk)
        feats.append(f.to(store_dtype).cpu())
    return torch.cat(feats, dim=0)
