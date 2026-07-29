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
        # Keep UINT8. As float32 this is 3.1MB/image: at 800 scenes (~4000 pairs)
        # Ic+Id alone would be ~25GB and the cache build OOMs a 31GB box.
        # precompute_backbone_features() scales to [0,1] per chunk on the GPU.
        arr = np.stack(lst)
        return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()

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
        # load_bproc_samples() hands us uint8 to keep host RAM bounded; the
        # PyBullet path (generate_samples) already yields floats in [0,1].
        if chunk.dtype == torch.uint8:
            chunk = chunk.float().div_(255.0)
        f = net.backbone(chunk)
        feats.append(f.to(store_dtype).cpu())
    return torch.cat(feats, dim=0)


def build_feature_cache(scene_dir, net, device, out_dir, val_frac=0.1,
                        policy=Policy.PBVS_Straight, intrinsic=None, batch=8,
                        seed=0):
    """STREAMING frozen-ViT feature cache -> on-disk .npy memmaps.

    Replaces the load-everything-then-torch.save path, which cannot scale: at
    7115 pairs that held Ic+Id (11.2GB uint8) plus Fc+Fd (11.2GB each fp16) for a
    ~28GB peak and was OOM-killed (exit 137) on a 31GB box -- and torch.save()
    additionally needs all 22.4GB resident to serialise. Projected 43GB at the
    next DAgger round, 59GB at the one after.

    Here peak RAM is ONE SCENE of images (~64MB) plus one GPU batch, regardless
    of dataset size. Also ~5x less backbone work for the desired image: every
    pair in a scene shares one desired view, so its features are computed once
    and broadcast instead of recomputed per pair.

    Fd IS DEDUPLICATED PER SCENE. Every pair in a scene shares one desired view,
    so storing it per-pair duplicated it ~16x: at 16251 pairs / 980 scenes that is
    25.6GB of redundancy and the cache (51.1GB) no longer fit the disk. Stored per
    scene it is 1.5GB, and training maps pair -> scene via meta["scene_id"].

    Layout:  <out_dir>/fc.npy   float16 [N, H16,W16,C]   per PAIR
             <out_dir>/fd.npy   float16 [S, H16,W16,C]   per SCENE (deduped)
             <out_dir>/meta.pt  vel_si, tPo_norm, tr, va, n_scenes, scene_id
    """
    import os, glob as _g
    from cns.utils.perception import CameraIntrinsic

    intr = intrinsic or CameraIntrinsic(512, 512, 512, 512, 256, 256)
    dummy, dz = np.zeros((1, 2)), np.ones(1)
    files = sorted(_g.glob(os.path.join(scene_dir, "scene_*.npz")))
    if not files:
        raise SystemExit(f"no scene_*.npz under {scene_dir}")

    # ---- pass 1: labels only. npz is lazy, so reading poses/wP does NOT
    # decompress the images -- this is cheap even for thousands of scenes.
    per_file, vsis, tpos, scene_id = [], [], [], []
    for fi, f in enumerate(files):
        z = np.load(f)
        poses, wP = z["poses"], z["wP"]
        rows = []
        for i in range(1, poses.shape[0]):
            _, (tpo, vsi) = supervisor_vel(policy, dummy, dz, dummy, dz, intr,
                                           poses[i], poses[0], wP)
            rows.append(len(vsis))
            vsis.append(vsi); tpos.append(tpo); scene_id.append(fi)
        per_file.append((f, rows))
    n = len(vsis)
    print(f"[feat] {n} pairs from {len(files)} scenes; streaming to {out_dir}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    probe = net.backbone(torch.zeros(1, 3, 512, 512, device=device))
    _, H16, W16, C = probe.shape
    del probe
    fc = np.lib.format.open_memmap(os.path.join(out_dir, "fc.npy"), mode="w+",
                                   dtype=np.float16, shape=(n, H16, W16, C))
    fd = np.lib.format.open_memmap(os.path.join(out_dir, "fd.npy"), mode="w+",
                                   dtype=np.float16, shape=(len(files), H16, W16, C))

    # ---- pass 2: one scene at a time ----
    net.eval()
    with torch.no_grad():
        for si, (f, rows) in enumerate(per_file):
            z = np.load(f)
            imgs = z["images"]                       # [M+1,H,W,3] uint8
            des = torch.from_numpy(imgs[0]).permute(2, 0, 1)[None].to(device)
            fdes = net.backbone(des.float().div_(255.0)).to(torch.float16).cpu().numpy()[0]
            for k0 in range(0, len(rows), batch):
                idx = rows[k0:k0 + batch]
                cur = torch.from_numpy(imgs[1 + k0:1 + k0 + len(idx)])
                cur = cur.permute(0, 3, 1, 2).to(device).float().div_(255.0)
                fcur = net.backbone(cur).to(torch.float16).cpu().numpy()
                for j, r in enumerate(idx):
                    fc[r] = fcur[j]
            fd[si] = fdes                            # one row per scene (deduped)
            if (si + 1) % 100 == 0:
                print(f"[feat]   {si+1}/{len(per_file)} scenes", flush=True)
    fc.flush(); fd.flush()

    nv = max(8, int(n * val_frac))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    torch.save({"vsi": torch.tensor(np.stack(vsis), dtype=torch.float32),
                "tPo_norm": torch.tensor(np.array(tpos), dtype=torch.float32),
                "tr": perm[nv:], "va": perm[:nv],
                "n_scenes": len(files), "shape": (n, H16, W16, C),
                "scene_id": torch.tensor(scene_id, dtype=torch.long)},
               os.path.join(out_dir, "meta.pt"))
    print(f"[feat] cached -> {out_dir} ({n} pairs, {(n+len(files))*H16*W16*C*2/1e9:.1f}GB on disk; "
          f"fd deduped {n}->{len(files)} rows)",
          flush=True)
    return n
