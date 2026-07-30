"""Growable feature cache for CONCURRENT DAgger -- paper Fig. 3.

Fig. 3 runs two sampling processes at the same time and one training process:

    Simulation #1 (uniform, offline)  --,
                                        +--> Database {a, a*} --> Training
    Simulation #2 (DAgger, uses          |         ^
        the CURRENT policy's action)  ---'         |
                    Synchronize Weights Periodically

The 4060 pipeline instead ran DAgger as a PHASE 2: freeze a checkpoint, roll out,
save, retrain from scratch on the union. That is a different algorithm, and it
failed in a specific measurable way. Because expert convergence is geometric,
most steps of a converging trajectory sit near the goal (79.4% of round-0
rollouts had ||vel_si||<0.5), so aggregating whole rounds onto a far-heavy
uniform set drifted the training mix 1.8% -> 41% -> 56.3% near-goal, and the
closed-loop TE ratio got WORSE (1.350 -> 1.569) even while offline l_dir improved
64%: the model gained near-goal skill and lost the far field, which is where every
episode starts. Running both samplers concurrently is what keeps the database
balanced, and `frac` below makes that mix explicit rather than accidental.

This class owns the DAgger half of the database. It is a SEPARATE set of memmaps
from the uniform cache so the uniform half is never rewritten, pre-allocated to a
fixed reserve so rows can be filled in place while training reads them.

Layout (mirrors build_feature_cache so nothing new has to be learned):
    <dir>/fc.npy  float16 [R, H16,W16,C]  per PAIR
    <dir>/fd.npy  float16 [S, H16,W16,C]  per SCENE
    <dir>/ic.npy  uint8   [R, H,W,3]      per PAIR
    <dir>/id.npy  uint8   [S, H,W,3]      per SCENE
    <dir>/meta.pt n_pairs, n_scenes, vsi, tPo_norm, scene_id, ingested
"""
import glob
import os

import numpy as np
import torch

from cns.sim.supervisor import supervisor_vel, Policy


class DaggerPool:
    def __init__(self, cache_dir, H16, W16, C, H1=512, W1=512,
                 reserve_pairs=20000, reserve_scenes=4000, intrinsic=None,
                 min_vel=0.05):
        """min_vel: reject pairs whose ||vel_si|| is below this.

        Belt-and-braces against the geometric pile-up described above. The
        collector already log-subsamples its states, but this pool is fed by
        whatever lands in a directory, so the unlearnable band is rejected here
        too. 0.05 is Sec. 6.3's recommendation: it keeps the useful near-goal band
        (~1.2-6 patches) and drops what one 16px patch cannot resolve.
        """
        from cns.utils.perception import CameraIntrinsic
        self.dir = cache_dir
        self.min_vel = float(min_vel)
        self.n_rejected = 0
        self.intr = intrinsic or CameraIntrinsic(W1, H1, 512, 512, W1 // 2, H1 // 2)
        os.makedirs(cache_dir, exist_ok=True)
        self.meta_p = os.path.join(cache_dir, "meta.pt")

        def mm(name, dtype, shape):
            p = os.path.join(cache_dir, name)
            mode = "r+" if os.path.exists(p) else "w+"
            if mode == "r+":
                return np.load(p, mmap_mode="r+")
            return np.lib.format.open_memmap(p, mode="w+", dtype=dtype, shape=shape)

        self.fc = mm("fc.npy", np.float16, (reserve_pairs, H16, W16, C))
        self.fd = mm("fd.npy", np.float16, (reserve_scenes, H16, W16, C))
        self.ic = mm("ic.npy", np.uint8, (reserve_pairs, H1, W1, 3))
        self.id = mm("id.npy", np.uint8, (reserve_scenes, H1, W1, 3))
        self.reserve_pairs = self.fc.shape[0]
        self.reserve_scenes = self.fd.shape[0]

        if os.path.exists(self.meta_p):
            m = torch.load(self.meta_p, map_location="cpu")
            self.n_pairs = int(m["n_pairs"])
            self.n_scenes = int(m["n_scenes"])
            self.vsi = m["vsi"]
            self.tpo = m["tPo_norm"]
            self.scene_id = m["scene_id"]
            self.ingested = set(m.get("ingested", []))
        else:
            self.n_pairs = self.n_scenes = 0
            self.vsi = torch.zeros(0, 6)
            self.tpo = torch.zeros(0)
            self.scene_id = torch.zeros(0, dtype=torch.long)
            self.ingested = set()

    # ------------------------------------------------------------------
    def full(self):
        return (self.n_pairs >= self.reserve_pairs
                or self.n_scenes >= self.reserve_scenes)

    def save_meta(self):
        torch.save({"n_pairs": self.n_pairs, "n_scenes": self.n_scenes,
                    "vsi": self.vsi, "tPo_norm": self.tpo,
                    "scene_id": self.scene_id,
                    "ingested": sorted(self.ingested)}, self.meta_p)

    @torch.no_grad()
    def ingest(self, scene_dir, backbone, device, policy=Policy.PBVS_Straight,
               max_scenes=64, batch=8, prune_ingested=False):
        """Encode any scene_*.npz not yet seen. Returns the number of new pairs.

        Cheap enough to call from inside the training loop: one scene is ~7 views,
        so ~7 backbone forwards (~0.1 s), against a collector that produces a scene
        every few seconds at worst.
        """
        files = sorted(glob.glob(os.path.join(scene_dir, "scene_*.npz")))
        new = [f for f in files if os.path.basename(f) not in self.ingested]
        if not new:
            return 0
        new = new[:max_scenes]

        dummy, dz = np.zeros((1, 2)), np.ones(1)
        added_pairs = 0
        vsis, tpos, sids = [], [], []
        for f in new:
            if self.full():
                print(f"[dagger] pool FULL ({self.n_pairs}/{self.reserve_pairs} pairs, "
                      f"{self.n_scenes}/{self.reserve_scenes} scenes); "
                      f"stopping ingest", flush=True)
                break
            try:
                z = np.load(f)
                imgs, poses, wP = z["images"], z["poses"], z["wP"]
            except Exception as e:
                # A scene still being written by the collector will read as
                # truncated; skip it WITHOUT marking it ingested so the next pass
                # picks it up once complete.
                print(f"[dagger] skip {os.path.basename(f)} ({type(e).__name__})", flush=True)
                continue
            n_view = min(imgs.shape[0], poses.shape[0])
            if n_view < 2:
                self.ingested.add(os.path.basename(f))
                continue

            si = self.n_scenes
            des = torch.from_numpy(imgs[0]).permute(2, 0, 1)[None].to(device)
            self.fd[si] = backbone(des.float().div_(255.0)).to(
                torch.float16).cpu().numpy()[0]
            self.id[si] = imgs[0]
            self.n_scenes += 1

            for k0 in range(1, n_view, batch):
                sl = slice(k0, min(k0 + batch, n_view))
                room = self.reserve_pairs - self.n_pairs
                raw = imgs[sl][:room]
                if len(raw) == 0:
                    break
                cur = torch.from_numpy(raw).permute(0, 3, 1, 2).to(device)
                fcur = backbone(cur.float().div_(255.0)).to(torch.float16).cpu().numpy()
                for j in range(len(raw)):
                    _, (tpo, vsi) = supervisor_vel(
                        policy, dummy, dz, dummy, dz, self.intr,
                        poses[sl.start + j], poses[0], wP)
                    if float(np.linalg.norm(vsi)) < self.min_vel:
                        self.n_rejected += 1
                        continue
                    r = self.n_pairs
                    self.fc[r] = fcur[j]
                    self.ic[r] = raw[j]
                    vsis.append(vsi); tpos.append(tpo); sids.append(si)
                    self.n_pairs += 1
                    added_pairs += 1
            self.ingested.add(os.path.basename(f))
            # The pool's memmaps now hold everything this scene contributes, so
            # the raw npz is redundant storage. The first 40k run kept both:
            # 19GB of pool + 16GB of already-ingested scenes in data/dagger_live,
            # on the same disk the run eventually filled.
            if prune_ingested:
                try:
                    os.remove(f)
                except OSError:
                    pass

        if added_pairs:
            self.vsi = torch.cat([self.vsi,
                                  torch.tensor(np.stack(vsis), dtype=torch.float32)])
            self.tpo = torch.cat([self.tpo,
                                  torch.tensor(np.array(tpos), dtype=torch.float32)])
            self.scene_id = torch.cat([self.scene_id,
                                       torch.tensor(sids, dtype=torch.long)])
            self.fc.flush(); self.fd.flush(); self.ic.flush(); self.id.flush()
            self.save_meta()
        return added_pairs

    def stats(self):
        if self.n_pairs == 0:
            return "empty"
        n = self.vsi.norm(dim=-1)
        return (f"{self.n_pairs} pairs / {self.n_scenes} scenes, "
                f"||vel_si|| min {n.min():.3f} median {n.median():.3f}, "
                f"near-goal(<0.5) {100 * (n < 0.5).float().mean():.1f}%, "
                f"rejected<{self.min_vel} {self.n_rejected}")
