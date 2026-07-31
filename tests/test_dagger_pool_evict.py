"""Eviction invariants for cns.sim.dagger_pool.DaggerPool.

The failure mode worth a test: pairs address their goal image indirectly, via
scene_id into fd/id. Recycle a scene slot while a pair still points at it and
that pair trains against ANOTHER scene's goal -- no crash, no warning, just a
worse number. Every scene here is stamped with a unique pixel value so a
mismatched pair is detectable by comparison rather than by inspection.

Run: python tests/test_dagger_pool_evict.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cns.sim.dagger_pool as dp

H16 = W16 = 2
C = 3
H1 = W1 = 8
N_VIEW = 5              # 1 goal + 4 current -> 4 pairs per scene


def fake_backbone(x):
    """Shape-correct stand-in for RADIO: [B,3,H,W] -> [B,H16,W16,C], carrying
    the scene stamp through so feature rows can be checked too."""
    stamp = x[:, 0, 0, 0] * 255.0
    return stamp[:, None, None, None].expand(-1, H16, W16, C).clone()


def write_scene(d, name, stamp):
    """A scene whose every pixel is `stamp`, so pair/scene pairing is checkable."""
    imgs = np.full((N_VIEW, H1, W1, 3), stamp, dtype=np.uint8)
    poses = np.zeros((N_VIEW, 4, 4), dtype=np.float32)
    poses[:] = np.eye(4)
    poses[:, 2, 3] = np.linspace(1.0, 0.2, N_VIEW)
    np.savez(os.path.join(d, name), images=imgs, poses=poses,
             wP=np.zeros((8, 3), dtype=np.float32))


def check(pool, label):
    """Assert every invariant that eviction could break."""
    assert int(pool.pair_valid.sum()) == pool.n_pairs, f"{label}: pair count"
    assert int((pool.scene_seq >= 0).sum()) == pool.n_scenes, f"{label}: scene count"
    assert pool.n_pairs <= pool.reserve_pairs, f"{label}: pairs over reserve"
    assert pool.n_scenes <= pool.reserve_scenes, f"{label}: scenes over reserve"

    rows = pool.pair_valid.nonzero(as_tuple=True)[0]
    for r in rows.tolist():
        si = int(pool.scene_id[r])
        assert pool.scene_seq[si] >= 0, \
            f"{label}: live pair {r} -> dead scene slot {si}"
        # The stamp check: a recycled slot shows up here as a mismatch.
        assert int(pool.ic[r][0, 0, 0]) == int(pool.id[si][0, 0, 0]), \
            (f"{label}: pair {r} stamp {int(pool.ic[r][0, 0, 0])} != "
             f"scene {si} stamp {int(pool.id[si][0, 0, 0])}")

    s = pool.sample(8)
    if pool.n_pairs:
        assert len(s) == 8 and bool(pool.pair_valid[s].all()), f"{label}: sample dead row"
    else:
        assert len(s) == 0, f"{label}: sample from empty pool"


def main():
    # Isolate the eviction logic: supervisor_vel needs real geometry, and its
    # output is not what is under test.
    dp.supervisor_vel = lambda *a, **k: (None, (0.5, np.array(
        [0.3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)))

    root = tempfile.mkdtemp(prefix="dagger_evict_")
    cache, scenes = os.path.join(root, "pool"), os.path.join(root, "live")
    os.makedirs(scenes)
    try:
        # 20 pair slots / 6 scene slots against 4 pairs per scene: pairs bind
        # first at 5 scenes, so eviction is driven by pair pressure, as in the
        # real run (8000 pairs vs 1600 scene slots).
        pool = dp.DaggerPool(cache, H16, W16, C, H1=H1, W1=W1,
                             reserve_pairs=20, reserve_scenes=6, min_tv=0.015, min_rw=0.012)
        check(pool, "empty")
        assert pool.n_pairs == 0

        stamp = 0
        for rnd in range(6):
            for k in range(3):
                stamp += 1
                write_scene(scenes, f"scene_{rnd:02d}_{k:02d}.npz", stamp)
            added = pool.ingest(scenes, fake_backbone, "cpu", max_scenes=64, batch=2)
            check(pool, f"round {rnd}")
            print(f"round {rnd}: +{added} -> {pool.stats()}")

        # Past the reserve, the pool must sit AT capacity, not below it: the
        # original bug looked exactly like a pool that stopped growing.
        assert pool.n_pairs == 20, f"expected a full ring, got {pool.n_pairs}"
        assert pool.n_evicted > 0, "nothing was ever evicted"

        # FIFO: survivors are the newest scenes.
        live = (pool.scene_seq >= 0).nonzero(as_tuple=True)[0]
        seqs = pool.scene_seq[live]
        assert int(seqs.min()) == pool.seq - len(live), \
            f"survivors not contiguous-newest: {sorted(seqs.tolist())} of {pool.seq}"

        # Ingest is idempotent per file, and pruning removes what it consumed.
        assert pool.ingest(scenes, fake_backbone, "cpu") == 0, "re-ingested a scene"
        n_before = pool.n_pairs
        write_scene(scenes, "scene_99_00.npz", 250)
        assert pool.ingest(scenes, fake_backbone, "cpu", prune_ingested=True) == 4
        assert not os.path.exists(os.path.join(scenes, "scene_99_00.npz")), "not pruned"
        assert pool.n_pairs == n_before, "ring grew past reserve"
        check(pool, "after prune")

        # Reload: the mask has to survive a restart, or a resumed run trains on
        # dead slots.
        want = (pool.n_pairs, pool.n_scenes, pool.pair_valid.clone(),
                pool.scene_seq.clone(), pool.seq)
        again = dp.DaggerPool(cache, H16, W16, C, H1=H1, W1=W1,
                             reserve_pairs=20, reserve_scenes=6, min_tv=0.015, min_rw=0.012)
        assert (again.n_pairs, again.n_scenes) == want[:2], "counts lost on reload"
        assert bool((again.pair_valid == want[2]).all()), "pair mask lost on reload"
        assert bool((again.scene_seq == want[3]).all()), "scene order lost on reload"
        assert again.seq == want[4], "sequence counter lost on reload"
        check(again, "reloaded")

        # Old-format meta (contiguous prefix, no mask) must still load.
        legacy = os.path.join(root, "legacy")
        os.makedirs(legacy)
        for n in ("fc", "fd", "ic", "id"):
            shutil.copy(os.path.join(cache, n + ".npy"), legacy)
        torch.save({"n_pairs": 8, "n_scenes": 2, "vsi": torch.ones(8, 6),
                    "tPo_norm": torch.ones(8),
                    "scene_id": torch.tensor([0] * 4 + [1] * 4),
                    "ingested": []}, os.path.join(legacy, "meta.pt"))
        old = dp.DaggerPool(legacy, H16, W16, C, H1=H1, W1=W1,
                            reserve_pairs=20, reserve_scenes=6)
        assert (old.n_pairs, old.n_scenes) == (8, 2), \
            f"legacy meta: got {old.n_pairs}/{old.n_scenes}"
        assert old.seq == 2 and bool(old.pair_valid[:8].all()) \
            and not bool(old.pair_valid[8:].any()), "legacy prefix not widened"

        print(f"\nOK  ring held at {pool.n_pairs}/{pool.reserve_pairs} pairs, "
              f"{pool.n_scenes}/{pool.reserve_scenes} scenes, "
              f"{pool.n_evicted} scenes evicted")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
