"""Merge freshly rendered scenes into the main dataset with safe renumbering.

bproc_gen.py always writes scene_0000.npz upward, so a second render collides
with the existing dataset. Copy (not move) into the next free index range,
validating each .npz first so a truncated render can't poison the feature cache.

    python3 merge_scenes.py data/bproc_extra data/bproc_train
"""
import glob, os, shutil, sys
import numpy as np


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data/bproc_extra"
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/bproc_train"
    existing = sorted(glob.glob(os.path.join(dst, "scene_*.npz")))
    used = {int(os.path.basename(p)[6:10]) for p in existing}
    offset = (max(used) + 1) if used else 0
    new = sorted(glob.glob(os.path.join(src, "scene_*.npz")))
    print(f"[merge] {len(existing)} existing in {dst}, {len(new)} candidates in {src}, "
          f"writing from index {offset:04d}", flush=True)

    ok, bad = 0, []
    for p in new:
        try:                                   # force a real read, not just open
            with np.load(p) as z:
                for k in z.files:
                    _ = z[k].shape
        except Exception as e:
            bad.append((p, repr(e)[:100]))
            continue
        tgt = os.path.join(dst, f"scene_{offset + ok:04d}.npz")
        if os.path.exists(tgt):                # never clobber
            raise SystemExit(f"[merge] ABORT: {tgt} already exists")
        shutil.copy2(p, tgt)
        ok += 1

    total = len(glob.glob(os.path.join(dst, "scene_*.npz")))
    print(f"MERGE_OK copied={ok} corrupt={len(bad)}")
    for p, e in bad[:5]:
        print(f"  corrupt: {p}  {e}", flush=True)
    print(f"MERGE_TOTAL={total}", flush=True)
    if ok == 0:
        raise SystemExit("[merge] ABORT: nothing copied")


if __name__ == "__main__":
    main()
