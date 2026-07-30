"""Tune IsaacSim render settings: how few warmup frames and rt_subframes are safe?

Two settings trade speed against correctness:

  --rt-subframes  RTX accumulation per frame. Too low = noisy images.
  --warmup        throwaway frames after a scene change. Too low = the annotator
                  hands back the PREVIOUS SCENE's pixels, which would pair images
                  with the wrong poses and silently poison the dataset.

The second failure is the dangerous one: it is invisible in aggregate statistics.
This script measures both directly.

    ~/isaacsim/python.sh scripts/bench_isaac.py --usd data/gso_usd \
        --hdri data/hdri --tex data/cc_textures
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", default="data/gso_usd")
    ap.add_argument("--hdri", default="data/hdri")
    ap.add_argument("--tex", default="data/cc_textures")
    ap.add_argument("--trials", type=int, default=8)
    args = ap.parse_args()

    from cns.render.isaac_scene import launch
    app = launch(headless=True)
    from cns.render.isaac_scene import IsaacSceneGen
    from cns.sim.pose_sampling import sample_desired, sample_initial

    rng = np.random.RandomState(0)

    # ---------- part 1: noise vs rt_subframes, at a fixed pose ----------------
    print("\n=== noise vs rt_subframes (fixed scene+pose) ===", flush=True)
    gen = IsaacSceneGen(args.usd, args.hdri, args.tex, rt_subframes=64,
                        want_masks=False)
    gen.reset(rng)
    gen.warmup(16)
    pose = sample_desired(rng, np.zeros(3))
    gen.set_pose(pose)
    for _ in range(8):
        gen.step(subframes=64)
    ref = np.asarray(gen.ann_rgb.get_data())[..., :3].astype(np.float32)

    print(f"{'subframes':>10} {'ms/frame':>9} {'MAE vs ref':>11} {'PSNR dB':>8}")
    for sf in (1, 2, 4, 8, 16, 32):
        gen.set_pose(pose)
        gen.step(subframes=sf)                      # settle at this setting
        t0 = time.time()
        for _ in range(args.trials):
            gen.step(subframes=sf)
            img = np.asarray(gen.ann_rgb.get_data())[..., :3].astype(np.float32)
        ms = 1000 * (time.time() - t0) / args.trials
        mae = float(np.abs(img - ref).mean())
        mse = float(((img - ref) ** 2).mean())
        psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-9))
        print(f"{sf:>10} {ms:>9.1f} {mae:>11.3f} {psnr:>8.1f}", flush=True)

    # ---------- part 2: does a fresh scene leak the previous one? ------------
    # Build scene A, render it. Build scene B, render after N warmup frames, and
    # compare against a fully settled B. If warmup is too small the "B" frame
    # still looks like A -- that is the dataset-poisoning failure.
    print("\n=== scene-change staleness vs warmup ===", flush=True)
    print(f"{'warmup':>7} {'MAE vs settled B':>17} {'MAE vs A':>10}  verdict")
    for w in (0, 1, 2, 3, 4, 6, 8, 12):
        rngA = np.random.RandomState(100)
        gen.reset(rngA)
        gen.warmup(16)
        gen.set_pose(pose)
        for _ in range(4):
            gen.step(subframes=8)
        imgA = np.asarray(gen.ann_rgb.get_data())[..., :3].astype(np.float32)

        rngB = np.random.RandomState(777)
        gen.reset(rngB)
        gen.warmup(w)
        gen.set_pose(pose)
        gen.step(subframes=8)
        imgB_fast = np.asarray(gen.ann_rgb.get_data())[..., :3].astype(np.float32)
        # now let B settle properly and take the reference
        for _ in range(16):
            gen.step(subframes=8)
        imgB_ref = np.asarray(gen.ann_rgb.get_data())[..., :3].astype(np.float32)

        d_b = float(np.abs(imgB_fast - imgB_ref).mean())
        d_a = float(np.abs(imgB_fast - imgA).mean())
        # Healthy: close to settled B, far from A. Broken: the reverse.
        verdict = "OK" if d_b < d_a else "STALE -- leaks previous scene"
        print(f"{w:>7} {d_b:>17.2f} {d_a:>10.2f}  {verdict}", flush=True)

    print("\n[bench] DONE", flush=True)
    app.close()


if __name__ == "__main__":
    main()
