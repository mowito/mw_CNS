"""Why do objects render untextured through isaac_eval_server but textured through
the isaac_scene generator? (Sec. 6.19)

Renders ONE identical scene (fixed rng seed -> identical models, poses, lighting)
under conditions that isolate the two candidate causes, in a single Isaac startup:

  A  want_masks=False, warmup=2     the eval-server / collector configuration
  B  A + 120 extra settle frames    tests asynchronous texture/material streaming
  C  B + instance_segmentation      tests whether attaching the seg annotator is
                                    what forces material resolution

If B is textured and A is not, it is a settle-time problem (raise warmup).
If C is textured and A/B are not, attaching the annotator is doing the work.
If all three are untextured, the object materials are not being resolved at all
and the cause is in the USD reference, not in the render loop.

    ~/isaacsim/python.sh scripts/probe_albedo.py --out /tmp/albedo_probe.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/albedo_probe.npz")
    ap.add_argument("--usd", default="data/gso_usd")
    ap.add_argument("--hdri", default="data/hdri")
    ap.add_argument("--tex", default="data/cc_textures")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    from cns.render.isaac_scene import launch
    app = launch(headless=True, gpu=args.gpu)

    from cns.render.isaac_scene import IsaacSceneGen
    from cns.sim.pose_sampling import sample_desired

    # want_masks=False reproduces isaac_eval_server.py exactly.
    gen = IsaacSceneGen(args.usd, args.hdri, args.tex, rt_subframes=2,
                        want_masks=False)

    rng = np.random.RandomState(args.seed)
    gen.reset(rng)
    # Which models this scene actually used -- if a texture file is missing on
    # disk the render cannot show one, so record the paths for checking.
    used = [p.GetPath().pathString for p in
            gen.stage.GetPrimAtPath("/World/objects").GetChildren()]
    refs = []
    for p in gen.stage.GetPrimAtPath("/World/objects").GetChildren():
        try:
            for r in p.GetReferences() if hasattr(p, "GetReferences") else []:
                refs.append(str(r))
        except Exception:
            pass
    print(f"[probe] objects: {used}", flush=True)

    gen.warmup(2)
    pose = sample_desired(rng, gen.scene_center)
    # Fix exposure once so brightness is identical across A/B/C and cannot be
    # mistaken for a texture difference.
    gen.auto_expose(pose, rng)

    shots = {}
    a, _ = gen.render(pose, want_mask=False)
    shots["A_masksFalse_warm2"] = a
    print(f"[probe] A mean={a.mean():.1f} std={a.std():.1f}", flush=True)

    gen.warmup(120)
    b, _ = gen.render(pose, want_mask=False)
    shots["B_plus120settle"] = b
    print(f"[probe] B mean={b.mean():.1f} std={b.std():.1f}  "
          f"|B-A|={np.abs(b.astype(int)-a.astype(int)).mean():.2f}", flush=True)

    try:
        seg = gen._rep.AnnotatorRegistry.get_annotator("instance_segmentation")
        seg.attach(gen.rp)
        gen.warmup(8)
        c, _ = gen.render(pose, want_mask=False)
        shots["C_segAttached"] = c
        print(f"[probe] C mean={c.mean():.1f} std={c.std():.1f}  "
              f"|C-B|={np.abs(c.astype(int)-b.astype(int)).mean():.2f}", flush=True)
    except Exception as e:
        print(f"[probe] C skipped: {type(e).__name__}: {e}", flush=True)

    np.savez_compressed(args.out, **shots)
    print(f"[probe] wrote {args.out} with {list(shots)}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
