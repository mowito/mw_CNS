"""Persistent IsaacSim render server -- drop-in replacement for bproc_eval_server.

IsaacSim's bundled interpreter has no torch (checked: python 3.12.13, numpy 2.3.1,
no torch), and the scene must stay resident between renders, so the policy cannot
live in this process. Same split BlenderProc needed, and DELIBERATELY the same
file protocol, so the existing clients work against Isaac with no changes:

    eval_servo_bproc.py   closed-loop eval in the training domain (gate 5)
    collect_dagger.py     DAgger rollouts (Fig. 3 "Simulation Process #2")

Protocol (identical to cns/render/bproc_eval_server.py):

    server -> READY                          server is up
    server -> ep{i}_meta.npz (+ .ok)         tar, cur0, wP, desired
    client -> ep{i}_req{k}.npy (+ .ok)       4x4 OpenCV cam-to-world to render
    server -> ep{i}_resp{k}.npy (+ .ok)      uint8 HxWx3 render of that pose
    client -> ep{i}_end                      end this episode
    server -> SERVER_DONE

Scene building and pose sampling come from the SAME code the offline generator
uses (IsaacSceneGen + cns.sim.pose_sampling), which is the whole point: the
bproc server once carried its own copy of an older sampler and every rollout and
eval silently ran on a different pose distribution than training.

    ~/isaacsim/python.sh cns/render/isaac_eval_server.py \
        --workdir /tmp/servo_ipc --episodes 20 \
        --usd data/gso_usd --hdri data/hdri --tex data/cc_textures
"""
import argparse, os, sys, time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))


def touch(path):
    with open(path, "w") as f:
        f.write("1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--usd", default="data/gso_usd")
    ap.add_argument("--hdri", default="data/hdri")
    ap.add_argument("--tex", default="data/cc_textures")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rt-subframes", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no-auto-expose", action="store_true",
                    help="serve raw lighting; DIFFERENT from the training domain")
    # Basin characterization: start episodes at a BOUNDED pose error instead of
    # the full CNS-v1 initial distribution. Measures the convergence basin of the
    # raw policy separately from the far-field/matching problems.
    ap.add_argument("--start-te", type=float, default=0.0,
                    help=">0: initial pose is a perturbation of the desired pose "
                         "with translation error up to this (metres)")
    ap.add_argument("--start-re", type=float, default=30.0,
                    help="rotation error cap (deg) when --start-te is set")
    args = ap.parse_args()

    from cns.render.isaac_scene import launch
    app = launch(headless=True, gpu=args.gpu)

    from cns.render.isaac_scene import IsaacSceneGen
    from cns.sim.pose_sampling import sample_desired, sample_initial
    from cns.sim.pose_perturb import perturb_pose

    W = args.workdir
    os.makedirs(W, exist_ok=True)
    gen = IsaacSceneGen(args.usd, args.hdri, args.tex,
                        rt_subframes=args.rt_subframes, want_masks=False)
    rng = np.random.RandomState(args.seed)

    def render_pose(wcT):
        rgb, _ = gen.render(wcT, want_mask=False)
        return rgb

    touch(os.path.join(W, "READY"))
    print("[server] READY", flush=True)

    for ep in range(args.episodes):
        # ---- fresh scene, built by the SAME generator the training set used ----
        wP = gen.reset(rng)
        gen.warmup(args.warmup)
        tar = sample_desired(rng, gen.scene_center)
        # MATCH THE TRAINING DOMAIN. isaac_scene.py's generator auto-exposes every
        # scene against its desired view, so training images sit at mean ~126 with
        # only 1.1% outside [25,215]. Serving raw random dome intensity here would
        # evaluate the policy on a brightness distribution it never saw -- exactly
        # the "evaluate in the domain you trained in" failure of Sec. 6.9, which
        # previously cost hours chasing a policy bug that was a renderer mismatch.
        if not args.no_auto_expose:
            gen.auto_expose(tar, rng)
        if args.start_te > 0:
            cur0, _te0, _re0 = perturb_pose(tar, rng, te_min=args.start_te * 0.3,
                                            te_max=args.start_te,
                                            re_min=args.start_re * 0.3,
                                            re_max=args.start_re)
        else:
            cur0 = sample_initial(rng, gen.scene_center)
        desired = render_pose(tar)
        np.savez(os.path.join(W, f"ep{ep}_meta.npz"),
                 tar=tar, cur0=cur0, wP=wP, desired=desired)
        touch(os.path.join(W, f"ep{ep}_meta.ok"))
        print(f"[server] ep {ep} scene ready ({len(wP)} objs)", flush=True)

        # ---- serve render requests until the client ends the episode ----
        k = 0
        t_ep = last = time.time()
        while True:
            if os.path.exists(os.path.join(W, f"ep{ep}_end")):
                break
            req = os.path.join(W, f"ep{ep}_req{k}.npy")
            if os.path.exists(req) and os.path.exists(req + ".ok"):
                wcT = np.load(req)
                img = render_pose(wcT)
                np.save(os.path.join(W, f"ep{ep}_resp{k}.npy"), img)
                touch(os.path.join(W, f"ep{ep}_resp{k}.npy.ok"))
                k += 1
                last = time.time()
                continue
            if time.time() - last > args.timeout:
                print(f"[server] timeout waiting for client on ep {ep}", flush=True)
                break
            time.sleep(0.002)      # Isaac renders in ~20-90ms; poll faster than bproc
        el = time.time() - t_ep
        # Sweep this episode's files: clients delete per-step req/resp as they
        # consume them, but the meta/end markers and any step files orphaned by a
        # client crash would otherwise accumulate for the life of the server.
        for fn in os.listdir(W):
            if fn.startswith(f"ep{ep}_"):
                try:
                    os.remove(os.path.join(W, fn))
                except OSError:
                    pass
        print(f"[server] ep {ep} served {k} renders in {el:.1f}s "
              f"({1000*el/max(k,1):.0f} ms/render incl. IPC)", flush=True)

    touch(os.path.join(W, "SERVER_DONE"))
    print("SERVER_DONE", flush=True)
    app.close()


if __name__ == "__main__":
    main()
