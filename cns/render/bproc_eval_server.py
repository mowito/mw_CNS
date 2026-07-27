import blenderproc as bproc   # MUST be the very first statement (rewrites interpreter)
# Persistent BlenderProc render server for closed-loop servo evaluation.
#
#   blenderproc run cns/render/bproc_eval_server.py -- \
#       --workdir /tmp/servo_ipc --episodes 20 --seed 777 \
#       --meshes <GSO_dir> --hdri data/hdri
#
# WHY a server: the trained policy only works in the photorealistic BlenderProc
# domain (a PyBullet-rendered eval scores WORSE than predicting the mean --
# frozen-RADIO feature cosine across the two renderers is only 0.36). So the
# closed loop must render in Blender. But Blender startup is seconds, and a
# 20-episode x 40-step eval needs ~800 renders, so we cannot re-launch per step;
# the scene has to stay resident. Torch also can't easily live in Blender's
# bundled python, so the POLICY runs in the normal CUDA interpreter and talks to
# this process over files.
#
# Scene construction is imported from bproc_gen so the eval domain is identical
# to the training domain (same HDRIs, PBR randomization, framing, intrinsics).
#
# Protocol (all inside --workdir):
#   server -> READY                        server is up
#   server -> ep{i}_meta.npz (+ .ok)       tar/cur0 poses, wP, desired image
#   client -> ep{i}_req{k}.npy             4x4 OpenCV cam-to-world to render
#   server -> ep{i}_resp{k}.npy (+ .ok)    uint8 HxWx3 render of that pose
#   client -> ep{i}_end                    episode finished, build the next one
#   server -> SERVER_DONE                  all episodes served
import argparse, os, glob, sys, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# blenderproc is already imported above, so re-importing it inside bproc_gen is a
# no-op -- lets us reuse the exact scene-building used to make the training set.
from cns.render.bproc_gen import look_at_cv, _CV2GL, rand_material, make_objects


def touch(p):
    with open(p, "w") as f:
        f.write("1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--meshes", default=None)
    ap.add_argument("--hdri", default="data/hdri")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="give up waiting on the client after this many seconds")
    argv = sys.argv
    tail = argv[argv.index("--") + 1:] if "--" in argv else argv[1:]
    args, _ = ap.parse_known_args(tail)

    W = args.workdir
    os.makedirs(W, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    hdris = sorted(glob.glob(os.path.join(args.hdri, "*.hdr")))
    mesh_files = None
    if args.meshes:
        mesh_files = []
        for ext in ("*.obj", "*.ply", "*.glb", "*.gltf"):
            mesh_files += glob.glob(os.path.join(args.meshes, "**", ext), recursive=True)
        mesh_files = sorted(mesh_files)
    print(f"[server] {len(mesh_files or [])} meshes, {len(hdris)} hdris", flush=True)

    bproc.init()
    K = np.array([[512, 0, 256], [0, 512, 256], [0, 0, 1]], float)
    bproc.camera.set_intrinsics_from_K_matrix(K, 512, 512)
    bproc.renderer.set_max_amount_of_samples(48)   # same as the training renders
    bproc.renderer.set_output_format(enable_transparency=False)

    def render_pose(wcT):
        bproc.utility.reset_keyframes()
        bproc.camera.add_camera_pose(wcT @ _CV2GL)
        data = bproc.renderer.render()
        return np.asarray(data["colors"], dtype=np.uint8)[0]

    touch(os.path.join(W, "READY"))
    print("[server] READY", flush=True)

    for ep in range(args.episodes):
        # ---- build a fresh scene, exactly like the training generator ----
        bproc.utility.reset_keyframes()
        for o in bproc.object.get_all_mesh_objects():
            o.delete()
        ground = bproc.object.create_primitive("PLANE"); ground.set_scale([2, 2, 1])
        rand_material(ground, rng)
        if hdris:
            bproc.world.set_world_background_hdr_img(hdris[rng.randint(len(hdris))])
        objs = make_objects(mesh_files, int(rng.randint(1, 7)), rng)
        centers = np.array([o.get_location() for o in objs], float)      # wP
        corners = np.concatenate([np.asarray(o.get_bound_box()) for o in objs], axis=0)
        lo, hi = corners.min(0), corners.max(0)
        scene_center = 0.5 * (lo + hi)
        scene_radius = float(0.5 * np.linalg.norm(hi - lo)) + 1e-3

        def sample_pose():
            fill = rng.uniform(0.45, 0.75)
            r = float(np.clip(2.0 * scene_radius / fill, 0.35, 3.0))
            th = rng.uniform(0, 6.28); ph = rng.uniform(np.radians(20), np.radians(75))
            eye = scene_center + r * np.array(
                [np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), np.sin(ph)])
            return look_at_cv(eye, scene_center + rng.uniform(-0.05, 0.05, 3))

        tar = sample_pose()
        cur0 = sample_pose()
        desired = render_pose(tar)
        np.savez(os.path.join(W, f"ep{ep}_meta.npz"),
                 tar=tar, cur0=cur0, wP=centers, desired=desired)
        touch(os.path.join(W, f"ep{ep}_meta.ok"))
        print(f"[server] ep {ep} scene ready ({len(objs)} objs)", flush=True)

        # ---- serve render requests until the client ends the episode ----
        k = 0
        last = time.time()
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
            time.sleep(0.02)
        print(f"[server] ep {ep} served {k} renders", flush=True)

    touch(os.path.join(W, "SERVER_DONE"))
    print("SERVER_DONE", flush=True)


if __name__ == "__main__":
    main()
