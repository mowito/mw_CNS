import blenderproc as bproc   # MUST be the very first statement (rewrites interpreter)
# BlenderProc photorealistic data generation for CNSv2 (fidelity path).
#
# Run INSIDE BlenderProc (not the training interpreter):
#   blenderproc run cns/render/bproc_gen.py -- \
#       --out data/bproc --scenes 60 --currents 4 [--meshes <GSO_dir>] --hdri data/hdri
#
# Renders scenes with HDRI lighting + PBR materials (paper Table I domain
# randomization), CNSv2 canonical intrinsic (fx=fy=512, cx=cy=256, 512x512).
# Each scene: one desired camera pose + several current poses on the upper
# hemisphere. Saves per-scene .npz with images + camera poses (wcT, OpenCV
# camera-to-world) + object centers (wP). A separate loader labels them with the
# PBVS supervisor -> (vel_si, tPo_norm), matching the PyBullet path's samples.
#
# Objects: GSO/.obj meshes from --meshes if given, else Blender primitives with
# randomized PBR materials as stand-ins (swap in GSO once downloaded).
import argparse, os, glob, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from cns.sim.pose_perturb import perturb_pose
from cns.sim.pose_sampling import sample_desired, sample_initial


def look_at_cv(eye, center, up=(0, 0, 1)):
    """OpenCV camera-to-world (x right, y down, +z toward center) -- for supervisor."""
    eye = np.asarray(eye, float); center = np.asarray(center, float)
    z = center - eye; z /= (np.linalg.norm(z) + 1e-9)
    up = np.asarray(up, float)
    if abs(np.dot(z, up)) > 0.98:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z); x /= (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    T = np.eye(4); T[:3, :3] = np.stack([x, y, z], axis=1); T[:3, 3] = eye
    return T


# OpenCV -> Blender/OpenGL camera axes (flip y and z)
_CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])


def rand_material(obj, rng):
    mat = obj.new_material("randmat")
    mat.set_principled_shader_value("Base Color",
                                    list(rng.uniform(0.05, 0.95, 3)) + [1.0])
    mat.set_principled_shader_value("Roughness", float(rng.uniform(0.1, 0.9)))
    mat.set_principled_shader_value("Metallic", float(rng.uniform(0.0, 0.8)))


def make_objects(meshes, n, rng):
    objs = []
    prims = ["CUBE", "SPHERE", "CYLINDER", "CONE", "MONKEY"]
    for _ in range(n):
        if meshes:
            path = meshes[rng.randint(len(meshes))]
            loaded = bproc.loader.load_obj(path)
            o = loaded[0]
            o.set_scale([rng.uniform(0.1, 0.3)] * 3)   # GSO meshes ~ real-metric; rescale
        else:
            o = bproc.object.create_primitive(prims[rng.randint(len(prims))])
            o.set_scale([rng.uniform(0.08, 0.18)] * 3)
        o.set_location([rng.uniform(-0.18, 0.18), rng.uniform(-0.18, 0.18),
                        rng.uniform(0.03, 0.12)])          # tighter cluster
        o.set_rotation_euler([rng.uniform(0, 6.28) for _ in range(3)])
        rand_material(o, rng)
        objs.append(o)
    return objs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--currents", type=int, default=4)
    ap.add_argument("--meshes", default=None, help="dir of GSO .obj meshes (optional)")
    ap.add_argument("--hdri", default="data/hdri")
    ap.add_argument("--seed", type=int, default=0)
    # Fraction of CURRENT poses drawn as near-goal perturbations of the desired
    # pose instead of independent hemisphere samples. Independent sampling alone
    # leaves only ~5% of pairs with ||vel_si||<0.5, and the policy is at chance
    # (translation cosine 0.51) in exactly that regime, so the closed loop stalls
    # or diverges near the goal. See cns/sim/pose_perturb.py.
    ap.add_argument("--near-frac", type=float, default=0.0,
                    help="0..1 fraction of current poses perturbed from the desired pose")
    # BlenderProc strips the "--" separator; our flags sit among Blender's own
    # argv, so pick them out with parse_known_args over the whole argv.
    argv = __import__("sys").argv
    tail = argv[argv.index("--") + 1:] if "--" in argv else argv[1:]
    args, _ = ap.parse_known_args(tail)

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    hdris = sorted(glob.glob(os.path.join(args.hdri, "*.hdr")))
    mesh_files = None
    if args.meshes:
        mesh_files = []
        for ext in ("*.obj", "*.ply", "*.glb", "*.gltf"):
            mesh_files += glob.glob(os.path.join(args.meshes, "**", ext), recursive=True)
        mesh_files = sorted(mesh_files)
        print(f"found {len(mesh_files)} mesh files under {args.meshes}", flush=True)

    bproc.init()
    K = np.array([[512, 0, 256], [0, 512, 256], [0, 0, 1]], float)
    bproc.camera.set_intrinsics_from_K_matrix(K, 512, 512)
    bproc.renderer.set_max_amount_of_samples(48)      # low sample count -> faster
    bproc.renderer.set_output_format(enable_transparency=False)

    for s in range(args.scenes):
        bproc.utility.reset_keyframes()
        for o in bproc.object.get_all_mesh_objects():
            o.delete()
        ground = bproc.object.create_primitive("PLANE"); ground.set_scale([2, 2, 1])
        rand_material(ground, rng)
        if hdris:
            bproc.world.set_world_background_hdr_img(hdris[rng.randint(len(hdris))])

        objs = make_objects(mesh_files, int(rng.randint(1, 7)), rng)
        centers = np.array([o.get_location() for o in objs], float)  # wP for supervisor
        # scene bounding sphere (from object bound boxes) for adaptive framing
        corners = np.concatenate([np.asarray(o.get_bound_box()) for o in objs], axis=0)
        lo, hi = corners.min(0), corners.max(0)
        scene_center = 0.5 * (lo + hi)
        scene_radius = float(0.5 * np.linalg.norm(hi - lo)) + 1e-3

        # desired pose (frame 0) + current poses (frames 1..M)
        poses_cv = []
        # CNS-v1-faithful sampling (cns/sim/pose_sampling.py). The previous
        # sampler was wrong in three ways: symmetric elevation 20-75 deg for both
        # poses (CNS v1: desired 70-90, initial 30-90 -- the asymmetry IS the
        # task), NO in-plane roll at all (CNS v1: initial up to 60 deg, and the
        # paper cites large initial in-plane rotation as where rivals fail), and
        # scene-relative distance 0.35-3.0m (CNS v1: absolute r 0.5-0.9m).
        def sample_pose_desired():
            return sample_desired(rng, scene_center)

        def sample_pose_initial():
            return sample_initial(rng, scene_center)
        # view 0 = desired; views 1..M = currents. A --near-frac share of the
        # currents are perturbations of the desired pose so the near-goal regime
        # the closed loop must land in is actually represented.
        desired = sample_pose_desired()
        poses_cv.append(desired)
        bproc.camera.add_camera_pose(desired @ _CV2GL)
        for j in range(args.currents):
            if rng.uniform() < args.near_frac:
                wcT, _te, _re = perturb_pose(desired, rng)
            else:
                wcT = sample_pose_initial()
            poses_cv.append(wcT)
            bproc.camera.add_camera_pose(wcT @ _CV2GL)

        data = bproc.renderer.render()
        colors = np.asarray(data["colors"], dtype=np.uint8)   # [M+1,H,W,3]
        np.savez_compressed(
            os.path.join(args.out, f"scene_{s:04d}.npz"),
            images=colors, poses=np.asarray(poses_cv), wP=centers)
        print(f"scene {s+1}/{args.scenes}: {colors.shape[0]} views", flush=True)

    print("BPROC_GEN_DONE")


if __name__ == "__main__":
    main()
