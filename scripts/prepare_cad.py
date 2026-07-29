"""Prepare CAD parts for CNSv2 data generation.

    ~/blender/blender-4.2.1-linux-x64/blender --background --python \
        scripts/prepare_cad.py -- --input <file-or-dir> --out data/custom_parts \
        [--texture speckle|<image.png>] [--tris 30000] [--keep-scale]

Emits the GSO layout so `bproc_gen --meshes` finds parts with NO code change:

    <out>/<part_name>/meshes/model.obj + model.mtl + texture.png

WHY THIS EXISTS — a raw CAD export breaks the pipeline in four silent ways, and
only the first is about file format:

1. FORMAT. BlenderProc loads .obj/.ply/.stl/.dae/.fbx/.glb/.gltf/.usd*, but NOT
   .sldprt (proprietary — export from SolidWorks) and NOT .step/.stp (no importer
   in Blender). For STEP use the existing converter from the twin project,
   `ffs_twin_ws/isaac/convert_step.py` (STEP -> USD via Isaac's HOOPS converter;
   it needs the LD_PRELOAD + fresh-kit-subprocess workaround for an ODA symbol
   clash), then feed the .usd here.

2. SCALE. STEP/CAD is normally millimetres. bproc_gen assumes GSO-normalized
   meshes and applies `set_scale(uniform(0.1, 0.3))`, so a 200 mm part imports as
   200 units = **200 metres**. We normalize max bbox extent to 1.0 so the random
   scale yields 10–30 cm parts, matching GSO.

3. ORIGIN. `bproc_gen` takes `wP` (object centres) from `o.get_location()`, and wP
   sets `tPo_norm`, which normalizes **every velocity label**
   (`supervisor_vel`: `vel_si[:3] /= tPo_norm`). CAD parts routinely carry an
   assembly origin far from the geometry, and nothing errors — the labels are just
   quietly wrong. We move the origin to the geometry bounds centre.

4. NO UVs / NO TEXTURE. This is the one that actually kills servo performance.
   A UV-less CAD mesh renders as flat metal and yields almost no ViT patch
   correspondence — the twin project hit exactly this ("UV-less instanced CAD mesh
   rendered flat metal -> ALIKED finds ~nothing"). CNSv2 is entirely
   correspondence-driven, and textureless objects are the paper's hardest case
   (Table II: SIFT+CNS fails on them outright). So we smart-UV-unwrap and bind a
   high-frequency texture. `--texture speckle` synthesises one; prefer passing a
   real image for sim-to-real work.

Also decimates (STEP tessellation can be millions of triangles, and Blender
already leaks ~80 MB/scene in bproc_gen) and recomputes outward normals.
"""
import os
import sys

import bpy
import numpy as np

MESH_EXT = {".obj", ".ply", ".stl", ".dae", ".fbx", ".glb", ".gltf",
            ".usd", ".usda", ".usdc"}
CAD_EXT = {".step", ".stp", ".sldprt", ".sldasm", ".iges", ".igs", ".x_t", ".x_b"}


def argv_after_ddash():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="mesh file, or a directory of them")
    ap.add_argument("--out", default="data/custom_parts")
    ap.add_argument("--tris", type=int, default=30000, help="decimate target triangles")
    ap.add_argument("--texture", default="speckle",
                    help="'speckle' to synthesise, or a path to an image")
    ap.add_argument("--tex-size", type=int, default=1024)
    ap.add_argument("--keep-scale", action="store_true",
                    help="skip unit-extent normalization (only if already ~unit)")
    return ap.parse_args(argv_after_ddash())


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for blk in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for item in list(blk):
            if item.users == 0:
                blk.remove(item)


def _try(*ops):
    """Blender 4.x renamed several importers (import_scene.obj -> wm.obj_import).
    Try each candidate so this works across 4.x point releases."""
    last = None
    for fn, kw in ops:
        try:
            path = fn.split(".")
            op = bpy.ops
            for p in path:
                op = getattr(op, p)
            op(**kw)
            return True
        except Exception as e:            # noqa: BLE001 - operator may not exist
            last = e
    raise RuntimeError(f"no working importer: {last}")


def import_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in CAD_EXT:
        raise SystemExit(
            f"\n{path}\n  '{ext}' is a native CAD format that Blender cannot read.\n"
            f"  .sldprt/.sldasm : export from SolidWorks (OBJ/STL directly if it can, else STEP)\n"
            f"  .step/.stp/.iges: convert first, e.g.\n"
            f"      python3 ~/ffs_twin_ws/isaac/convert_step.py --input {path} "
            f"--output part.usd\n"
            f"    then re-run this script on part.usd\n")
    if ext not in MESH_EXT:
        raise SystemExit(f"{path}: unsupported extension {ext}")
    if ext == ".obj":
        _try(("wm.obj_import", {"filepath": path}),
             ("import_scene.obj", {"filepath": path}))
    elif ext == ".stl":
        _try(("wm.stl_import", {"filepath": path}),
             ("import_mesh.stl", {"filepath": path}))
    elif ext == ".ply":
        _try(("wm.ply_import", {"filepath": path}),
             ("import_mesh.ply", {"filepath": path}))
    elif ext in (".glb", ".gltf"):
        _try(("import_scene.gltf", {"filepath": path}))
    elif ext == ".fbx":
        _try(("import_scene.fbx", {"filepath": path}))
    elif ext == ".dae":
        _try(("wm.collada_import", {"filepath": path}))
    else:                                                    # usd / usda / usdc
        _try(("wm.usd_import", {"filepath": path}))
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def join_meshes(objs):
    """CAD assemblies import as many parts; bproc_gen treats one file as one
    object and reads a single location, so collapse to one mesh."""
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def recentre_and_scale(obj, keep_scale):
    # (3) origin -> geometry bounds centre, then object to world origin, so
    # o.get_location() is a truthful object centre for wP/tPo_norm.
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    dims = np.array(obj.dimensions, float)
    if keep_scale:
        return dims, 1.0
    m = float(dims.max())
    if m <= 0:
        raise SystemExit("degenerate mesh (zero extent)")
    # (2) unit max extent -> bproc_gen's set_scale(0.1..0.3) gives 10-30cm parts
    f = 1.0 / m
    obj.scale = (f, f, f)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.context.view_layer.update()
    return dims, f


def decimate(obj, target_tris):
    bpy.ops.object.mode_set(mode="OBJECT")
    me = obj.data
    tris = sum(max(len(p.vertices) - 2, 0) for p in me.polygons)
    if tris <= target_tris:
        return tris, tris
    mod = obj.modifiers.new("dec", "DECIMATE")
    mod.ratio = max(min(target_tris / float(tris), 1.0), 1e-4)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    after = sum(max(len(p.vertices) - 2, 0) for p in obj.data.polygons)
    return tris, after


def fix_normals_and_uv(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    # (4) CAD has no UVs; smart project gives usable ones for a texture
    try:
        bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.02)
    except Exception:
        bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode="OBJECT")


def make_speckle_image(name, size, seed=0):
    """High-frequency noise so the frozen ViT has patch-level detail to match on.
    Flat CAD surfaces produce near-zero correspondence signal without this."""
    rng = np.random.RandomState(seed)
    base = rng.rand(size, size, 1)
    # mix scales: fine speckle + coarse blotches, so features exist at several
    # patch sizes (patches are 16 px of a 512 px render)
    coarse = rng.rand(size // 16, size // 16, 1).repeat(16, 0).repeat(16, 1)
    tint = rng.uniform(0.35, 0.95, size=(1, 1, 3))
    rgb = np.clip(0.55 * base + 0.45 * coarse, 0, 1) * tint
    img = bpy.data.images.new(name, width=size, height=size)
    px = np.concatenate([rgb, np.ones((size, size, 1))], axis=-1)
    img.pixels = px.astype(np.float32).ravel()
    return img


def assign_material(obj, tex_path, tex_size, seed):
    mat = bpy.data.materials.new("cad_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    tex = nt.nodes.new("ShaderNodeTexImage")
    if tex_path and tex_path != "speckle":
        if not os.path.isfile(tex_path):
            raise SystemExit(f"--texture {tex_path} not found")
        tex.image = bpy.data.images.load(tex_path)
    else:
        tex.image = make_speckle_image(f"speckle_{seed}", tex_size, seed)
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    # metallic CAD look kills contrast under HDRI; keep it dielectric and rough
    bsdf.inputs["Roughness"].default_value = 0.55
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.05
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return tex.image


def export_gso_layout(obj, out_root, name, image):
    d = os.path.join(out_root, name, "meshes")
    os.makedirs(d, exist_ok=True)
    # pack the texture next to the obj so the .mtl reference resolves anywhere
    tex_dst = os.path.join(d, "texture.png")
    image.filepath_raw = tex_dst
    image.file_format = "PNG"
    image.save()
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    objp = os.path.join(d, "model.obj")
    try:
        bpy.ops.wm.obj_export(filepath=objp, export_selected_objects=True,
                              export_materials=True, path_mode="COPY")
    except Exception:
        bpy.ops.export_scene.obj(filepath=objp, use_selection=True,
                                 use_materials=True, path_mode="COPY")
    return objp


def main():
    a = parse_args()
    files = []
    if os.path.isdir(a.input):
        for root, _, fs in os.walk(a.input):
            for f in fs:
                if os.path.splitext(f)[1].lower() in MESH_EXT | CAD_EXT:
                    files.append(os.path.join(root, f))
    else:
        files = [a.input]
    if not files:
        raise SystemExit(f"no mesh/CAD files under {a.input}")
    print(f"[cad] {len(files)} file(s) -> {a.out}", flush=True)

    ok = 0
    for i, f in enumerate(sorted(files)):
        name = os.path.splitext(os.path.basename(f))[0]
        name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        print(f"\n[cad] ({i+1}/{len(files)}) {f}", flush=True)
        clear_scene()
        try:
            objs = import_any(f)
            if not objs:
                print("  no mesh objects imported; skipping"); continue
            obj = join_meshes(objs)
            dims, sf = recentre_and_scale(obj, a.keep_scale)
            t0, t1 = decimate(obj, a.tris)
            fix_normals_and_uv(obj)
            img = assign_material(obj, a.texture, a.tex_size, seed=i)
            p = export_gso_layout(obj, a.out, name, img)
            print(f"  orig extent {dims.round(4)} (max {dims.max():.4f}) -> scale x{sf:.5g}")
            print(f"  tris {t0} -> {t1};  uv=smart;  tex={'speckle' if a.texture=='speckle' else a.texture}")
            print(f"  wrote {p}", flush=True)
            ok += 1
        except SystemExit as e:
            print(f"  SKIP: {e}")
        except Exception as e:                                    # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}")
    print(f"\nCAD_PREPARED ok={ok}/{len(files)} out={a.out}", flush=True)
    print(f"Use with:  blenderproc run cns/render/bproc_gen.py -- "
          f"--meshes {a.out} ...", flush=True)


if __name__ == "__main__":
    main()
