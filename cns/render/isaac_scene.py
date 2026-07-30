"""NVIDIA IsaacSim renderer for CNSv2 -- the renderer the paper actually uses.

Paper Fig. 3: "We use IsaacSim to render realistic images." This replaces the
BlenderProc/Cycles path, which measured 0.702 s per 512x512 render on a 4060 and
made the paper's dt = 1/50 s (273 expert steps per episode) economically
impossible. Measured here: ~33 ms/frame, i.e. ~21x faster.

SAMPLE CONTRACT -- identical to bproc_gen.py, so `load_bproc_samples()` and
`build_feature_cache()` need no changes and the pipeline stays renderer-agnostic:

    per-scene .npz:
      images [M+1, H, W, 3] uint8   view 0 = DESIRED, views 1..M = currents
      poses  [M+1, 4, 4]   float64  wcT, OpenCV camera-to-world
      wP     [n_obj, 3]    float64  object centres in world coords
      masks  [M+1, H, W]   uint32   instance segmentation (optional, Fig. 3)
      K      [3, 3]        float64  camera intrinsic (Fig. 3 lists it as input)

POSE SAMPLING IS IMPORTED, NEVER REIMPLEMENTED. `bproc_eval_server.py` once
carried its own copy of an older sampler, so an entire DAgger campaign silently
ran on the wrong distribution (d* 0.78-2.81 m and 0.0 deg in-plane roll, versus
0.47-0.93 m / 17.1 deg from the real sampler). Everything here comes from
`cns.sim.pose_sampling`.

Domain randomization (paper Sec. H): 1-6 GSO objects with randomized size and
pose, background textures AND materials from ambientCG CC0, ambient light from
Poly Haven HDRIs.

Usage -- must run under IsaacSim's own interpreter:

    ~/isaacsim/python.sh cns/render/isaac_scene.py \
        --out data/isaac_train --scenes 800 --currents 4 \
        --usd data/gso_usd --hdri data/hdri --tex data/cc_textures
"""
import argparse, glob, os, sys, time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

# OpenCV camera-to-world -> USD/OpenGL camera axes (USD looks down -Z with +Y up;
# OpenCV looks down +Z with +Y down). Same convention flip bproc_gen.py uses.
CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])

CANONICAL = dict(fx=512.0, fy=512.0, cx=256.0, cy=256.0, H=512, W=512)


def launch(headless=True, renderer="RaytracedLighting", gpu=0):
    """Create the SimulationApp. MUST happen before any omni/pxr import.

    multi_gpu is OFF deliberately. With both 5090s active Isaac reported
    "CUDA Peer Memory Copies from device[0] to device[1] is NOT possible as peer
    access is disabled -- Multi-GPU interoperability will be slower as copies
    across GPUs will go through main memory". Rendering fits comfortably on one
    card, and pinning it to one GPU is what frees the other for training, which is
    how Fig. 3's concurrent DAgger is meant to run.
    """
    from isaacsim import SimulationApp
    return SimulationApp({
        "headless": headless,
        "renderer": renderer,
        "multi_gpu": False,
        "active_gpu": int(gpu),
        "physics_gpu": int(gpu),
    })


class IsaacSceneGen:
    """Builds randomized table-top scenes and renders them at exact camera poses.

    The stage is created ONCE and only object prims / material texture paths are
    swapped per scene. Rebuilding the stage per scene is both slow and the kind of
    thing that leaks -- BlenderProc leaked ~80 MB/scene and hit 27 GB RSS by scene
    348, so this path is deliberately churn-free and `--report-rss` exists to
    prove it.
    """

    def __init__(self, usd_dir, hdri_dir=None, tex_dir=None, intr=None,
                 rt_subframes=8, want_masks=True):
        import omni.usd
        from pxr import UsdGeom, UsdLux, Gf, Sdf, UsdShade, Usd
        import omni.replicator.core as rep

        self._usd = omni.usd
        self._UsdGeom, self._UsdLux = UsdGeom, UsdLux
        self._Gf, self._Sdf, self._UsdShade, self._Usd = Gf, Sdf, UsdShade, Usd
        self._rep = rep

        k = dict(CANONICAL)
        if intr:
            k.update(intr)
        self.k = k
        self.rt_subframes = rt_subframes

        # convert_gso_to_usd.py writes <usd_dir>/<name>/<name>.usd -- one directory
        # per model, so each keeps its own materials/textures/texture.png. The flat
        # layout is also accepted for older conversions, but warned about, because
        # a flat dir means every model shares one texture file.
        self.models = sorted(glob.glob(os.path.join(usd_dir, "*", "*.usd")))
        flat = sorted(glob.glob(os.path.join(usd_dir, "*.usd")))
        if flat and not self.models:
            shared = os.path.join(usd_dir, "materials", "textures")
            if os.path.isdir(shared):
                raise SystemExit(
                    f"[isaac] REFUSING to use {usd_dir}: it is a FLAT conversion "
                    f"with a shared {shared}, so all models reference one texture "
                    f"(every GSO texture is named texture.png). Re-run "
                    f"scripts/convert_gso_to_usd.py to get one dir per model.")
            self.models = flat
        if not self.models:
            raise SystemExit(f"no *.usd under {usd_dir} -- run scripts/convert_gso_to_usd.py")
        self.hdris = sorted(glob.glob(os.path.join(hdri_dir, "*.hdr"))) if hdri_dir else []
        self.texes = []
        if tex_dir:
            # ambientCG layout: <Asset>/<Asset>_1K-JPG_Color.jpg
            self.texes = sorted(glob.glob(os.path.join(tex_dir, "*", "*_Color.jpg")))
        print(f"[isaac] {len(self.models)} USD models, {len(self.hdris)} HDRIs, "
              f"{len(self.texes)} background textures", flush=True)

        self.last_mask_info = None
        self._build_stage()
        self._attach_render(want_masks)

    # ---- stage -------------------------------------------------------------
    def _build_stage(self):
        UsdGeom, UsdLux, Gf = self._UsdGeom, self._UsdLux, self._Gf
        ctx = self._usd.get_context()
        ctx.new_stage()
        stage = ctx.get_stage()
        self.stage = stage
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        # Z-up: pose_sampling / look_at_cv both treat world +z as up.
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Xform.Define(stage, "/World/objects")

        self._make_ground()
        self.dome = UsdLux.DomeLight.Define(stage, "/World/dome")
        self.dome.CreateIntensityAttr(1000.0)
        # DomeLight's own xform lets us rotate the environment, which decorrelates
        # lighting direction from the HDRI choice.
        self._dome_rot = UsdGeom.Xformable(self.dome).AddRotateZOp()

        self._make_camera()

    def _make_ground(self):
        """A textured quad. Built as an explicit Mesh with UVs so background
        textures actually tile -- UsdGeom.Plane carries no st primvar."""
        UsdGeom, Gf = self._UsdGeom, self._Gf
        s = 3.0                       # +/-3 m: always fills the frame at r<=0.9 m
        mesh = UsdGeom.Mesh.Define(self.stage, "/World/ground")
        mesh.CreatePointsAttr([Gf.Vec3f(-s, -s, 0), Gf.Vec3f(s, -s, 0),
                               Gf.Vec3f(s, s, 0), Gf.Vec3f(-s, s, 0)])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
        mesh.SetNormalsInterpolation("vertex")
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", self._Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.varying)
        # 4 tiles across the quad; randomized per scene via _tile_scale.
        st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(4, 0), Gf.Vec2f(4, 4), Gf.Vec2f(0, 4)])
        self._ground_st = st
        self.ground = mesh
        self._ground_mat = self._make_textured_material("/World/looks/ground")
        self._UsdShade.MaterialBindingAPI(mesh).Bind(self._ground_mat["material"])

    def _make_textured_material(self, path):
        """UsdPreviewSurface + UvTexture. Returns the handles we mutate per scene."""
        UsdShade, Sdf, Gf = self._UsdShade, self._Sdf, self._Gf
        mat = UsdShade.Material.Define(self.stage, path)
        shader = UsdShade.Shader.Define(self.stage, path + "/surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        rough = shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
        rough.Set(0.5)
        metal = shader.CreateInput("metallic", Sdf.ValueTypeNames.Float)
        metal.Set(0.0)
        diffuse = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)

        reader = UsdShade.Shader.Define(self.stage, path + "/stReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

        tex = UsdShade.Shader.Define(self.stage, path + "/tex")
        tex.CreateIdAttr("UsdUVTexture")
        file_in = tex.CreateInput("file", Sdf.ValueTypeNames.Asset)
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result")
        tex_out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        diffuse.ConnectToSource(tex_out)

        mat.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface")
        return {"material": mat, "shader": shader, "file": file_in,
                "diffuse": diffuse, "rough": rough, "metal": metal,
                "tex_out": tex_out}

    def _make_camera(self):
        UsdGeom, Gf = self._UsdGeom, self._Gf
        k = self.k
        cam = UsdGeom.Camera.Define(self.stage, "/World/cam")
        # fx = focalLength * W / horizontalAperture. Pick aperture = 24 and solve,
        # which also fixes cx=cy=W/2,H/2 (USD cameras are centred by construction).
        # fx = f * W / ap_h  and  fy = f * H / ap_v. Fix ap_h = 24 and solve both:
        #   f    = fx * ap_h / W
        #   ap_v = f * H / fy = ap_h * (fx/fy) * (H/W)
        ap_h = 24.0
        f = float(k["fx"]) * ap_h / float(k["W"])
        ap_v = ap_h * (float(k["fx"]) / float(k["fy"])) * (float(k["H"]) / float(k["W"]))
        cam.CreateFocalLengthAttr(f)
        cam.CreateHorizontalApertureAttr(ap_h)
        cam.CreateVerticalApertureAttr(ap_v)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1e4))
        xf = UsdGeom.Xformable(cam)
        xf.ClearXformOpOrder()
        self._cam_op = xf.AddTransformOp()
        self.cam = cam

    def intrinsic_matrix(self):
        k = self.k
        return np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]], float)

    # ---- render product ----------------------------------------------------
    def _attach_render(self, want_masks):
        rep = self._rep
        self.rp = rep.create.render_product("/World/cam", (self.k["W"], self.k["H"]))
        self.ann_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.ann_rgb.attach(self.rp)
        self.ann_seg = None
        if want_masks:
            try:
                self.ann_seg = rep.AnnotatorRegistry.get_annotator("instance_segmentation")
                self.ann_seg.attach(self.rp)
            except Exception as e:
                print(f"[isaac] instance_segmentation unavailable ({e}); "
                      f"continuing without masks", flush=True)

    # ---- per-scene randomization ------------------------------------------
    def reset(self, rng):
        """Rebuild the object set, ground texture and lighting. Returns wP."""
        Gf, UsdGeom = self._Gf, self._UsdGeom
        stage = self.stage

        # Objects: remove then re-add. Prim removal under a stable parent avoids
        # the stage churn (and leak) that per-scene stage rebuilds cause.
        for prim in list(stage.GetPrimAtPath("/World/objects").GetChildren()):
            stage.RemovePrim(prim.GetPath())

        n_obj = int(rng.randint(1, 7))                 # paper: 1~6 objects
        centres, extents = [], []
        for i in range(n_obj):
            path = f"/World/objects/obj_{i}"
            xf = UsdGeom.Xform.Define(stage, path)
            model = self.models[rng.randint(len(self.models))]
            xf.GetPrim().GetReferences().AddReference(model)
            # GSO models_normalized are unit-normalized, so scale IS the object's
            # size in metres. Paper randomizes object size.
            s = float(rng.uniform(0.08, 0.30))
            pos = np.array([rng.uniform(-0.18, 0.18), rng.uniform(-0.18, 0.18),
                            s * 0.5])
            x = UsdGeom.Xformable(xf)
            x.ClearXformOpOrder()
            x.AddTranslateOp().Set(Gf.Vec3d(*pos.tolist()))
            x.AddRotateXYZOp().Set(Gf.Vec3f(*(rng.uniform(0, 360, 3).tolist())))
            x.AddScaleOp().Set(Gf.Vec3f(s, s, s))
            self._set_semantics(xf.GetPrim(), f"obj_{i}")
            centres.append(pos)
            extents.append(s)

        # Background texture + material randomization (paper Sec. H).
        if self.texes:
            t = self.texes[rng.randint(len(self.texes))]
            self._ground_mat["file"].Set(self._Sdf.AssetPath(t))
        self._ground_mat["rough"].Set(float(rng.uniform(0.15, 0.9)))
        self._ground_mat["metal"].Set(float(rng.uniform(0.0, 0.4)))
        tile = float(rng.uniform(1.0, 8.0))
        self._ground_st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(tile, 0),
                             Gf.Vec2f(tile, tile), Gf.Vec2f(0, tile)])

        # Ambient light: HDRI + random yaw + random intensity.
        if self.hdris:
            h = self.hdris[rng.randint(len(self.hdris))]
            self.dome.CreateTextureFileAttr().Set(self._Sdf.AssetPath(h))
        self.dome.GetIntensityAttr().Set(float(rng.uniform(400.0, 2000.0)))
        self._dome_rot.Set(float(rng.uniform(0.0, 360.0)))

        # Camera poses must be sampled about the SCENE centre, not the world
        # origin: objects scatter over +/-0.18 m, so centring on the origin would
        # frame the cluster off-centre and change d* relative to what bproc_gen
        # produced. bproc_gen takes the bounding-box centre of all objects
        # (bproc_gen.py:120-123); the GSO models are unit-normalized so an object
        # scaled by s has half-extent s/2, which reproduces that box here.
        c = np.asarray(centres, float)
        e = np.asarray(extents, float)[:, None] * 0.5
        lo, hi = (c - e).min(0), (c + e).max(0)
        self.scene_center = 0.5 * (lo + hi)
        self.scene_radius = float(0.5 * np.linalg.norm(hi - lo)) + 1e-3
        return c

    def _set_semantics(self, prim, label):
        """Instance masks need semantics. The API moved in Isaac 6.0, so try the
        current path first and degrade quietly -- masks are a Fig. 3 extra, not a
        requirement for the RGB+pose contract."""
        try:
            from pxr import UsdSemantics
            api = UsdSemantics.LabelsAPI.Apply(prim, "class")
            api.CreateLabelsAttr([label])
            return
        except Exception:
            pass
        try:
            from isaacsim.core.utils.semantics import add_labels
            add_labels(prim, labels=[label], instance_name="class")
        except Exception:
            pass

    # ---- rendering ---------------------------------------------------------
    def set_pose(self, c2w_cv):
        m = np.asarray(c2w_cv, float) @ CV2GL
        # Gf.Matrix4d is row-major and USD transforms are row-vector convention,
        # so the numpy column-vector matrix goes in transposed.
        self._cam_op.Set(self._Gf.Matrix4d(*m.T.flatten().tolist()))

    def step(self, subframes=None):
        self._rep.orchestrator.step(
            rt_subframes=self.rt_subframes if subframes is None else subframes,
            pause_timeline=False)

    def render(self, c2w_cv, want_mask=True):
        """Set the camera to an OpenCV c2w pose and return (rgb uint8 [H,W,3], mask)."""
        self.set_pose(c2w_cv)
        self.step()
        rgb = np.asarray(self.ann_rgb.get_data())
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]                       # annotator gives RGBA
        rgb = np.ascontiguousarray(rgb.astype(np.uint8))
        mask = None
        if want_mask and self.ann_seg is not None:
            d = self.ann_seg.get_data()
            if isinstance(d, dict):
                # Keep the id->prim mapping. Without it the mask is ambiguous:
                # the GROUND PLANE gets a normal instance id (id 1, and there is
                # no id 0 at all), so "ids != 0" counts the floor as an object.
                # That produced a frame-fill measurement of 0.99 when the true
                # value is 0.32 -- believable enough to have been acted on.
                self.last_mask_info = d.get("info")
                d = d.get("data")
            mask = np.ascontiguousarray(np.asarray(d).astype(np.uint32))
        return rgb, mask

    def warmup(self, n=8):
        """The RTX pipeline needs a few frames to settle after a scene change;
        without this the first render of a scene can come back black or stale."""
        for _ in range(n):
            self.step()

    def auto_expose(self, c2w_cv, rng, target_lo=25.0, target_hi=215.0, tries=6):
        """Nudge dome intensity until a reference view is neither black nor blown.

        Randomizing intensity over a wide range (needed for illumination
        robustness -- the paper's real-world Table II has an explicit
        inconsistent-illumination column) occasionally lands on a scene so dark
        the image is unusable: measured 7 near-black frames in 2800. Those pairs
        carry a perfectly good pose label attached to an image with no signal, so
        they are pure label noise. Rescale rather than resample the HDRI, which
        keeps the chosen environment map.
        """
        for _ in range(tries):
            self.set_pose(c2w_cv)
            self.step()
            m = float(np.asarray(self.ann_rgb.get_data())[..., :3].mean())
            if target_lo <= m <= target_hi:
                return m
            cur = float(self.dome.GetIntensityAttr().Get() or 1000.0)
            # Aim at the middle of the acceptable band, clamped so one bad frame
            # cannot drive the intensity to an absurd value.
            want = 0.5 * (target_lo + target_hi)
            scale = float(np.clip(want / max(m, 1.0), 0.25, 4.0))
            self.dome.GetIntensityAttr().Set(float(np.clip(cur * scale, 50.0, 20000.0)))
        return m


# --------------------------------------------------------------------------
# Offline scene generation (paper Sec. H, "Simulation Process #1": uniform)
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", type=int, default=800)
    ap.add_argument("--currents", type=int, default=4)
    ap.add_argument("--usd", default="data/gso_usd")
    ap.add_argument("--hdri", default="data/hdri")
    ap.add_argument("--tex", default="data/cc_textures")
    ap.add_argument("--seed", type=int, default=0)
    # Both defaults come from scripts/bench_isaac.py rather than guesswork.
    # rt_subframes: MAE against a 64-subframe reference was 0.32-0.69/255 (PSNR
    # 47-52 dB) at EVERY setting from 1 to 32, i.e. quality is already denoiser-
    # limited and accumulation buys nothing here -- but 32 costs 126 ms/frame
    # against 18 ms at 1-2. warmup: a fresh scene rendered with warmup=0 already
    # differed from the PREVIOUS scene by MAE 90.6 and from a fully settled render
    # of itself by only 2.36, so there is no scene leakage; 2 frames puts it at
    # 0.99, indistinguishable from settled.
    ap.add_argument("--rt-subframes", type=int, default=2,
                    help="RTX accumulation per frame; higher = cleaner, slower")
    ap.add_argument("--warmup", type=int, default=2,
                    help="throwaway frames after each scene change")
    ap.add_argument("--near-frac", type=float, default=0.0,
                    help="0..1 fraction of currents perturbed from the desired pose")
    ap.add_argument("--no-masks", action="store_true")
    ap.add_argument("--no-auto-expose", action="store_true",
                    help="disable the brightness guard (see IsaacSceneGen.auto_expose)")
    ap.add_argument("--headless", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0,
                    help="which GPU renders; the other stays free for training")
    args = ap.parse_args()

    app = launch(headless=bool(args.headless), gpu=args.gpu)

    # Imported AFTER SimulationApp exists, and imported rather than reimplemented
    # -- see the module docstring on bproc_eval_server's duplicated sampler.
    from cns.sim.pose_sampling import sample_desired, sample_initial
    from cns.sim.pose_perturb import perturb_pose

    gen = IsaacSceneGen(args.usd, args.hdri, args.tex,
                        rt_subframes=args.rt_subframes,
                        want_masks=not args.no_masks)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    K = gen.intrinsic_matrix()

    # Staleness guard: Replicator can hand back the PREVIOUS frame if the render
    # has not caught up, which would silently pair images with the wrong poses --
    # exactly the class of bug that cost this project a whole DAgger campaign. So
    # prove up front that moving the camera changes the pixels.
    gen.reset(rng)
    gen.warmup(12)
    a, _ = gen.render(sample_desired(rng, gen.scene_center), want_mask=False)
    b, _ = gen.render(sample_initial(rng, gen.scene_center), want_mask=False)
    diff = float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
    print(f"[isaac] staleness check: mean |I_a - I_b| = {diff:.2f}", flush=True)
    if diff < 1.0:
        raise SystemExit(
            "[isaac] ABORT: two different camera poses produced near-identical "
            "images. The annotator is returning stale frames -- raise --warmup / "
            "--rt-subframes before generating a dataset.")
    if float(a.std()) < 1.0:
        raise SystemExit(f"[isaac] ABORT: rendered image is flat (std {a.std():.2f}); "
                         f"lighting or materials are not being applied.")

    t0 = time.time()
    n_frames = 0
    for s in range(args.scenes):
        wP = gen.reset(rng)
        gen.warmup(args.warmup)

        # view 0 = desired; views 1..M = currents. Sampling is CNS-v1-faithful:
        # desired phi 70-90 deg / roll +/-15, initial phi 30-90 / roll +/-60.
        desired = sample_desired(rng, gen.scene_center)
        # Fix the exposure against the DESIRED view before rendering anything, so
        # every view of the scene shares one lighting level.
        if not args.no_auto_expose:
            gen.auto_expose(desired, rng)
        poses = [desired]
        for _ in range(args.currents):
            if rng.uniform() < args.near_frac:
                wcT, _te, _re = perturb_pose(desired, rng)
            else:
                wcT = sample_initial(rng, gen.scene_center)
            poses.append(wcT)

        imgs, masks = [], []
        for p in poses:
            rgb, mask = gen.render(p, want_mask=not args.no_masks)
            imgs.append(rgb)
            if mask is not None:
                masks.append(mask)
            n_frames += 1

        payload = dict(images=np.stack(imgs).astype(np.uint8),
                       poses=np.asarray(poses, float),
                       wP=wP, K=K)
        if masks:
            payload["masks"] = np.stack(masks).astype(np.uint32)
            if gen.last_mask_info is not None:
                payload["mask_info"] = np.array(repr(gen.last_mask_info))
        np.savez_compressed(os.path.join(args.out, f"scene_{s:04d}.npz"), **payload)

        if (s + 1) % 25 == 0:
            el = time.time() - t0
            fps = n_frames / el
            try:
                rss = int(open(f"/proc/{os.getpid()}/status").read()
                          .split("VmRSS:")[1].split()[0]) / 1e6
            except Exception:
                rss = float("nan")
            print(f"[isaac] scene {s+1}/{args.scenes}  {fps:.1f} frame/s "
                  f"({1000/fps:.0f} ms/frame)  rss {rss:.1f}GB  "
                  f"eta {(args.scenes-s-1)*(el/(s+1))/60:.1f}min", flush=True)

    el = time.time() - t0
    print(f"[isaac] DONE {args.scenes} scenes / {n_frames} frames in {el/60:.1f}min "
          f"= {1000*el/n_frames:.0f} ms/frame -> {args.out}", flush=True)
    print("ISAAC_GEN_DONE", flush=True)
    app.close()


if __name__ == "__main__":
    main()
