"""One-time batch conversion of GSO .obj meshes to USD for IsaacSim.

USD cannot reference .obj directly, and running omni.kit.asset_converter per
scene during data generation would dominate render time. So convert all 944 GSO
models once, then `isaac_scene.py` just references the resulting .usd files.

Run with IsaacSim's own interpreter:

    ~/isaacsim/python.sh scripts/convert_gso_to_usd.py \
        --src data/gso/models/google_scanned_objects/models_normalized \
        --out data/gso_usd

Each GSO model dir is `<name>/meshes/{model.obj,model.mtl,texture.png}`; output is
`<out>/<name>.usd`. Existing outputs are skipped, so this is resumable.
"""
import argparse, asyncio, os, sys, time, glob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="GSO models_normalized dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})

    import omni.kit.asset_converter as ac
    import carb

    os.makedirs(args.out, exist_ok=True)
    objs = sorted(glob.glob(os.path.join(args.src, "*", "meshes", "model.obj")))
    if args.limit:
        objs = objs[:args.limit]
    print(f"[usd] {len(objs)} GSO .obj found under {args.src}", flush=True)

    def task_options():
        o = ac.AssetConverterContext()
        # GSO meshes are already unit-normalized and single-material; skip the
        # expensive extras and keep the diffuse texture.
        o.ignore_animations = True
        o.ignore_camera = True
        o.ignore_light = True
        o.single_mesh = True
        o.smooth_normals = True
        o.embed_textures = True
        return o

    async def convert(src, dst):
        inst = ac.get_instance()
        task = inst.create_converter_task(src, dst, None, task_options())
        ok = await task.wait_until_finished()
        if not ok:
            return False, task.get_status(), task.get_error_message()
        return True, None, None

    ok = skip = fail = 0
    t0 = time.time()
    loop = asyncio.get_event_loop()
    for i, src in enumerate(objs, 1):
        # .../<name>/meshes/model.obj -> <name>
        name = os.path.basename(os.path.dirname(os.path.dirname(src)))
        # EACH MODEL GETS ITS OWN DIRECTORY. The converter writes extracted
        # textures to `<dst_dir>/materials/textures/<basename>`, and EVERY GSO
        # model's texture is named `texture.png` -- so a flat output dir made all
        # 944 USDs reference one shared materials/textures/texture.png (whichever
        # model converted last). Renders came back with every object wearing the
        # same skin, which no shape or pose check would ever flag.
        dst = os.path.join(args.out, name, f"{name}.usd")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            skip += 1
            continue
        try:
            good, status, err = loop.run_until_complete(convert(src, dst))
        except Exception as e:                      # converter can throw on bad meshes
            good, status, err = False, type(e).__name__, str(e)
        if good:
            ok += 1
        else:
            fail += 1
            print(f"[usd] FAIL {name}: {status} {err}", flush=True)
        if i % 25 == 0:
            rate = i / (time.time() - t0)
            eta = (len(objs) - i) / max(rate, 1e-6) / 60
            print(f"[usd] {i}/{len(objs)} ok={ok} skip={skip} fail={fail} "
                  f"{rate:.1f}/s eta {eta:.1f}min", flush=True)

    print(f"[usd] DONE ok={ok} skipped={skip} failed={fail} -> {args.out} "
          f"in {(time.time()-t0)/60:.1f}min", flush=True)
    app.close()


if __name__ == "__main__":
    sys.exit(main())
