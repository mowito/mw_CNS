"""Fetch CC0 PBR background/ground textures from ambientCG.

Paper Table I randomizes the background over **32k texture images** and materials;
this repo had ZERO -- `bproc_gen.rand_material()` only jitters Principled BSDF
base colour/roughness/metallic, which is procedural and image-free. ambientCG is
the canonical CC0 source (it is what BlenderProc's own `cc_textures` downloader
pulls) and gives real PBR map sets, usable both as ground-plane materials and as
plain background images.

    python3 scripts/fetch_cc_textures.py --out data/cc_textures --res 1K-JPG

Each asset arrives as a zip of {Color, NormalGL, Roughness, Displacement, AO}
maps and is extracted to `<out>/<AssetId>/`. Existing assets are skipped, so this
is re-runnable and resumable. ~1.6k assets at 1K-JPG is roughly 9 GB.
"""
import argparse, json, os, sys, time, urllib.request, urllib.error, zipfile, io

API = "https://ambientcg.com/api/v2/full_json"
UA = {"User-Agent": "cnsv2-train/1.0"}


def get(url, timeout=120):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_assets():
    """Walk the paginated asset list (100 per page)."""
    out, offset = [], 0
    while True:
        url = (f"{API}?type=Material&sort=Alphabet&include=downloadData"
               f"&limit=100&offset={offset}")
        page = json.loads(get(url))
        found = page.get("foundAssets", [])
        if not found:
            break
        out += found
        offset += len(found)
        print(f"[tex] listed {len(out)} assets...", flush=True)
        if len(found) < 100:
            break
    return out


def pick_link(asset, res):
    """Find the download link for the requested resolution/format tag.

    Walks the structure rather than assuming its shape: ambientCG returns
    `downloadFolders` as a dict for most assets but as a LIST for some, and the
    same is true of `downloadFiletypeCategories`. Indexing by .values()
    unconditionally crashed the run at asset 239/2005 with
    "AttributeError: 'list' object has no attribute 'values'".
    """
    def walk(node):
        if isinstance(node, dict):
            if node.get("attribute") == res and node.get("downloadLink"):
                yield node
                return
            for v in node.values():
                yield from walk(v)
        elif isinstance(node, list):
            for v in node:
                yield from walk(v)

    for dl in walk(asset.get("downloadFolders", {})):
        return dl.get("downloadLink"), dl.get("size", 0)
    return None, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cc_textures")
    ap.add_argument("--res", default="1K-JPG",
                    help="ambientCG variant tag, e.g. 1K-JPG / 2K-JPG")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    assets = list_assets()
    if args.limit:
        assets = assets[:args.limit]
    print(f"[tex] {len(assets)} assets available on ambientCG", flush=True)

    ok = skip = fail = 0
    for i, a in enumerate(assets, 1):
        aid = a.get("assetId")
        if not aid:
            continue
        dest = os.path.join(args.out, aid)
        # A non-empty dir means a previous run already extracted this asset.
        if os.path.isdir(dest) and os.listdir(dest):
            skip += 1
            continue
        link, size = pick_link(a, args.res)
        if not link:
            print(f"[tex] {i}/{len(assets)} {aid}: no {args.res} variant", flush=True)
            fail += 1
            continue
        try:
            blob = get(link)
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                os.makedirs(dest, exist_ok=True)
                z.extractall(dest)
            ok += 1
            if ok % 25 == 0 or i < 5:
                print(f"[tex] {i}/{len(assets)} ok={ok} skip={skip} fail={fail} "
                      f"({aid}, {len(blob)/1e6:.1f}MB)", flush=True)
        except (urllib.error.URLError, zipfile.BadZipFile, OSError, TimeoutError) as e:
            fail += 1
            print(f"[tex] {i}/{len(assets)} {aid}: {type(e).__name__} {e}", flush=True)
            time.sleep(2)   # ambientCG throttles bursts; back off rather than hammer

    print(f"[tex] DONE ok={ok} skipped={skip} failed={fail} -> {args.out}", flush=True)
    if fail:
        print("[tex] re-run to retry the failures (existing assets are skipped)")


if __name__ == "__main__":
    sys.exit(main())
