"""Fetch additional CC0 HDRI environment maps from Poly Haven.

The paper's Table I randomizes ambient lighting over ~733 HDRI maps; this repo
shipped with 5, which is thin both for sim generalization and (more importantly)
for transferring to a real RealSense under unseen lighting.

    python3 scripts/fetch_hdris.py --out data/hdri --count 150 --res 1k

Poly Haven assets are CC0. Existing files are skipped, so this is re-runnable.
"""
import argparse, json, os, sys, urllib.request

API = "https://api.polyhaven.com/assets?t=hdris"
FILE = "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/{res}/{name}_{res}.hdr"


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "cnsv2-train/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/hdri")
    ap.add_argument("--count", type=int, default=150)
    ap.add_argument("--res", default="1k", choices=["1k", "2k", "4k"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    names = sorted(json.loads(get(API)).keys())
    print(f"[hdri] {len(names)} available on Poly Haven", flush=True)
    have = {os.path.splitext(f)[0] for f in os.listdir(args.out) if f.endswith(".hdr")}
    print(f"[hdri] {len(have)} already present", flush=True)

    # Even stride over the sorted list -> a spread of lighting conditions rather
    # than 150 alphabetically-adjacent (often near-duplicate) maps.
    todo = [n for n in names if n not in have and f"{n}_{args.res}" not in have]
    step = max(1, len(todo) // max(args.count, 1))
    todo = todo[::step][:args.count]

    ok = fail = 0
    for i, n in enumerate(todo):
        dest = os.path.join(args.out, f"{n}_{args.res}.hdr")
        if os.path.exists(dest):
            continue
        try:
            data = get(FILE.format(name=n, res=args.res))
            with open(dest, "wb") as f:
                f.write(data)
            ok += 1
            if ok % 10 == 0:
                print(f"[hdri] {ok}/{len(todo)} ok ({n}, {len(data)/1e6:.1f}MB)", flush=True)
        except Exception as e:
            fail += 1
            print(f"[hdri] FAILED {n}: {type(e).__name__} {e}", flush=True)
            if os.path.exists(dest):
                os.remove(dest)          # never leave a truncated .hdr behind
    total = len([f for f in os.listdir(args.out) if f.endswith(".hdr")])
    print(f"HDRI_DONE downloaded={ok} failed={fail} total_now={total}", flush=True)


if __name__ == "__main__":
    main()
