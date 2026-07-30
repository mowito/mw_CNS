"""Browse the IsaacSim-generated dataset as image contact sheets.

Statistics do not catch a bad renderer. The shared-texture bug (Sec. 6.11) passed
every pose/label/histogram check and was visible only by looking at the pictures,
so this exists to make looking easy.

    # 6 random scenes from the training set
    python scripts/view_scenes.py --data data/isaac_train --n 6 --out /tmp/sheet.png

    # specific scenes, with instance masks beside each view
    python scripts/view_scenes.py --scenes 0,1,2 --masks --out /tmp/sheet.png

    # DAgger rollouts: consecutive states along one trajectory
    python scripts/view_scenes.py --data data/dagger_live --n 4 --out /tmp/dagger.png

Column 0 is the DESIRED view (green bar); the rest are current views, annotated
with the pose error the PBVS supervisor labels them with -- so you can see
whether the picture and the label agree.
"""
import argparse, glob, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def pose_error(cur, tar):
    from scipy.spatial.transform import Rotation as R
    d = np.linalg.inv(tar) @ cur
    te = float(np.linalg.norm(d[:3, 3]))
    re = float(np.degrees(np.linalg.norm(R.from_matrix(d[:3, :3]).as_rotvec())))
    return te, re


def label(img, text, bar=None):
    """Burn a caption strip under an image; optional coloured top bar."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    W, H = im.size
    out = Image.new("RGB", (W, H + 22), (16, 16, 16))
    out.paste(im, (0, 0))
    d = ImageDraw.Draw(out)
    d.text((4, H + 5), text, fill=(235, 235, 235))
    if bar:
        d.rectangle([0, 0, W - 1, 4], fill=bar)
    return np.asarray(out)


def mask_rgb(m):
    """Colourize a uint32 instance mask deterministically."""
    ids = np.unique(m)
    out = np.zeros(m.shape + (3,), np.uint8)
    for i, v in enumerate(ids):
        if v == 0:
            continue
        rng = np.random.RandomState(int(v) * 9973 + 17)
        out[m == v] = rng.randint(60, 255, 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/isaac_train")
    ap.add_argument("--n", type=int, default=6, help="random scenes to show")
    ap.add_argument("--scenes", default="", help="explicit indices, e.g. 0,5,12")
    ap.add_argument("--views", type=int, default=5, help="views per scene (incl. desired)")
    ap.add_argument("--masks", action="store_true", help="show instance mask beside each view")
    ap.add_argument("--scale", type=int, default=256, help="thumbnail size in px")
    ap.add_argument("--out", default="/tmp/isaac_scenes.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from PIL import Image

    files = sorted(glob.glob(os.path.join(args.data, "scene_*.npz")))
    if not files:
        raise SystemExit(f"no scene_*.npz under {args.data}")
    if args.scenes:
        idx = [int(i) for i in args.scenes.split(",")]
        files = [files[i] for i in idx if i < len(files)]
    else:
        rng = np.random.RandomState(args.seed)
        files = [files[i] for i in rng.choice(len(files), min(args.n, len(files)), replace=False)]

    rows = []
    for f in files:
        z = np.load(f)
        imgs, poses = z["images"], z["poses"]
        has_mask = args.masks and "masks" in z.files
        n = min(args.views, imgs.shape[0])
        cells = []
        for i in range(n):
            if i == 0:
                cap, bar = f"DESIRED  {os.path.basename(f)[:-4]}", (40, 200, 90)
            else:
                te, re = pose_error(poses[i], poses[0])
                cap, bar = f"cur {i}   TE {te*1000:.0f}mm  RE {re:.0f}deg", (70, 130, 220)
            cells.append(label(imgs[i], cap, bar))
            if has_mask:
                cells.append(label(mask_rgb(z["masks"][i]), f"  mask {i}", (150, 150, 150)))
        rows.append(np.concatenate(cells, axis=1))

    w = min(r.shape[1] for r in rows)
    sheet = np.concatenate([r[:, :w] for r in rows], axis=0)
    im = Image.fromarray(sheet)
    # scale so each cell lands near --scale px wide
    per_cell = w / (args.views * (2 if args.masks else 1))
    k = args.scale / per_cell
    if k < 1.0:
        im = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
    im.save(args.out)
    print(f"{len(files)} scenes -> {args.out}  ({im.width}x{im.height})")
    print(f"open with:  xdg-open {args.out}")


if __name__ == "__main__":
    main()
