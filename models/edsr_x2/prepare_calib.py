"""生成校准图：从 DIV2K 验证集下载 N 张，center-crop 到 360x640。

校准只跑几次 forward，5~16 张就够了；多了也只是慢。
若网络不可达，加 --fallback-random 用噪声图占位（量化质量会差，仅用于流程验证）。
"""
import argparse
import io
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


def fetch_div2k(out_dir: Path, n: int = 8):
    out_dir.mkdir(parents=True, exist_ok=True)
    base = "https://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_valid_HR/"
    saved = []
    for i in range(801, 801 + n):
        url = f"{base}{i:04d}.png"
        out = out_dir / f"{i:04d}.png"
        if not out.exists():
            print(f"  fetch {url}")
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            Image.open(io.BytesIO(data)).convert("RGB").save(out)
        saved.append(out)
    return saved


def make_random(out_dir: Path, n: int, h: int, w: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    rng = np.random.default_rng(0)
    for i in range(n):
        arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        p = out_dir / f"random_{i:03d}.png"
        Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


def crop(images, out_dir: Path, h: int, w: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for src in images:
        im = Image.open(src).convert("RGB")
        cw, ch = im.size  # PIL: (W, H)
        if cw < w or ch < h:
            im = im.resize((max(cw, w), max(ch, h)), Image.BICUBIC)
            cw, ch = im.size
        left = (cw - w) // 2
        top = (ch - h) // 2
        im.crop((left, top, left + w, top + h)).save(out_dir / src.name)
        paths.append(out_dir / src.name)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="calib")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--h", type=int, default=360)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--fallback-random", action="store_true",
                    help="DIV2K 不可达时用随机噪声（仅供流程验证，PSNR 会很差）")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    out = here / args.out

    try:
        raws = fetch_div2k(here / "_raw", args.n)
        paths = crop(raws, out, args.h, args.w)
    except (urllib.error.URLError, TimeoutError) as e:
        if not args.fallback_random:
            raise
        print(f"DIV2K unreachable ({e}); falling back to random noise")
        paths = make_random(out, args.n, args.h, args.w)

    dataset_txt = here / "dataset.txt"
    dataset_txt.write_text("\n".join(f"./{args.out}/{p.name}" for p in paths) + "\n")
    print(f"wrote {dataset_txt} ({len(paths)} entries)")


if __name__ == "__main__":
    main()
