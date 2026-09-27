"""生成校准图 + dataset.txt。

校准只跑几次 forward，5~16 张就够；多了只是慢。

★工作目录约定：本脚本的所有输入输出都相对 **CWD**，不是脚本所在目录。
  workout 里以 `-w /workspace/models/<name>` 调用，因此：
      <cwd>/calib/*.png     校准图
      <cwd>/_raw/*.png      DIV2K 原图缓存
      <cwd>/dataset.txt     一行一个 ./calib/<name>
  dataset.txt 里的路径是 './calib/...' 形式，消费方必须同样以模型目录为 CWD。
  （旧版本用 Path(__file__).parent 锚定，挪目录后就会指错 —— 别改回去。）

usage:
    python3 prepare_calib.py --n 8 --h 360 --w 640 --source div2k [--div2k-start 801]
    python3 prepare_calib.py --n 8 --h 360 --w 640 --source random
    python3 prepare_calib.py --n 8 --h 360 --w 640 --source dir:myimages
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

DIV2K_BASE = "https://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_valid_HR/"
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def fetch_div2k(raw_dir, n, start):
    raw_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i in range(start, start + n):
        url = "%s%04d.png" % (DIV2K_BASE, i)
        out = raw_dir / ("%04d.png" % i)
        if not out.exists():
            print("  fetch %s" % url)
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            Image.open(io.BytesIO(data)).convert("RGB").save(out)
        saved.append(out)
    return saved


def make_random(out_dir, n, h, w, seed):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        p = out_dir / ("random_%03d.png" % i)
        Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


def crop(images, out_dir, h, w):
    """center-crop 到 h×w，结果写进 out_dir（文件名沿用原名）。

    刻意放在 fetch 的 try 之外：图下载残缺导致解码失败时，必须硬失败，
    不能被"降级成随机噪声"的分支吞掉 —— 那会把一个显式错误换成一个更坏的产物。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for src in images:
        im = Image.open(src).convert("RGB")     # 解码失败在这里抛，不捕获
        cw, ch = im.size                        # PIL: (W, H)
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
    ap.add_argument("--out", default="calib",
                    help="校准图输出子目录，相对 CWD。必须是相对路径。")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--h", type=int, default=360)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--source", default="div2k",
                    help="div2k / random / dir:<相对路径>")
    ap.add_argument("--div2k-start", type=int, default=801)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fallback-random", action="store_true",
                    help="校准源拉不到时用随机噪声占位。"
                         "★产出的 NBG 质量更差且从产物上看不出来 —— "
                         "只应由 model.json 的 calib.allow_fallback 显式开启。")
    args = ap.parse_args()

    if os.path.isabs(args.out) or ".." in Path(args.out).parts:
        # dataset.txt 写的是 './<out>/<name>'，绝对路径会拼出 './abs/path' 这种坏路径
        sys.exit("--out 必须是相对路径且不含 '..'，实际 %r" % args.out)

    cwd = Path.cwd()
    out = cwd / args.out
    fallback_used = False

    if args.source == "random":
        paths = make_random(out, args.n, args.h, args.w, args.seed)

    elif args.source.startswith("dir:"):
        src = cwd / args.source[4:]
        if not src.is_dir():
            sys.exit("--source dir: 指向的目录不存在：%s" % src)
        imgs = sorted(p for p in src.iterdir()
                      if p.is_file() and p.suffix.lower() in IMG_EXT)
        if len(imgs) < args.n:
            sys.exit("--source dir:%s 里只有 %d 张图，需要 %d 张"
                     % (src, len(imgs), args.n))
        paths = crop(imgs[:args.n], out, args.h, args.w)

    elif args.source == "div2k":
        # ★只捕获 fetch 段。`except OSError` 覆盖 urllib.error.URLError（其子类）、
        #   ConnectionError，以及 **Python 3.8 的 socket.timeout** —— 3.8 里
        #   socket.timeout 还不是 TimeoutError 的别名（3.10 才合并），
        #   旧代码写 `except (URLError, TimeoutError)` 导致读超时直接逃逸，
        #   --fallback-random 这条分支永远走不到。别把这里"简化"回那个写法。
        try:
            raws = fetch_div2k(cwd / "_raw", args.n, args.div2k_start)
        except OSError as exc:
            if not args.fallback_random:
                sys.exit("DIV2K 不可达（%s）。\n"
                         "  校准图必须是真实图像 —— 随机噪声标定出来的量化参数是垃圾。\n"
                         "  要强行用噪声跑通流程，在 model.json 里设 calib.allow_fallback=true。"
                         % exc)
            print("WARNING: DIV2K 不可达（%s）—— 按要求降级为随机噪声。" % exc)
            print("WARNING: 本次产出的 NBG 量化质量不可信，仅可用于流程验证。")
            fallback_used = True
            paths = make_random(out, args.n, args.h, args.w, args.seed)
        else:
            paths = crop(raws, out, args.h, args.w)   # ← 在 try 之外
    else:
        sys.exit("--source 必须是 div2k / random / dir:<相对路径>，实际 %r" % args.source)

    dataset_txt = cwd / "dataset.txt"
    dataset_txt.write_text("\n".join("./%s/%s" % (args.out, p.name) for p in paths) + "\n")
    print("wrote %s (%d entries, source=%s%s)"
          % (dataset_txt, len(paths), args.source,
             ", FALLBACK=RANDOM" if fallback_used else ""))


if __name__ == "__main__":
    main()
