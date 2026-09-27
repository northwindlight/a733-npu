"""产出 build_info.json —— NBG 旁边那份机器可读的编译记录。

为什么需要它：README 里用散文记着"输出 scale=2.646, zero_point=103"，
而板端 C 代码要靠这两个数做 dequant。散文是会被抄错的，也会和产物脱节。
把"这次到底编了什么"落成 JSON，板端就有一个确定的取值处。

usage:
    python3 make_build_info.py --model edsr_x2 --config models/edsr_x2/model.json \
        --npu v3 --onnx models/edsr_x2/edsr_x2.onnx \
        --nbg models/edsr_x2/wksp/.../network_binary.nb \
        --qtype uint8 --vsconfig VIP9000NANODI_PLUS_PID0X1000003B \
        --out models/edsr_x2/build_info.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--npu", required=True)
    ap.add_argument("--vsconfig", required=True,
                    help="VSIMULATOR_CONFIG 的字符串（--optimize 收的是字符串不是数字）")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--nbg", required=True)
    ap.add_argument("--qtype", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--output-scale", type=float, default=None)
    ap.add_argument("--output-zero-point", type=int, default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())

    # 校准来源与是否降级：dataset.txt 里 random_* 前缀就是降级的痕迹
    calib_dir = cfg_path.parent
    entries = []
    ds = calib_dir / "dataset.txt"
    if ds.exists():
        entries = [ln.strip() for ln in ds.read_text().splitlines() if ln.strip()]
    fallback = any("random_" in e for e in entries)

    info = {
        "_comment": "由 scripts/make_build_info.py 生成。板端做 dequant 需要的 "
                    "output_quant 在这里取，别再从 README 里抄。",
        "model": args.model,
        "schema": cfg.get("schema"),
        "license": cfg.get("license"),
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(),
        "npu": {
            "target": args.npu,
            "optimize_string": args.vsconfig,
        },
        "io": {
            "inputs": [{"name": i["name"], "shape": i["shape"]} for i in cfg.get("inputs", [])],
            "outputs": list(cfg.get("outputs", [])),
        },
        "quant": {
            "qtype": args.qtype,
            "quantizer": cfg.get("quant", {}).get("quantizer"),
        },
        "calib": {
            "declared_source": cfg.get("calib", {}).get("source"),
            "n": len(entries),
            "fallback_to_random": fallback,
            "trustworthy": not fallback,
        },
        "artifacts": {
            "onnx": {"path": str(args.onnx), "sha256": sha256(args.onnx)},
            "nbg": {
                "path": str(args.nbg),
                "sha256": sha256(args.nbg),
                "size_bytes": Path(args.nbg).stat().st_size,
            },
        },
        "output_quant": {
            "_comment": "pegasus 的量化产物里怎么自动提取尚未验证，目前靠人工回填。"
                        "空值表示还没填 —— 板端遇到 null 应当拒绝，不要默认 1.0/0。",
            "scale": args.output_scale,
            "zero_point": args.output_zero_point,
        },
    }

    Path(args.out).write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    if fallback:
        sys.stderr.write(
            "WARNING: 本次校准降级成了随机噪声 —— 产出的 NBG 量化质量不可信。\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
