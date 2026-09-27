"""EDSR x2 (super-image, eugenesiow/edsr-base) → ONNX opset 11。

VeriSilicon pegasus (acuity 6.30.x) 接受 opset 11；DepthToSpace 在更高 opset
有 mode 属性差异，固定 11 最稳。

默认导出**静态形状**（batch=1, C, H, W）。NBG 本来就是定形状的，
dynamic_axes 只会制造"导出形状 ≠ 编译形状"这类错位 —— 老版本的 workflow
就因为这个把 resolution 输入变成了死参数。

usage:
    python3 export.py --shape 3,360,640 --output edsr_x2.onnx
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from super_image import EdsrModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True,
                    help="CHW, 逗号分隔，例如 3,360,640（x2 后是 720x1280）")
    ap.add_argument("--output", required=True,
                    help="输出 ONNX 路径。故意不给默认值：写 CWD 的默认值正是"
                         "让通用驱动产生'编了个别的文件'这类事故的根源。")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--dynamic", action="store_true",
                    help="导出动态 H/W（默认关。NBG 定形状，除非有特别理由否则别开）")
    args = ap.parse_args()

    c, h, w = (int(x) for x in args.shape.split(","))

    model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=args.scale)
    # super-image wraps in DataParallel
    model = model.module if hasattr(model, "module") else model
    model.eval()

    dummy = torch.randn(1, c, h, w)
    kwargs = {}
    if args.dynamic:
        kwargs["dynamic_axes"] = {
            "input":  {0: "batch", 2: "h", 3: "w"},
            "output": {0: "batch", 2: "h", 3: "w"},
        }

    torch.onnx.export(
        model, dummy, args.output,
        opset_version=11,
        input_names=["input"], output_names=["output"],
        **kwargs,
    )

    m = onnx.load(args.output)
    print("opset: %d" % m.opset_import[0].version)
    for n in m.graph.node:
        if n.op_type == "DepthToSpace":
            for a in n.attribute:
                v = a.i if a.type == onnx.AttributeProto.INT else a.s
                print("  DepthToSpace.%s: %s" % (a.name, v))
            # PyTorch pixel_shuffle ≡ DepthToSpace(mode="CRD")，而 ONNX 默认是 "DCR"。
            # 忽略 mode 的后端会产出形状正确、通道错乱的图（网格伪影，不报错）。
            modes = [a.s for a in n.attribute if a.name == "mode"]
            if modes and modes[0] != b"CRD":
                raise SystemExit(
                    "DepthToSpace.mode=%r，期望 b'CRD'。"
                    "模式不对会静默产出通道错乱的结果。" % modes[0])

    sess = ort.InferenceSession(args.output)
    x = np.random.randn(1, c, h, w).astype(np.float32)
    y = sess.run(None, {"input": x})
    size = Path(args.output).stat().st_size
    print("OK: %s -> %s  (%.1f KiB)" % (x.shape, y[0].shape, size / 1024))


if __name__ == "__main__":
    main()
