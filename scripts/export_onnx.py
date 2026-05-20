"""EDSR x2 (super-image, eugenesiow/edsr-base) → ONNX opset 11.

VeriSilicon pegasus (acuity 6.30.x) 接受 opset 11；DepthToSpace 在更高 opset
有 mode 属性差异，固定 11 最稳。

usage:
    python export_onnx.py --resolution 3,360,640 --output edsr_x2.onnx
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
    ap.add_argument("--resolution", default="3,360,640",
                    help="CHW, comma-separated. Default 3,360,640 (720x1280 after x2).")
    ap.add_argument("--output", default="edsr_x2.onnx")
    ap.add_argument("--scale", type=int, default=2)
    args = ap.parse_args()

    c, h, w = (int(x) for x in args.resolution.split(","))

    model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=args.scale)
    # super-image wraps in DataParallel
    model = model.module if hasattr(model, "module") else model
    model.eval()

    dummy = torch.randn(1, c, h, w)
    torch.onnx.export(
        model, dummy, args.output,
        opset_version=11,
        input_names=["input"], output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch", 2: "h", 3: "w"},
            "output": {0: "batch", 2: "h", 3: "w"},
        },
    )

    m = onnx.load(args.output)
    print(f"opset: {m.opset_import[0].version}")
    for n in m.graph.node:
        if n.op_type == "DepthToSpace":
            for a in n.attribute:
                v = a.i if a.type == onnx.AttributeProto.INT else a.s
                print(f"  DepthToSpace.{a.name}: {v}")

    sess = ort.InferenceSession(args.output)
    x = np.random.randn(1, c, h, w).astype(np.float32)
    y = sess.run(None, {"input": x})
    print(f"OK: {x.shape} -> {y[0].shape}  ({Path(args.output).stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
