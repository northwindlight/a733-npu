"""Compare a board srpipe uint8 frame to the same RGB ONNX input.

srpipe must run exactly one tile with margin=0, no sharpening, GBR output.
Example: compare_nbg.py --onnx model.onnx --input tile.png --actual out.gbrp
Exit nonzero below --min-psnr. --scramble-reference intentionally simulates the
CRD/DCR PixelShuffle bug and must make the gate fail on a nontrivial image.
"""
import argparse
import json
import numpy as np
import onnxruntime as ort
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--actual", required=True)
    ap.add_argument("--min-psnr", type=float, default=30)
    ap.add_argument("--scramble-reference", action="store_true")
    args = ap.parse_args()
    src = np.asarray(Image.open(args.input).convert("RGB"))
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    expected_shape = session.get_inputs()[0].shape
    x = src.transpose(2, 0, 1)[None].astype(np.float32) / 255
    if list(x.shape) != expected_shape:
        raise ValueError("Input shape %s != ONNX %s" % (x.shape, expected_shape))
    ref = session.run(None, {session.get_inputs()[0].name: x})[0][0]
    c, h, w = ref.shape
    if args.scramble_reference:
        scale = w // src.shape[1]
        # CRD [(c,bh,bw),h,w] interpreted as DCR [(bh,bw,c),h,w].
        lh, lw = h // scale, w // scale
        packed = ref.reshape(c, lh, scale, lw, scale).transpose(0, 2, 4, 1, 3)
        ref = packed.reshape(scale, scale, c, lh, lw).transpose(2, 3, 0, 4, 1).reshape(c, h, w)
    expected = np.clip(np.floor(ref * 255 + 0.5), 0, 255).astype(np.float32)
    actual = np.fromfile(args.actual, dtype=np.uint8).reshape(c, h, w)[[2, 0, 1]].astype(np.float32)
    error = actual - expected
    mse = float(np.mean(error * error))
    psnr = float(10 * np.log10(255 ** 2 / mse)) if mse else 999.0
    report = {"shape": [c, h, w], "mae": float(np.abs(error).mean()),
              "max_error": float(np.abs(error).max()), "psnr_db": psnr,
              "min_psnr_db": args.min_psnr, "pass": psnr >= args.min_psnr,
              "scrambled_reference": args.scramble_reference}
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
