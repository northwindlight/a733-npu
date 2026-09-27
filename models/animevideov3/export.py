"""Real-ESRGAN `realesr-animevideov3` (x4) → ONNX opset 11。

这是用户 Win Server 上 sr-tool 用的同一个模型（`-Model realesr-animevideov3`），
所以它的板上帧率可以和核显的 2.9 s/帧 直接对比。

★ 为什么把架构内联进来而不是 `pip install realesrgan basicsr`：
  basicsr 依赖链历史上反复被 torchvision 的内部改动打断，而且会拖慢 CI。
  这里只需要一个 60 行的类，内联更可控。

★ DepthToSpace 的 mode 必须是 "CRD"
  PyTorch 的 pixel_shuffle ≡ ONNX DepthToSpace(mode="CRD")，而 ONNX 的**默认是 "DCR"**。
  忽略 mode 的后端会给出**形状正确、通道错乱**的输出 —— 表现为网格/拼贴伪影，
  而且不报错。这是 Real-ESRGAN 转 ONNX 的经典翻车点。本脚本显式校验，
  下游还有一道数值比对闸（见 README「验证」）。

usage:
    python3 export.py --shape 3,180,320 --output animevideov3.onnx
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# SRVGGNetCompact — 摘自 xinntao/Real-ESRGAN (Apache-2.0) 的
# realesrgan/archs/srvgg_arch.py，去掉 basicsr 的 ARCH_REGISTRY 装饰器，
# 其余逐字保留（含 forward 的残差实现）。
# https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/archs/srvgg_arch.py
# ---------------------------------------------------------------------------


class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.

    It is a compact network structure, which performs upsampling in the last
    layer and no convolution is conducted on the HR feature space.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16,
                 upscale=4, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            # activation
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out


# ---------------------------------------------------------------------------

WEIGHTS_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
               "v0.2.5.0/realesr-animevideov3.pth")
WEIGHTS_SHA256 = "b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d"
WEIGHTS_SIZE = 2504012


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_weights(dest):
    if dest.exists() and sha256_of(dest) == WEIGHTS_SHA256:
        print("权重已缓存且哈希正确：%s" % dest)
        return dest
    print("下载 %s" % WEIGHTS_URL)
    urllib.request.urlretrieve(WEIGHTS_URL, dest)
    got = sha256_of(dest)
    if got != WEIGHTS_SHA256:
        if dest.exists():            # 不用 missing_ok=，那是 Python 3.9+ 的 API
            dest.unlink()
        sys.exit("权重 sha256 不符：\n  期望 %s\n  实际 %s\n"
                 "上游换了文件，或下载被截断。已删除该文件。" % (WEIGHTS_SHA256, got))
    if dest.stat().st_size != WEIGHTS_SIZE:
        sys.exit("权重大小异常：%d（期望 %d）" % (dest.stat().st_size, WEIGHTS_SIZE))
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True, help="CHW，逗号分隔，例如 3,180,320")
    ap.add_argument("--output", required=True)
    ap.add_argument("--scale", type=int, default=4,
                    help="v3 官方只发 x4；x2/x3 的 ncnn 版本只是 x4 加了个缩放，不是独立模型")
    ap.add_argument("--weights", default=None, help="本地 .pth 路径（默认下载到本目录）")
    args = ap.parse_args()

    c, h, w = (int(x) for x in args.shape.split(","))

    here = Path(__file__).resolve().parent
    wpath = Path(args.weights) if args.weights else here / "realesr-animevideov3.pth"
    fetch_weights(wpath)

    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16,
                            upscale=args.scale, act_type='prelu')
    ckpt = torch.load(str(wpath), map_location="cpu")
    # 官方 .pth 是完整 checkpoint，key 视发布版本可能是 params_ema 或 params
    key = "params_ema" if "params_ema" in ckpt else "params"
    model.load_state_dict(ckpt[key], strict=True)
    model.eval()
    print("权重 key=%s，参数量 %.2f M" % (key, sum(p.numel() for p in model.parameters()) / 1e6))

    dummy = torch.randn(1, c, h, w)
    # torch >= 2.6 默认走 dynamo 导出路径（符号映射表不同）。这里 pin 的是 2.0.1，
    # 默认就是 TorchScript 导出器 —— 本文档开头引用的 DepthToSpace(CRD) 映射
    # 正是那条路径的行为。若将来升级 torch，请显式传 dynamo=False 保持同一路径。
    torch.onnx.export(model, dummy, args.output, opset_version=11,
                      input_names=["input"], output_names=["output"])

    m = onnx.load(args.output)
    print("opset: %d" % m.opset_import[0].version)
    n_d2s = 0
    for n in m.graph.node:
        if n.op_type == "DepthToSpace":
            n_d2s += 1
            attrs = {a.name: (a.i if a.type == onnx.AttributeProto.INT else a.s)
                     for a in n.attribute}
            print("  DepthToSpace: %r" % attrs)
            if attrs.get("mode") != b"CRD":
                sys.exit("FAIL: DepthToSpace.mode=%r，期望 b'CRD'。\n"
                         "  ONNX 默认是 'DCR'，两者通道排布不同 —— 用错会得到"
                         "形状正确但通道错乱的图，而且不会报错。"
                         % attrs.get("mode"))
    if n_d2s == 0:
        sys.exit("FAIL: 图里没有 DepthToSpace —— PixelShuffle 没被导出成预期算子，"
                 "检查 torch/onnx 版本。")

    sess = ort.InferenceSession(args.output)
    x = np.random.randn(1, c, h, w).astype(np.float32)
    y = sess.run(None, {"input": x})
    print("OK: %s -> %s  (%.1f KiB)"
          % (x.shape, y[0].shape, Path(args.output).stat().st_size / 1024))


if __name__ == "__main__":
    main()
