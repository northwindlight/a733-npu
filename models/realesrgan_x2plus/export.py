"""Official RealESRGAN_x2plus (23 RRDBs), native x2, static ONNX opset 11.

Architecture follows BasicSR rrdbnet_arch.py (Apache-2.0), revision
8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a. See NOTICE and LICENSE.basicsr.
The input pixel-unshuffle and first 3x3 conv are folded EXACTLY into a
6x6/stride-2 conv, avoiding a rank-six reshape/transpose on the NPU converter.
The original architecture remains the independent numerical reference.
"""
import argparse
import hashlib
import inspect
from pathlib import Path
import urllib.request

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F

WEIGHTS_URL = ('https://github.com/xinntao/Real-ESRGAN/releases/download/'
               'v0.2.1/RealESRGAN_x2plus.pth')
# Filled from the official release and checked on every export.
WEIGHTS_SHA256 = '49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb'


class ResidualDenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        for i in range(1, 6):
            setattr(self, 'conv%d' % i,
                    nn.Conv2d(64 + (i - 1) * 32, 32 if i < 5 else 64, 3, 1, 1))

    def forward(self, x):
        features = [x]
        for i in range(1, 5):
            features.append(F.leaky_relu(getattr(self, 'conv%d' % i)(
                torch.cat(features, 1)), negative_slope=0.2))
        return self.conv5(torch.cat(features, 1)) * 0.2 + x


class RRDB(nn.Module):
    def __init__(self):
        super().__init__()
        self.rdb1 = ResidualDenseBlock()
        self.rdb2 = ResidualDenseBlock()
        self.rdb3 = ResidualDenseBlock()

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_first = nn.Conv2d(12, 64, 3, 1, 1)
        self.body = nn.Sequential(*(RRDB() for _ in range(23)))
        self.conv_body = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.folded = False

    def fold_unshuffle(self):
        first = nn.Conv2d(3, 64, 6, 2, 2)
        with torch.no_grad():
            first.weight.copy_(self.conv_first.weight.reshape(64, 3, 2, 2, 3, 3)
                               .permute(0, 1, 4, 2, 5, 3).reshape(64, 3, 6, 6))
            first.bias.copy_(self.conv_first.bias)
        self.conv_first = first
        self.folded = True

    def forward(self, x):
        feat = self.conv_first(x if self.folded else F.pixel_unshuffle(x, 2))
        feat = feat + self.conv_body(self.body(feat))
        feat = F.leaky_relu(self.conv_up1(F.interpolate(feat, scale_factor=2,
                                                       mode='nearest')), 0.2)
        feat = F.leaky_relu(self.conv_up2(F.interpolate(feat, scale_factor=2,
                                                       mode='nearest')), 0.2)
        return self.conv_last(F.leaky_relu(self.conv_hr(feat), 0.2))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(weights, folded=True):
    if sha256(weights) != WEIGHTS_SHA256:
        raise ValueError('Official weights SHA256 mismatch')
    state = torch.load(str(weights), map_location='cpu', weights_only=True)
    state = state.get('params_ema', state.get('params', state))
    model = RRDBNet()
    model.load_state_dict(state, strict=True)
    if folded:
        model.fold_unshuffle()
    return model.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shape', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--weights')
    args = ap.parse_args()
    c, h, w = map(int, args.shape.split(','))
    if c != 3 or min(h, w) <= 0 or h % 2 or w % 2:
        ap.error('Expected 3,H,W with positive even H and W')
    torch.set_num_threads(2)
    weights = Path(args.weights) if args.weights else Path(__file__).with_name('RealESRGAN_x2plus.pth')
    if not weights.exists():
        urllib.request.urlretrieve(WEIGHTS_URL, str(weights))
    model = load_model(weights)
    original = load_model(weights, folded=False)
    # Test the weight rearrangement, including convolution padding at borders.
    with torch.no_grad():
        for shape in [(1, 3, 16, 20), (1, 3, 22, 18)]:
            x = torch.from_numpy(np.random.default_rng(7).random(shape).astype('float32'))
            a, b = model(x).numpy(), original(x).numpy()
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=2e-5)
            print('Original / folded max error:', np.abs(a - b).max(), flush=True)
    del original
    dummy = torch.zeros(1, c, h, w)
    options = {'dynamo': False} if 'dynamo' in inspect.signature(torch.onnx.export).parameters else {}
    torch.onnx.export(model, dummy, args.output, opset_version=11,
                      input_names=['input'], output_names=['output'], **options)
    graph = onnx.load(args.output)
    onnx.checker.check_model(graph)
    assert not any(n.op_type in ('SpaceToDepth', 'DepthToSpace', 'Transpose', 'Reshape')
                   for n in graph.graph.node), 'Unshuffle should be folded, not converted'
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(args.output, sess_options=opts, providers=['CPUExecutionProvider'])
    x = np.random.default_rng(11).random((1, c, h, w)).astype('float32')
    actual = session.run(None, {'input': x})[0]
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=3e-5)
    print('PyTorch / ORT max error:', np.abs(actual - expected).max(), flush=True)
    print('Parameters:', sum(p.numel() for p in model.parameters()),
          'ONNX bytes:', Path(args.output).stat().st_size, flush=True)


if __name__ == '__main__':
    main()
