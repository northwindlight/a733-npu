#!/usr/bin/env python3
"""Real-ESRGAN `realesr-general-x4v3` (x4) → ONNX opset 11。

和隔壁 `animevideov3` 是**同一个类**（SRVGGNetCompact），只是 `num_conv` 16→32。
所以这里不复制那 150 行架构代码，只把权重信息塞进 argv 再调它。

★ 为什么要这个模型：
  `realesr-animevideov3` 是给**动画**调的（训练集是平面色块 + 硬线条），
  喂**实拍**素材会把纹理抹成塑料块 —— 俗称"油画感"，且边缘还原不出来。
  `realesr-general-x4v3` 训练集换成通用/实拍内容，同一套架构，
  所以 NBG 编译流程、板端 srpipe、Web 服务全都不用改。

★ 代价：num_conv 16→32 ⇒ 约 **2× 慢**（NPU 代价仍 ∝ 源总像素，
  只是每像素的常数从 ~0.547 µs 涨到 ~1.09 µs）。

★ 这里的 sha256/大小是**从权重文件实测出来的**，不是抄的：
  realesr-general-x4v3.pth 是 zip+pickle，解开 data.pkl 读 _rebuild_tensor_v2
  的 size 参数，测得 num_conv=32 / num_feat=64 / upscale=4，
  checkpoint 顶层 key 是 `params`（animevideov3 是 `params_ema`）。

usage:（由 CI 调用，不是手工）
    python3 export.py --shape 3,180,320 --output realesr_general.onnx
"""
import pathlib
import runpy
import sys

sys.argv += [
    "--weights-url",
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.5.0/realesr-general-x4v3.pth",
    "--weights-sha256",
    "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
    "--weights-size", "4885111",
    "--weights-name", "realesr-general-x4v3.pth",
    "--num-conv", "32",
]

_SIBLING = pathlib.Path(__file__).resolve().parent.parent / "animevideov3" / "export.py"
runpy.run_path(str(_SIBLING), run_name="__main__")
