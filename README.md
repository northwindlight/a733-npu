# a733-npu

把 EDSR x2 超分模型编成 Amlogic A733（VeriSilicon VIP9000 NanoDI Plus NPU）能直接 load 的 NBG 单文件。

## 流水线

```
super-image/eugenesiow/edsr-base
        │  scripts/export_onnx.py
        ▼
   edsr_x2.onnx (opset 11, 1×3×360×640)
        │  scripts/pipeline.sh  ←  ubuntu-npu:v2.0.10.1 容器
        │     ├ pegasus import onnx          → .json/.data
        │     ├ pegasus quantize uint8       → .quantize（asymmetric_affine）
        │     └ pegasus export ovxlib        → wksp/edsr_x2_uint8/.../network_binary.nb
        ▼
        部署到 A733（720p 实时 SR：360×640 → 720×1280）
```

## 仓库布局

```
.github/workflows/build_nbg.yml      ← GitHub Actions 触发完整流水线
scripts/
  env.sh                             ← source env.sh v3 设置 NPU target
  export_onnx.py                     ← PyTorch → ONNX
  patch_inputmeta.py                 ← 把 normalize.json 写进 pegasus 生成的 inputmeta
  pipeline.sh <name> [v1|v2|v3] [CHW] ← 容器内主流程
models/edsr_x2/
  normalize.json                     ← mean/scale/reverse_channel（每模型一份，必备）
  prepare_calib.py                   ← 拉 DIV2K 8 张 → 360×640 → dataset.txt
ubuntu-npu_v2.0.10.1.tar.zip         ← VSI acuity-toolkit Docker 镜像（gitignored，本地保存）
```

## NPU target

`scripts/env.sh` 里硬编码三档（来自 amlogic ai-sdk-main/models/env.sh 的 `pegasus_setup.sh`）：

| 档位 | VSIMULATOR_CONFIG | shader_core | 用途 |
|------|-------------------|-------------|------|
| v1 | `VIP9000PICO_PID0XEE` | 0.5 | 入门 NPU |
| v2 | `VIP9000NANOSI_PLUS_PID0X10000016` | 1 | 中端 |
| v3 | `VIP9000NANODI_PLUS_PID0X1000003B` | 1 | **A733** |

`--optimize` 接的是这个字符串而不是数字 —— acuity 用它选目标 NPU 的指令调度。

## 本地复现（self-hosted alpine 上）

```bash
# 一次性：把 docker 镜像装进去
docker load < ubuntu-npu_v2.0.10.1.tar.zip   # 或 docker pull ghcr.io/northwindlight/ubuntu-npu:v2.0.10.1

# 阶段 1：导出 ONNX（任意 python 3.11 环境）
pip install torch==2.0.1 onnx onnxruntime super-image pillow
python3 scripts/export_onnx.py --output models/edsr_x2/edsr_x2.onnx

# 阶段 2：校准数据 + pegasus 转换
docker run --rm -v "$PWD/models/edsr_x2:/wk" -w /wk ubuntu-npu:v2.0.10.1 \
    python3 prepare_calib.py --n 8

docker run --rm -v "$PWD:/workspace" ubuntu-npu:v2.0.10.1 \
    bash /workspace/scripts/pipeline.sh edsr_x2 v3
```

产物：`models/edsr_x2/wksp/edsr_x2_uint8/<name>/network_binary.nb`

## CI 注意事项

- `pegasus-convert` job 需要 `[self-hosted, linux, x64]` runner，且 6G+ 内存。
- 镜像不在 self-hosted runner 上时，先 `docker pull ghcr.io/northwindlight/ubuntu-npu:v2.0.10.1`。
- `gen_nbg` 阶段是 CPU 模拟器，单模型 ~4h（EDSR 类视觉模型典型值），workflow 里 `timeout-minutes: 720`。

## 校准说明

EDSR 的输入是 0..255 像素直接喂网络（不做 mean/std 归一化），所以 `models/edsr_x2/normalize.json`
里 `mean=[0,0,0], scale=[1,1,1], reverse_channel=false`。

换模型时改 `normalize.json` 即可，pipeline.sh 会自动 patch 进 pegasus 生成的 inputmeta：

| 模型类型 | mean | scale | reverse_channel |
|----------|------|-------|-----------------|
| super-image / EDSR / SR 类 | [0,0,0] | [1,1,1] | false |
| torchvision ImageNet 预训练（ResNet/MobileNet） | [123.675,116.28,103.53] | [0.0171,0.0175,0.0174] | false |
| caffe / OpenCV 风格（输入 BGR） | 同上 | 同上 | true |
| YOLO v3/v5（pixel/255） | [0,0,0] | [0.0039,0.0039,0.0039] | false |

校准用 DIV2K 验证集前 8 张，center-crop 到 360×640。`asymmetric_affine + uint8 + normal` 算法
够用；如果 PSNR 掉超过 1dB，再考虑 `--algorithm kl_divergence` 或 `--MLE`。
