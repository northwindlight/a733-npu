# a733-npu

EDSR x2 超分模型 → 全志 A733 NPU 编译 + 推理全流程。

```
PyTorch (super-image) → ONNX opset11 → pegasus/acuity → uint8 quant → NBG → A733 真机
```

**实测：360×640 → 720×1280，纯 NPU 延迟 236ms（~4.2 fps），单核 VIP9000 NanoDI Plus。**

## 仓库

```
├── .github/workflows/build_nbg.yml    CI：ONNX 导出 + pegasus NBG 编译
├── scripts/
│   ├── env.sh                          source env.sh v3 → VSIMULATOR_CONFIG
│   ├── export_onnx.py                  PyTorch → ONNX (opset 11)
│   ├── patch_inputmeta.py              normalize.json → inputmeta.yml
│   └── pipeline.sh <name> [v3] [CHW]   ONNX → NBG 主流程
├── models/edsr_x2/
│   ├── normalize.json                  mean/scale/reverse_channel（每模型一份）
│   └── prepare_calib.py               DIV2K 校准图 → dataset.txt
└── README.md
```

## 快速开始

### 1. ONNX 导出（任意 Python 3.11 环境）

```bash
pip install torch==2.0.1 onnx onnxruntime super-image pillow
python3 scripts/export_onnx.py --output models/edsr_x2/edsr_x2.onnx
```

### 2. NBG 编译（ubuntu-npu:v2.0.10.1 容器）

```bash
docker pull ghcr.io/northwindlight/ubuntu-npu:v2.0.10.1  # 或者 docker load < 本地镜像

# 准备校准数据
docker run --rm -v "$PWD/models/edsr_x2:/wk" -w /wk \
    ghcr.io/northwindlight/ubuntu-npu:v2.0.10.1 \
    python3 prepare_calib.py --n 8

# 编译
docker run --rm -v "$PWD:/workspace" \
    ghcr.io/northwindlight/ubuntu-npu:v2.0.10.1 \
    bash /workspace/scripts/pipeline.sh edsr_x2 v3

# 产物：models/edsr_x2/wksp/*_nbg_unify/network_binary.nb
```

### 3. A733 真机推理

```bash
# radxa 上
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/northwind/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0

# 编译快速推理程序
cd ai-sdk/examples/lenet
gcc -O2 -o /tmp/edsr_video /path/to/edsr_video.c \
    ../libawnn_viplite/awnn_lib.c ../libawnn_viplite/awnn_quantize.c \
    -I../libawnn_viplite -I../libawutils -I../.. \
    -I../../viplite-tina/lib/aarch64-none-linux-gnu/v2.0/inc \
    -L../../viplite-tina/lib/aarch64-none-linux-gnu/v2.0/ -lNBGlinker -lVIPhal -lm

# 帧数据用 .dat 后缀（纯 binary uint8 NCHW，691200 字节）
/tmp/edsr_video network_binary.nb frames_in/ frames_out/

# 输出是 uint8 量化值，显示前需 dequant：
# pixel = (quantized_value - zero_point) * scale
# EDSR x2: scale=2.646, zero_point=103
```

## 实测结果

| 模型 | 输入 | 输出 | 纯 NPU 延迟 | 有效 FPS |
|------|------|------|-----------|---------|
| LeNet | 28×28×1 | 10 类 | 0.25ms | 4000 |
| ShuffleNetV2 | 224×224×3 | 1000 类 | 2.9ms | 343 |
| YOLOv5s | 640×640×3 | 检测 | 25.5ms | 39 |
| **EDSR x2** | **360×640×3** | **720×1280×3** | **236ms** | **4.2** |

A733 NPU 规格：VIP9000 NanoDI Plus，单核，VIP SRAM 296KB，Memory Pool 84MB（EDSR）。

## NPU target

`scripts/env.sh` 三档（来自 ai-sdk/models/env.sh）：

| 档位 | VSIMULATOR_CONFIG | NPU IP |
|------|-------------------|--------|
| v1 | `VIP9000PICO_PID0XEE` | Pico |
| v2 | `VIP9000NANOSI_PLUS_PID0X10000016` | NanoSI Plus |
| v3 | `VIP9000NANODI_PLUS_PID0X1000003B` | NanoDI Plus (**A733**) |

`nbinfo` 输出 `Target: 0x1000003b` 即 v3。**`--optimize` 接的是这个字符串，不是数字。**

## 归一化参数（normalize.json）

换模型时只改这个文件即可，`pipeline.sh` 会自动 patch 进 pegasus 生成的 inputmeta：

| 模型 | mean | scale | reverse_channel |
|------|------|-------|-----------------|
| super-image / EDSR | `[0,0,0]` | `[1,1,1]` | `false` |
| torchvision ImageNet | `[123.675,116.28,103.53]` | `[0.0171,0.0175,0.0174]` | `false` |
| caffe / OpenCV (BGR) | 同上 | 同上 | `true` |
| YOLO v3/v5 | `[0,0,0]` | `[0.0039,0.0039,0.0039]` | `false` |

## CI

Workflow 手动触发（`workflow_dispatch`），分两阶段：
1. **export-onnx**（ubuntu-22.04）：导出 ONNX → artifact
2. **pegasus-convert**（ubuntu-22.04）：login ghcr → pull 镜像 → pipeline.sh → NBG artifact

镜像 `ghcr.io/northwindlight/ubuntu-npu` private，需在 GitHub 包页面关联到 `a733-npu` 仓库，CI 用 `GITHUB_TOKEN` 自动拉。`timeout-minutes: 720`（EDSR ~4h）。

---

## 踩坑记录

以下是全流程中**非显然的陷阱**，每个都浪费了 4h+ 的 CI 跑。

### pegasus / acuity 侧

**1. ONNX dynamic_axes 必须显式传 `--input-size-list`**
ONNX 有 dynamic batch/H/W 时，acuity 不会自己解析尺寸，报 `Miss input size list`。必须传 `--input-size-list 3,360,640`（CHW 格式）。

**2. inputmeta 的 `lid` 是 import 后才确定的**
不要 checkin 固定 lid 的 inputmeta.yml。`pegasus generate inputmeta` 自动拿正确的 lid（如 `input_103`），然后用 `normalize.json` patch 归一化字段即可。

**3. LD_LIBRARY_PATH 顺序：torch/lib 必须在 vsimulator/lib 前面**
vsimulator/lib 里有自己的 `libc10.so` / `libtorch_cpu.so`，会跟 Python torch 冲突。设置 `LD_LIBRARY_PATH` 时把 torch 的 lib 放在最前面：

```bash
TORCH_LIB="/usr/local/lib/python3.8/dist-packages/torch/lib"
LD_LIBRARY_PATH="$TORCH_LIB:$VIV_SDK/vsimulator/lib:$VIV_SDK/common/lib"
```

**4. `--pack-nbg-unify` 会自动给 output-path 追加 `_nbg_unify` 后缀**
产物在 `wksp/<name>_uint8_nbg_unify/network_binary.nb`，不是 `wksp/<name>_uint8/<name>/`。

### radxa 真机侧

**5. vpm_run 输入用 `.dat` 后缀（纯 binary uint8 NCHW，无 header）**
`.tensor` 是 ASCII 文本格式（每行一个数），vpm_run 据此判定格式。`.raw` → segfault，`.dat` → 正常。简单 dd / numpy tofile 出来的 binary 只要改后缀就行。

**6. vpm_run 输出必写 56MB ASCII 文本，循环用它做视频会巨慢**
`--save_txt 1` 把 2.76M 个 float 写成 ASCII 字符串，单帧额外 ~10s。**循环推理必须用 C 直调 VIP API**（awnn_lib 包装层），一次 load NBG，循环只换输入，输出拿 raw buffer 避免任何格式转换。

**7. 推理输出是 uint8 量化值，显示前要 dequant**
`pixel = (uint8_value - zero_point) * scale`。EDSR x2 输出 `scale=2.646, zero_point=103`，不做 dequant 画面发白（偏高）或发暗。

**8. viplite-tina API 使用模式**
```c
awnn_init();
Awnn_Context_t *ctx = awnn_create(nbg_path);  // 一次 load
for (each frame) {
    awnn_set_input_buffers(ctx, input_buffers);  // ~0.2ms
    awnn_run(ctx);                                // ~236ms (EDSR)
    void *raw_out = awnn_get_output_buffer(ctx, 0);
    // raw_out 是 uint8 量化值，需要 dequant 后显示
}
awnn_destroy(ctx);
awnn_uninit();
```

**9. LD_LIBRARY_PATH 要包含 viplite-tina 路径**
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/viplite-tina/lib/aarch64-none-linux-gnu/v2.0
```

### Git 侧

**10. 大二进制不要进 git**
2.7GB docker 镜像用 ghcr（private 包）分发，`.`gitignore` 应 block `*.zip/*.tar/*.nb/*.onnx`。一旦误提交，git filter-repo 或 orphan 重建才能彻底瘦身。
