#!/bin/bash
# ONNX → Acuity → uint8 量化 → NBG 单 .nb 文件
# 目标：全志 A733 NPU (Allwinner A733, VIP9000 NanoDI Plus, v3)
#
# 用法（容器内）：
#   docker run --rm -v $PWD:/workspace ubuntu-npu:v2.0.10.1 \
#       bash /workspace/scripts/pipeline.sh edsr_x2 [v3]
#
# 期望目录结构：
#   models/<name>/<name>.onnx          ← 输入
#   models/<name>/normalize.json       ← mean/scale/reverse_channel（每模型一份）
#   models/<name>/dataset.txt          ← 校准样本路径列表
#   models/<name>/calib/*.jpg|*.png    ← 校准图（dataset.txt 引用）
# 产物：
#   models/<name>/wksp/<name>_uint8/network_binary.nb
set -euo pipefail

NAME=${1:?usage: pipeline.sh <model-name> [v1|v2|v3] [CHW]}
NPU=${2:-v3}
SHAPE=${3:-3,360,640}

# 仓库根 = 脚本上一级
ROOT=$(cd "$(dirname "$0")/.." && pwd)

# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh" "$NPU"

PEGASUS="python3 $ACUITY_PATH/pegasus.py"
WORK="$ROOT/models/$NAME"
[ -d "$WORK" ] || { echo "no such model dir: $WORK" >&2; exit 1; }
[ -f "$WORK/$NAME.onnx" ] || { echo "missing $WORK/$NAME.onnx" >&2; exit 1; }
[ -f "$WORK/normalize.json" ] || { echo "missing $WORK/normalize.json — see models/edsr_x2/normalize.json" >&2; exit 1; }
[ -f "$WORK/dataset.txt" ] || { echo "missing $WORK/dataset.txt — run 'python3 prepare_calib.py' first" >&2; exit 1; }

cd "$WORK"

echo "[0/4] clean intermediate"
rm -f "$NAME.json" "$NAME.data" "${NAME}_uint8.quantize" "${NAME}_inputmeta.yml"
rm -rf wksp

echo "[1/4] import onnx → acuity ir"
$PEGASUS import onnx \
    --model "$NAME.onnx" \
    --output-model "$NAME.json" \
    --output-data  "$NAME.data" \
    --inputs input --outputs output \
    --input-size-list "$SHAPE"

echo "[2/4] inputmeta (auto-generate, patch from normalize.json)"
# 真实 input lid 由 onnx 节点编号决定（如 input_103），不能写死。
# pegasus 生成模板，再用 normalize.json 覆盖 mean/scale/reverse_channel。
$PEGASUS generate inputmeta \
    --model "$NAME.json" \
    --input-meta-output "${NAME}_inputmeta.yml"
python3 "$ROOT/scripts/patch_inputmeta.py" \
    "${NAME}_inputmeta.yml" normalize.json

echo "[3/4] quantize uint8 (asymmetric_affine)"
$PEGASUS quantize \
    --model         "$NAME.json" \
    --model-data    "$NAME.data" \
    --device        CPU \
    --with-input-meta "${NAME}_inputmeta.yml" \
    --rebuild \
    --model-quantize  "${NAME}_uint8.quantize" \
    --quantizer       asymmetric_affine \
    --qtype           uint8

echo "[4/4] export ovxlib + pack NBG (target=$VSIMULATOR_CONFIG)"
# 仅在 export 阶段加 LD_LIBRARY_PATH —— vsimulator/lib 含独立 libtorch_cpu/libc10，
# 会和 acuitylib 自身的 torch 冲突，所以前几步不能加。
LD_LIBRARY_PATH="$VIV_SDK/vsimulator/lib:$VIV_SDK/common/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
$PEGASUS export ovxlib \
    --model           "$NAME.json" \
    --model-data      "$NAME.data" \
    --model-quantize  "${NAME}_uint8.quantize" \
    --dtype           quantized \
    --target-ide-project linux64 \
    --with-input-meta "${NAME}_inputmeta.yml" \
    --pack-nbg-unify \
    --optimize        "$VSIMULATOR_CONFIG" \
    --viv-sdk         "$VIV_SDK" \
    --output-path     "wksp/${NAME}_uint8/$NAME"

NB=$(find wksp -name '*.nb' | head -1)
if [ -z "$NB" ]; then
    echo "FAIL: no .nb produced" >&2
    exit 1
fi
echo "OK: $NB ($(du -h "$NB" | cut -f1))"
