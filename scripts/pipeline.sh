#!/bin/bash
# ONNX → Acuity → 量化 → NBG 单 .nb 文件
#
# 模型的一切声明都在 models/<name>/model.json 里（输入/输出名、形状、归一化、
# 量化精度、校准源）。本脚本不含任何模型专属常量。
#
# 用法（容器内）：
#   docker run --rm -v $PWD:/workspace -w /workspace \
#       ghcr.io/northwindlight/ubuntu-npu@sha256:<digest> \
#       bash scripts/pipeline.sh <model-name> [v1|v2|v3]
#
#   NPU 目标优先级：命令行 $2 > model.json 的 "npu" > v3
#
# 期望目录结构（详见 README「加一个模型」）：
#   models/<name>/model.json
#   models/<name>/<name>.onnx      ← 产物，由 export.py 生成 / 下载 / 拷贝
#   models/<name>/dataset.txt      ← prepare_calib.py 生成
#   models/<name>/calib/*.png      ← 同上
#
# 调试用 --stop-after 可以在任意阶段停下，避免每改一行就等一轮完整编译：
#   bash scripts/pipeline.sh edsr_x2 v3 --stop-after inputmeta
set -euo pipefail

NAME=${1:?usage: pipeline.sh <model-name> [v1|v2|v3] [--stop-after STAGE]}
NPU_ARG=${2:-}
STOP_AFTER=""
shift $(( $# > 2 ? 2 : $# ))
while [ $# -gt 0 ]; do
    case "$1" in
        --stop-after) STOP_AFTER=${2:?--stop-after 需要阶段名}; shift 2 ;;
        *) echo "未知参数: $1" >&2; exit 2 ;;
    esac
done

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK="$ROOT/models/$NAME"
CFG="$WORK/model.json"

die() { echo "FAIL: $*" >&2; exit 1; }

# ---------------------------------------------------------------- [0/5] 前置校验
# L2：只检查本脚本真正消费的东西，必须 Python 3.8 兼容且不依赖 onnx。
echo "[0/5] preflight"
[ -d "$WORK" ] || die "没有这个模型目录：$WORK"
[ -f "$CFG" ]  || die "缺少 $CFG
  ★最常见成因：.gitignore 的 'models/*/*.json' 把它吞了。
    确认文件真的进 git 了：git check-ignore -v $CFG"

cfg() { python3 "$ROOT/scripts/model_config.py" "$CFG" "$@" || die "model.json 解析失败（见上）"; }
mapfile -t IMPORT_ARGS   < <(cfg import-args)
mapfile -t QUANT_ARGS    < <(cfg quant-args)
mapfile -t CALIB_ARGS    < <(cfg calib-args)

# NPU 目标：命令行 > model.json > v3
# 用命令替换而不是 `| head -1` —— 管道在 set -o pipefail 下有 SIGPIPE 风险（见 [5/5] 的注释）
NPU="$NPU_ARG"
if [ -z "$NPU" ]; then
    NPU=$(python3 "$ROOT/scripts/model_config.py" "$CFG" get npu 2>/dev/null || true)
    NPU=${NPU:-v3}
fi
case "$NPU" in v1|v2|v3) ;; *) die "NPU 目标必须是 v1/v2/v3，实际 '$NPU'";; esac

ONNX="$WORK/$NAME.onnx"
[ -s "$ONNX" ] || die "缺少或为空的 ONNX：$ONNX
  ONNX 必须先物化到这里（model.json 的 onnx.from 决定怎么来）"
[ -s "$WORK/dataset.txt" ] || die "缺少 $WORK/dataset.txt —— 先跑 scripts/prepare_calib.py"
[ -d "$WORK/calib" ] || die "缺少 $WORK/calib/ —— 先跑 scripts/prepare_calib.py"

# dataset.txt 的内容也要查：旧版本只检查文件存在，里面指向不存在的图会一路带病编译
missing=0
while IFS= read -r line; do
    [ -z "$line" ] && continue
    [ -f "$WORK/${line#./}" ] || { echo "  dataset.txt 指向的文件不存在: $line" >&2; missing=$((missing+1)); }
done < "$WORK/dataset.txt"
[ "$missing" -eq 0 ] || die "dataset.txt 里有 $missing 个失效路径"
echo "  OK: onnx=$(du -h "$ONNX" | cut -f1)  calib=$(find "$WORK/calib" -type f | wc -l) 张  npu=$NPU"

# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh" "$NPU"
PEGASUS="python3 $ACUITY_PATH/pegasus.py"

# 量化精度来自 model.json；_uint8/_int16 后缀决定中间产物名
QTYPE=$(cfg get quant.qtype)
QSUFFIX="$QTYPE"
QNAME="${NAME}_${QSUFFIX}.quantize"

cd "$WORK"

echo "[0/5] clean intermediate"
rm -f "$NAME.json" "$NAME.data" "$QNAME" "${NAME}_inputmeta.yml"
rm -rf wksp

stop_here() {   # $1 = 刚完成的阶段
    if [ -n "$STOP_AFTER" ] && [ "$STOP_AFTER" = "$1" ]; then
        echo "--stop-after $1：在此停下"
        exit 0
    fi
}

echo "[1/5] import onnx → acuity ir"
$PEGASUS import onnx \
    --model "$NAME.onnx" \
    --output-model "$NAME.json" \
    --output-data  "$NAME.data" \
    "${IMPORT_ARGS[@]}"
stop_here import

echo "[2/5] inputmeta (auto-generate, patch from model.json)"
# 真实 input lid 由 onnx 节点编号决定（如 input_103），不能写死。
# pegasus 生成模板，再用 model.json 的 normalize 段覆盖 mean/scale/reverse_channel。
# 直接传 model.json：patch_inputmeta 会取其中的 "normalize" 子对象
# （旧的扁平 normalize.json 也仍然支持 —— 这条兼容是回归测试的基础）。
$PEGASUS generate inputmeta \
    --model "$NAME.json" \
    --input-meta-output "${NAME}_inputmeta.yml"
python3 "$ROOT/scripts/patch_inputmeta.py" \
    "${NAME}_inputmeta.yml" "$CFG"
stop_here inputmeta

echo "[3/5] quantize $QTYPE"
$PEGASUS quantize \
    --model         "$NAME.json" \
    --model-data    "$NAME.data" \
    --device        CPU \
    --with-input-meta "${NAME}_inputmeta.yml" \
    --rebuild \
    --model-quantize  "$QNAME" \
    "${QUANT_ARGS[@]}"
stop_here quantize

echo "[4/5] export ovxlib + pack NBG (target=$VSIMULATOR_CONFIG)"
# LD_LIBRARY_PATH 顺序很关键：torch/lib 必须在 vsimulator/lib 前面，
# 否则 vsimulator 的 libc10.so/libtorch_cpu.so 会覆盖 Python torch 的版本导致 import 崩溃。
TORCH_LIB="/usr/local/lib/python3.8/dist-packages/torch/lib"
LD_LIBRARY_PATH="$TORCH_LIB:$VIV_SDK/vsimulator/lib:$VIV_SDK/common/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
$PEGASUS export ovxlib \
    --model           "$NAME.json" \
    --model-data      "$NAME.data" \
    --model-quantize  "$QNAME" \
    --dtype           quantized \
    --target-ide-project linux64 \
    --with-input-meta "${NAME}_inputmeta.yml" \
    --pack-nbg-unify \
    --optimize        "$VSIMULATOR_CONFIG" \
    --viv-sdk         "$VIV_SDK" \
    --output-path     "wksp/${NAME}_${QSUFFIX}/$NAME"
stop_here export

echo "[5/5] locate NBG"
# ★不要写回 `find ... | head -1`：在 set -euo pipefail 下，find 多输出一行就会被
#   SIGPIPE 杀掉（退出码 141），于是脚本在下面的守卫之前就中止了 —— 静默失败、没有诊断。
#   而且 head -1 对多结果是非确定的。收集成数组再显式判定。
mapfile -t NBS < <(find wksp -type f -name '*.nb' -print)
if [ "${#NBS[@]}" -eq 0 ]; then
    echo "FAIL: 没产出 .nb。wksp 内容：" >&2
    find wksp -type f | sort >&2 || true
    exit 1
fi
if [ "${#NBS[@]}" -gt 1 ]; then
    echo "WARN: 产出多个 .nb，取第一个（按路径排序）：" >&2
    printf '  %s\n' "${NBS[@]}" >&2
elif [ ! -s "${NBS[0]}" ]; then
    die "产出的 .nb 是空文件：${NBS[0]}"
fi
NB="${NBS[0]}"
echo "OK: $NB ($(du -h "$NB" | cut -f1))"

# ---------------------------------------------------------------- build_info
GIT_SHA_ARG=()
# ★不要写成 `[ -n "$X" ] && ARR=(...)`：set -e 下测试为假会让整条列表返回 1 直接退出，
#   而那正是"没传 GIT_SHA"的正常情况。
if [ -n "${GIT_SHA:-}" ]; then
    GIT_SHA_ARG=(--git-sha "$GIT_SHA")
fi
python3 "$ROOT/scripts/make_build_info.py" \
    --model "$NAME" --config "$CFG" --npu "$NPU" "${GIT_SHA_ARG[@]}" \
    --onnx "$ONNX" --nbg "$NB" \
    --qtype "$QTYPE" \
    --vsconfig "$VSIMULATOR_CONFIG" \
    --out "$WORK/build_info.json"
echo "wrote $WORK/build_info.json"
