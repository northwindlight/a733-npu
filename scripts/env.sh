#!/bin/bash
# 设定 pegasus / VSI SDK 环境与 NPU target。
# 用法: source scripts/env.sh v3
#
# v1/v2/v3 对应不同 NPU IP（参考 amlogic ai-sdk-main/models/env.sh）：
#   v1 → VIP9000PICO_PID0XEE                  (shader_core 0.5)
#   v2 → VIP9000NANOSI_PLUS_PID0X10000016     (shader_core 1)
#   v3 → VIP9000NANODI_PLUS_PID0X1000003B     (shader_core 1)  ← A733
#
# 要在 ubuntu-npu:v2.0.10.1 容器里 source 才有意义。

case "${1:-v3}" in
  v1) export VSIMULATOR_SHADER_CORE_COUNT=0.5
      export VSIMULATOR_CONFIG=VIP9000PICO_PID0XEE ;;
  v2) export VSIMULATOR_SHADER_CORE_COUNT=1
      export VSIMULATOR_CONFIG=VIP9000NANOSI_PLUS_PID0X10000016 ;;
  v3) export VSIMULATOR_SHADER_CORE_COUNT=1
      export VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B ;;
  *)  echo "unknown NPU target: $1 (expect v1/v2/v3)" >&2; return 1 ;;
esac

export ACUITY_PATH=/usr/local/acuity_command_line_tools
export VIV_SDK=/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools

# 不要在这里设 LD_LIBRARY_PATH —— vsimulator/lib 里的 libc10/libtorch_cpu
# 会污染 acuitylib import 阶段的 torch。pipeline.sh 在 export ovxlib 步骤
# 单独包裹 LD_LIBRARY_PATH。

echo "NPU target: $VSIMULATOR_CONFIG (shader_core=$VSIMULATOR_SHADER_CORE_COUNT)"
