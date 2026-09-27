"""把归一化参数（mean/scale/reverse_channel）写进 pegasus 自动生成的 inputmeta.yml。

为什么不直接 checkin 整份 inputmeta：input lid 由 onnx 节点编号决定（如 input_103），
不同模型不同。pegasus generate inputmeta 给出正确 lid，这里只 patch 归一化字段。

配置来源支持两种形状（见下方 _extract_normalize）：
  扁平：{"mean": [...], "scale": [...], "reverse_channel": false}      ← 旧的 normalize.json
  嵌套：{"schema": 1, ..., "normalize": {"mean": [...], ...}}           ← 新的 model.json

字段含义：
  mean:            [c0, c1, ...] 每通道均值，单位与训练时一致
  scale:           [c0, c1, ...] 每通道缩放，pegasus 计算 (pixel - mean) * scale
  reverse_channel: true=BGR（torchvision/caffe 风格），false=RGB（super-image/手写代码常见）

usage: python3 patch_inputmeta.py <inputmeta.yml> <normalize.json|model.json>
"""
import json
import sys
from pathlib import Path


def _extract_normalize(data: dict, src: Path) -> dict:
    """兼容扁平（旧 normalize.json）与嵌套（新 model.json）两种配置形状。

    这一行式的兼容是故意的：它让新旧配置能在同一棵树里共存，因而可以拿同一个
    inputmeta 模板分别跑两条路径、逐字节 diff —— 把改造回归从"跑一轮 4 小时"
    压缩成"两次秒级调用"。
    """
    norm = data.get("normalize", data)

    if not isinstance(norm, dict):
        raise ValueError(f"{src}: 'normalize' 必须是对象，实际是 {type(norm).__name__}")
    for key in ("mean", "scale"):
        if key not in norm:
            raise ValueError(f"{src}: 缺少 '{key}'（也要检查是不是写在了 'normalize' 外面）")
        if not isinstance(norm[key], list) or not norm[key]:
            raise ValueError(f"{src}: '{key}' 必须是非空数组，实际是 {norm[key]!r}")
    if len(norm["mean"]) != len(norm["scale"]):
        raise ValueError(
            f"{src}: mean 有 {len(norm['mean'])} 项但 scale 有 {len(norm['scale'])} 项，"
            f"两者必须等长且等于输入通道数")
    if "reverse_channel" not in norm:
        # 不默认。patch 阶段静默取 False 会让 'revers_channel' 这类拼写错误
        # 产出"看起来正常、其实通道反了"的 NBG。宁可在这里硬失败。
        raise ValueError(
            f"{src}: 缺少 'reverse_channel'。它没有默认值 —— "
            f"拼错成 'revers_channel' 会静默按 RGB 编译出通道颠倒的模型。"
            f"显式写 \"reverse_channel\": false（RGB）或 true（BGR）")
    if not isinstance(norm["reverse_channel"], bool):
        raise ValueError(f"{src}: 'reverse_channel' 必须是 true/false，"
                         f"实际是 {norm['reverse_channel']!r}")
    return norm


def patch(meta_path: Path, norm_path: Path):
    norm = _extract_normalize(json.loads(norm_path.read_text()), norm_path)
    lines = meta_path.read_text().splitlines()
    out = []
    section = None  # 'mean' / 'scale' / None
    list_idx = 0

    for ln in lines:
        stripped = ln.strip()
        indent = len(ln) - len(ln.lstrip())

        # 进入 mean/scale list（只匹配真正的 key，不匹配注释里的 "scale:"）
        if stripped == "mean:":
            section, list_idx = "mean", 0
            out.append(ln)
            continue
        if stripped == "scale:":
            section, list_idx = "scale", 0
            out.append(ln)
            continue

        # 在 list 内：替换 "- value" 行
        if section and stripped.startswith("- ") and list_idx < 3:
            out.append(" " * indent + f"- {norm[section][list_idx]}")
            list_idx += 1
            if list_idx >= 3:
                section = None
            continue

        # 任何非 list-item 行都终止 list
        if section:
            section = None

        # reverse_channel 单独处理（注意忽略带 # 的注释行）
        if stripped.startswith("reverse_channel:") and not stripped.startswith("#"):
            val = "true" if norm.get("reverse_channel", False) else "false"
            out.append(" " * indent + f"reverse_channel: {val}")
            continue

        out.append(ln)

    meta_path.write_text("\n".join(out) + "\n")
    print(f"patched {meta_path.name}: mean={norm['mean']} scale={norm['scale']} "
          f"reverse_channel={norm.get('reverse_channel', False)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write("usage: patch_inputmeta.py <inputmeta.yml> <normalize.json|model.json>\n")
        sys.exit(2)
    try:
        patch(Path(sys.argv[1]), Path(sys.argv[2]))
    except (ValueError, OSError) as exc:
        # 不给 traceback：容器里这行报错要和其它阶段的 FAIL 一样一眼看懂
        sys.stderr.write("patch_inputmeta FAIL: %s\n" % exc)
        sys.exit(1)
