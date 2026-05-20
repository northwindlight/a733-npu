"""把 normalize.json 的 mean/scale/reverse_channel 写进 pegasus 自动生成的 inputmeta.yml。

为什么不直接 checkin 整份 inputmeta：input lid 由 onnx 节点编号决定（如 input_103），
不同模型不同。pegasus generate inputmeta 给出正确 lid，这里只 patch 归一化字段。

normalize.json 字段：
  mean:            [r, g, b] 三通道均值，单位与训练时一致
  scale:           [r, g, b] 三通道缩放，pegasus 计算 (pixel - mean) * scale
  reverse_channel: true=BGR（torchvision/caffe 风格），false=RGB（super-image/手写代码常见）

usage: python3 patch_inputmeta.py <inputmeta.yml> <normalize.json>
"""
import json
import sys
from pathlib import Path


def patch(meta_path: Path, norm_path: Path):
    norm = json.loads(norm_path.read_text())
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
    patch(Path(sys.argv[1]), Path(sys.argv[2]))
