"""L1 校验：在 `docker pull` 之前跑，把能提前发现的错误挡在 4 小时流水线之外。

为什么单独一层
--------------
pegasus 的报错普遍不指名道姓（"Miss input size list"、"import failed"），
一个拼错的键或写反的形状要等镜像拉完、import 跑完才炸。这一层在宿主机上
（有 onnx / onnxruntime / Python 3.11），秒级完成，且刻意把错误信息写成
**可以直接粘贴回 model.json 的形式**。

三层校验的分工：
  L1 本文件       宿主机，昂贵步骤之前，能读 ONNX
  L2 pipeline.sh  容器内，只检查它真正消费的东西（3.8 兼容、不依赖 onnx）
  L3 py_compile   CI 一步，挡住 3.9+ 语法溜进 3.8 容器

usage:
    python3 scripts/check_config.py --model edsr_x2
    python3 scripts/check_config.py --model edsr_x2 --onnx path/to.onnx
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_config  # noqa: E402  （同目录，复用同一套解析，避免两处口径）

# 名字同时进 docker -v 挂载和 shell，所以既防注入也防手滑
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,40}$")

KNOWN_TOP = {
    "schema", "_comment", "license", "onnx", "inputs", "outputs",
    "normalize", "quant", "npu", "calib",
}
KNOWN_NORMALIZE = {"_comment", "mean", "scale", "reverse_channel"}
KNOWN_ONNX = {"_comment", "from", "route", "path", "url", "sha256", "args"}
KNOWN_CALIB = {"_comment", "n", "source", "size", "div2k_start", "seed", "allow_fallback"}
KNOWN_QUANT = {"_comment", "qtype", "quantizer"}

MAX_OPSET = 12          # 11 是推荐值；12 是已知能用的上限
RECOMMENDED_OPSET = 11
NPU_POOL_BYTES = 84 * 1024 * 1024   # EDSR x2 实测的 memory pool 量级，仅作粗估参照

errors = []
warnings = []


def err(msg):
    errors.append(msg)


def warn(msg):
    warnings.append(msg)


def check_unknown_keys(d, known, where):
    for k in d:
        if k not in known:
            hint = ""
            low = k.lower().replace("-", "_")
            for good in known:
                if good.startswith("_") or good == k:
                    continue
                if low == good or low.replace("_", "") == good.replace("_", ""):
                    hint = "（是不是想写 '%s'？）" % good
                    break
            warn("%s: 未知键 '%s'%s —— 会被忽略。拼错的键不会报错，只会静默失效。"
                 % (where, k, hint))


def check_manifest(root, name):
    model_dir = root / "models" / name
    cfg_path = model_dir / "model.json"

    if not NAME_RE.match(name):
        err("模型名 %r 不合法：只允许小写字母/数字/下划线，且以字母或数字开头。\n"
            "  （这个名字会进 docker -v 挂载路径和 shell 命令，所以卡得严）" % name)
        return None, None
    if not model_dir.is_dir():
        err("目录不存在：%s" % model_dir)
        return None, None
    if not cfg_path.exists():
        err("找不到 %s\n"
            "  ★最常见的成因：.gitignore 里 `models/*/*.json` 把 model.json 一起吞了。\n"
            "    确认 .gitignore 有 `!models/*/model.json` 这一行，且文件真的进了 git：\n"
            "      git check-ignore -v models/%s/model.json" % (cfg_path, name))
        return None, None

    try:
        data = model_config.load(cfg_path)
    except model_config.ConfigError as exc:
        err(str(exc))
        return None, None

    check_unknown_keys(data, KNOWN_TOP, "model.json")
    if data.get("schema") != model_config.SCHEMA_SUPPORTED:
        err("schema 必须是 %d，实际 %r" % (model_config.SCHEMA_SUPPORTED, data.get("schema")))
    if not data.get("license"):
        warn("缺少 'license' 字段。模型权重常带许可证（例如 YOLOv5 是 AGPL-3.0），"
             "把它编成 NBG 再挂到公开 Release 有实际后果 —— 请显式声明。")

    norm = data.get("normalize")
    if isinstance(norm, dict):
        check_unknown_keys(norm, KNOWN_NORMALIZE, "normalize")
    onnx = data.get("onnx")
    if isinstance(onnx, dict):
        check_unknown_keys(onnx, KNOWN_ONNX, "onnx")
    calib = data.get("calib")
    if isinstance(calib, dict):
        check_unknown_keys(calib, KNOWN_CALIB, "calib")
        if calib.get("allow_fallback"):
            warn("calib.allow_fallback=true：校准源拉不到时会静默降级成随机噪声，"
                 "产出一个质量更差但看不出区别的 NBG。只在验证流程时开。")
    quant = data.get("quant")
    if isinstance(quant, dict):
        check_unknown_keys(quant, KNOWN_QUANT, "quant")

    # 复用 model_config 的解析器，保证"校验通过的" == "真正会被用的"
    for cmd in ("import-args", "quant-args", "calib-args", "onnx-from"):
        try:
            getattr(model_config, "cmd_" + cmd.replace("-", "_"))(data, cfg_path)
        except model_config.ConfigError as exc:
            err(str(exc))
        except AttributeError:
            pass

    # workflow 与 pipeline.sh 会按这些点号路径取值。路径写错（例如把数组当对象取）
    # 只有在跑到那一步才炸 —— 这里提前全走一遍。
    for key in ("inputs.0.shape", "quant.qtype", "calib.allow_fallback"):
        try:
            model_config.cmd_get(data, cfg_path, key)
        except model_config.ConfigError as exc:
            err(str(exc))
    kind = data.get("onnx", {}).get("from") if isinstance(data.get("onnx"), dict) else None
    for key in ({"script": ["onnx.route"], "url": ["onnx.url", "onnx.sha256"],
                 "file": ["onnx.path"]}.get(kind, [])):
        try:
            model_config.cmd_get(data, cfg_path, key)
        except model_config.ConfigError as exc:
            err(str(exc))

    # normalize 长度必须等于输入通道数（patch_inputmeta 依赖这条不变量）
    try:
        c = model_config._input_shapes(data, cfg_path)[0][1][0]
        if isinstance(norm, dict) and isinstance(norm.get("mean"), list):
            if len(norm["mean"]) != c:
                err("normalize.mean 有 %d 项，但输入通道数是 %d。\n"
                    "  两者必须相等 —— patch_inputmeta 按通道逐个替换。" % (len(norm["mean"]), c))
            if len(norm.get("scale", [])) != c:
                err("normalize.scale 有 %d 项，但输入通道数是 %d。"
                    % (len(norm.get("scale", [])), c))
    except model_config.ConfigError:
        pass   # 上面已经报过

    npu = data.get("npu")
    if npu is not None and npu not in ("v1", "v2", "v3"):
        err("npu 必须是 v1/v2/v3 之一，实际 %r" % npu)

    return data, cfg_path


def resolve_onnx_source(root, name, data, cfg_path, override):
    """决定 ONNX 从哪来。返回 (kind, 描述)。真正的物化在 workflow 里做。"""
    if override:
        p = Path(override)
        if not p.exists():
            err("--onnx 指定的文件不存在：%s" % p)
        return "override", str(p)

    onnx = data["onnx"]
    kind = onnx.get("from")
    if kind == "script":
        route = onnx.get("route")
        if not route:
            err("onnx.from=script 但缺少 'route'（生成脚本的相对路径，如 \"export.py\"）")
        elif not (root / "models" / name / route).exists():
            err("onnx.route 指向的脚本不存在：%s" % (root / "models" / name / route))
        return kind, route
    if kind == "file":
        p = onnx.get("path")
        if not p:
            err("onnx.from=file 但缺少 'path'")
        elif not (root / "models" / name / p).exists():
            err("onnx.path 指向的文件不存在：%s" % (root / "models" / name / p))
        return kind, p
    if kind == "url":
        if not onnx.get("url"):
            err("onnx.from=url 但缺少 'url'")
        if not onnx.get("sha256"):
            err("onnx.from=url 必须提供 'sha256'。\n"
                "  没有哈希的远程模型不可复现：上游换了文件你也看不出来。")
        elif not re.match(r"^[0-9a-f]{64}$", str(onnx.get("sha256", ""))):
            err("onnx.sha256 必须是 64 位小写十六进制")
        return kind, onnx.get("url")
    err("onnx.from 必须是 script / file / url 之一，实际 %r" % kind)
    return None, None


def check_onnx(onnx_path, data, cfg_path):
    """ONNX 侧检查。用 onnxruntime 而不是 graph.input 取 IO 名 —— IR<4 的模型
    会把 initializer 也列进 graph.input（resnet50-sim.onnx 就有 96 个"输入"）。"""
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        warn("装不上 onnx/onnxruntime（%s），跳过 ONNX 侧检查" % exc)
        return

    try:
        m = onnx.load(str(onnx_path), load_external_data=False)
        onnx.checker.check_model(str(onnx_path))
    except Exception as exc:
        err("ONNX 自身校验不通过：%s" % exc)
        return

    opsets = [(o.domain or "ai.onnx", o.version) for o in m.opset_import]
    ver = next((v for d, v in opsets if d == "ai.onnx"), None)
    if ver is None:
        warn("ONNX 没有声明默认 domain 的 opset")
    elif ver > MAX_OPSET:
        err("opset %d 高于已知可用上限 %d（推荐 %d）。"
            "DepthToSpace 的 mode 语义在更高 opset 有差异，pegasus 可能编出通道错乱的模型。"
            % (ver, MAX_OPSET, RECOMMENDED_OPSET))
    elif ver > RECOMMENDED_OPSET:
        warn("opset %d（推荐 %d，仓库已验证的档位）" % (ver, RECOMMENDED_OPSET))

    try:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        err("onnxruntime 无法加载这个模型：%s" % exc)
        return

    actual_in = [i.name for i in sess.get_inputs()]
    actual_out = [o.name for o in sess.get_outputs()]

    declared_in = [i["name"] for i in data["inputs"]]
    declared_out = list(data["outputs"])

    def paste_snippet(names, key):
        body = ",\n".join('      {"name": "%s", "shape": [1, 3, 224, 224]}' % n for n in names) \
            if key == "inputs" else ",\n".join('    "%s"' % n for n in names)
        return '    "%s": [\n%s\n    ]' % (key, body)

    if declared_in != actual_in:
        err("model.json 声明的输入名与 ONNX 实际不符。\n"
            "  model.json: %r\n  ONNX 实际 : %r\n"
            "  把 model.json 的这段改成（形状自己核对）：\n%s"
            % (declared_in, actual_in, paste_snippet(actual_in, "inputs")))
    if declared_out != actual_out:
        err("model.json 声明的输出名与 ONNX 实际不符。\n"
            "  model.json: %r\n  ONNX 实际 : %r\n"
            "  把 model.json 的这段改成：\n%s"
            % (declared_out, actual_out, paste_snippet(actual_out, "outputs")))

    for i in sess.get_inputs():
        if i.type != "tensor(float)":
            err("输入 '%s' 的类型是 %s，需要 tensor(float)。"
                "fp16 或已量化的 ONNX 会在 pegasus 深处报一个看不懂的错。" % (i.name, i.type))

    if declared_in == actual_in and declared_out == actual_out:
        import numpy as np
        feed = {}
        for spec in data["inputs"]:
            feed[spec["name"]] = np.zeros(spec["shape"], dtype=np.float32)
        try:
            outs = sess.run(None, feed)
        except Exception as exc:
            err("ORT 空跑失败：%s" % exc)
        else:
            shapes = [list(o.shape) for o in outs]
            print("  ORT 空跑 OK：%s -> %s" % (
                {k: v.shape for k, v in feed.items()}, shapes))
            # 粗估：最大一张输出张量按 fp32 算，和 EDSR 量级的内存池比一比
            biggest = max((int(np.prod(s)) for s in shapes if all(v > 0 for v in s)),
                          default=0)
            if biggest * 4 > NPU_POOL_BYTES:
                warn("最大输出张量约 %.1f MB (fp32)，超过参照内存池 %d MB 的一半。"
                     "NBG 的 memory pool 要装下所有中间激活 —— 编译大概率会在设备上 OOM。"
                     "考虑减小输入尺寸。" % (biggest * 4 / 1e6, NPU_POOL_BYTES // 1024 // 1024))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", default=None, help="仓库根（默认脚本的上一级）")
    ap.add_argument("--onnx", default=None, help="直接指定 ONNX（CI 里 ONNX 是 artifact 下来的时候用）")
    ap.add_argument("--skip-onnx", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent

    print("== check_config: %s ==" % args.model)
    data, cfg_path = check_manifest(root, args.model)

    if data is not None:
        kind, desc = resolve_onnx_source(root, args.model, data, cfg_path, args.onnx)
        print("  ONNX 来源：from=%s  %s" % (kind, desc))
        if not args.skip_onnx and kind in ("override", "file"):
            p = Path(args.onnx) if args.onnx else (root / "models" / args.model / desc)
            if p.exists():
                check_onnx(p, data, cfg_path)

    for w in warnings:
        print("  WARN  %s" % w)
    for e in errors:
        print("  ERROR %s" % e)
    if errors:
        print("== FAIL：%d 个错误 ==" % len(errors))
        return 1
    print("== PASS（%d 个警告）==" % len(warnings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
