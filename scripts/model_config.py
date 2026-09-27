"""从 models/<name>/model.json 构造 pegasus 命令行参数。

设计要点
--------
1. **不用 eval、不拼字符串**。每个子命令把 argv **每行一个 token** 打到 stdout，
   pipeline.sh 用 `mapfile -t ARGS < <(...)` 收进数组再展开。
   这样带空格/引号的路径不会成为注入面，argv 构造只有这一处实现。

2. **Python 3.8 安全**。本文件在容器内（Python 3.8）执行，
   不用 `X | Y` 注解、不用 `dict | dict`、不用 `tomllib`。
   CI 里有 `python3.8 -m py_compile scripts/*.py` 兜底。

3. **只做搬运，不做校验**。深度校验在 scripts/check_config.py（在宿主机上跑，
   能拿到 onnx/onnxruntime）。这里只保证"读得到、形状对"，
   失败的报错也必须指名道姓说清楚哪个文件哪个键。

usage:
    python3 model_config.py <model.json> import-args
    python3 model_config.py <model.json> quant-args
    python3 model_config.py <model.json> calib-args
    python3 model_config.py <model.json> get <dotted.key>
    python3 model_config.py <model.json> keys
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCHEMA_SUPPORTED = 1

REQUIRED_TOP = ("schema", "onnx", "inputs", "outputs", "normalize", "quant", "calib")
ONNX_FROM = ("script", "file", "url")
QTYPES = ("uint8", "int16")


class ConfigError(Exception):
    pass


def load(path):
    # type: (Path) -> dict
    if not path.exists():
        raise ConfigError("找不到配置文件：%s" % path)
    try:
        data = json.loads(path.read_text())
    except ValueError as exc:            # json.JSONDecodeError 是 ValueError 子类
        raise ConfigError("%s 不是合法 JSON：%s" % (path, exc))
    if not isinstance(data, dict):
        raise ConfigError("%s 顶层必须是对象" % path)
    return data


def _require(data, key, path):
    # type: (dict, str, Path) -> object
    if key not in data:
        raise ConfigError("%s: 缺少必需键 '%s'" % (path, key))
    return data[key]


def _input_shapes(data, path):
    # type: (dict, Path) -> list
    """返回 [(name, [c,h,w]), ...]。shape 是模型里的 NCHW，这里只要 CHW。"""
    inputs = _require(data, "inputs", path)
    if not isinstance(inputs, list) or not inputs:
        raise ConfigError("%s: 'inputs' 必须是非空数组" % path)
    out = []
    for i, item in enumerate(inputs):
        if not isinstance(item, dict):
            raise ConfigError("%s: inputs[%d] 必须是对象" % (path, i))
        name = item.get("name")
        shape = item.get("shape")
        if not name or not isinstance(name, str):
            raise ConfigError("%s: inputs[%d].name 必须是非空字符串" % (path, i))
        if not isinstance(shape, list) or len(shape) != 4 or not all(
                isinstance(v, int) and v > 0 for v in shape):
            raise ConfigError(
                "%s: inputs[%d].shape 必须是 4 个正整数 [N,C,H,W]，实际 %r"
                % (path, i, shape))
        if shape[0] != 1:
            raise ConfigError(
                "%s: inputs[%d].shape[0] (batch) 必须是 1，实际 %d —— "
                "NBG 不支持动态 batch" % (path, i, shape[0]))
        out.append((name, shape[1:]))
    return out


# ---------------------------------------------------------------- 子命令


def cmd_import_args(data, path):
    """pegasus import onnx 的输入/输出/尺寸参数。

    ★ `--inputs` 与 `--input-size-list` 是按位置配对的：第 i 个尺寸对应第 i 个输入。
      多输入模型的正确写法**尚未用真机验证过**（仓库目前只有一个单输入模型）。
      加多输入模型时，先用一个两输入的小网络确认 pegasus 收的是
      "一个 flag 后跟多个 CHW"还是"每个输入重复一次 flag"。
    """
    inputs = _input_shapes(data, path)
    outputs = _require(data, "outputs", path)
    if not isinstance(outputs, list) or not outputs or not all(
            isinstance(o, str) and o for o in outputs):
        raise ConfigError("%s: 'outputs' 必须是非空字符串数组" % path)

    toks = ["--inputs"] + [n for n, _ in inputs]
    toks += ["--outputs"] + list(outputs)
    toks += ["--input-size-list"] + [",".join(str(v) for v in chw) for _, chw in inputs]
    return toks


def cmd_quant_args(data, path):
    quant = _require(data, "quant", path)
    if not isinstance(quant, dict):
        raise ConfigError("%s: 'quant' 必须是对象" % path)
    qtype = quant.get("qtype")
    quantizer = quant.get("quantizer")
    if qtype not in QTYPES:
        raise ConfigError("%s: quant.qtype 必须是 %s 之一，实际 %r"
                          % (path, " / ".join(QTYPES), qtype))
    if not quantizer or not isinstance(quantizer, str):
        raise ConfigError("%s: quant.quantizer 必须是非空字符串" % path)
    return ["--quantizer", quantizer, "--qtype", qtype]


def cmd_calib_args(data, path):
    """prepare_calib.py 的参数。calib.size 是 [H, W]（不是 CHW！）。"""
    calib = _require(data, "calib", path)
    if not isinstance(calib, dict):
        raise ConfigError("%s: 'calib' 必须是对象" % path)
    inputs = _input_shapes(data, path)
    want_h, want_w = inputs[0][1][1], inputs[0][1][2]

    n = calib.get("n")
    if not isinstance(n, int) or n < 1:
        raise ConfigError("%s: calib.n 必须是 >=1 的整数，实际 %r" % (path, n))

    size = calib.get("size")
    if size is None:
        h, w = want_h, want_w
    else:
        if not isinstance(size, list) or len(size) != 2 or not all(
                isinstance(v, int) and v > 0 for v in size):
            raise ConfigError("%s: calib.size 必须是 [H, W] 两个正整数，实际 %r"
                              % (path, size))
        h, w = size
        if (h, w) != (want_h, want_w):
            raise ConfigError(
                "%s: calib.size=%r 与 inputs[0].shape 的 H,W=(%d,%d) 不一致。\n"
                "  校准图的尺寸必须等于 NPU 的实际输入尺寸，否则量化标定对不上。"
                % (path, size, want_h, want_w))

    source = calib.get("source")
    if source == "div2k":
        src_tok = "div2k"
    elif source == "random":
        src_tok = "random"
    elif isinstance(source, str) and source.startswith("dir:"):
        src_tok = source
    else:
        raise ConfigError(
            "%s: calib.source 必须是 \"div2k\" / \"random\" / \"dir:<相对路径>\"，实际 %r"
            % (path, source))

    toks = ["--n", str(n), "--h", str(h), "--w", str(w), "--source", src_tok]
    if source == "div2k":
        start = calib.get("div2k_start", 801)
        if not isinstance(start, int) or start < 1:
            raise ConfigError("%s: calib.div2k_start 必须是正整数" % path)
        toks += ["--div2k-start", str(start)]
    seed = calib.get("seed", 0)
    if not isinstance(seed, int):
        raise ConfigError("%s: calib.seed 必须是整数" % path)
    return toks + ["--seed", str(seed)]


def cmd_get(data, path, key):
    """点号路径取值，例如 quant.qtype / onnx.route / inputs.0.shape。

    数组用数字下标（inputs.0.shape）—— 只支持 dict 键会让 inputs.0.shape 直接失败。
    """
    node = data
    for part in key.split("."):
        if isinstance(node, dict):
            if part not in node:
                raise ConfigError("%s: 没有键 '%s'（在 '%s' 处走不下去）" % (path, key, part))
            node = node[part]
        elif isinstance(node, list):
            try:
                idx = int(part)
            except ValueError:
                raise ConfigError("%s: '%s' 的 '%s' 需要数组下标，实际不是整数"
                                  % (path, key, part))
            if not 0 <= idx < len(node):
                raise ConfigError("%s: '%s' 的下标 %d 越界（数组长度 %d）"
                                  % (path, key, idx, len(node)))
            node = node[idx]
        else:
            raise ConfigError("%s: '%s' 走到 %r 就下不去了（既不是对象也不是数组）"
                              % (path, key, node))
    if isinstance(node, (dict, list)):
        return [json.dumps(node, ensure_ascii=False)]
    if node is None:
        return [""]
    if isinstance(node, bool):
        # 用 Python 的 True/False 而不是 JSON 的 true/false：
        # workflow 里用 `[ "$X" = "True" ]` 判断，口径保持一致
        return [str(node)]
    return [str(node)]


def cmd_check_onnx_ref(data, path):
    """输出 ONNX 来源的种类（script/file/url），供 pipeline.sh 分流。"""
    onnx = _require(data, "onnx", path)
    if not isinstance(onnx, dict):
        raise ConfigError("%s: 'onnx' 必须是对象" % path)
    kind = onnx.get("from")
    if kind not in ONNX_FROM:
        raise ConfigError("%s: onnx.from 必须是 %s 之一，实际 %r"
                          % (path, " / ".join(ONNX_FROM), kind))
    return [kind]


def main(argv):
    if len(argv) < 3:
        sys.stderr.write(__doc__)
        return 2
    path = Path(argv[1])
    cmd = argv[2]
    try:
        data = load(path)
        if cmd == "import-args":
            toks = cmd_import_args(data, path)
        elif cmd == "quant-args":
            toks = cmd_quant_args(data, path)
        elif cmd == "calib-args":
            toks = cmd_calib_args(data, path)
        elif cmd == "onnx-from":
            toks = cmd_check_onnx_ref(data, path)
        elif cmd == "get":
            toks = cmd_get(data, path, argv[3])
        elif cmd == "keys":
            toks = sorted(data.keys())
        else:
            raise ConfigError("未知子命令 %r" % cmd)
    except ConfigError as exc:
        sys.stderr.write("model_config: %s\n" % exc)
        return 1

    for tok in toks:
        # token 里出现换行/回车会让 mapfile 切错行 —— 直接拒绝而不是悄悄产出坏 argv
        if "\n" in tok or "\r" in tok:
            sys.stderr.write("model_config: 参数里含换行，拒绝输出：%r\n" % tok)
            return 1
        print(tok)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
