# a733-npu

把**任意 ONNX** 编译成全志 A733 NPU（Vivante VIP9000）能跑的 NBG。

```
ONNX (opset 11) → pegasus/acuity → 量化 → NBG 单 .nb 文件 → A733 真机
```

**加一个模型 = 加一个目录。** 输入输出名、形状、归一化、量化精度、校准源全部写在
`models/<name>/model.json` 里，流水线脚本不含任何模型专属常量。

---

## 实测性能

**测量条件（缺一不可，否则数字不可比）**：NPU governor=performance @1008 MHz，
**DDR governor=performance @1800 MHz**，`vpm_run -b 1 -l N` 取 N 次均值，纯推理不含前后处理。

| 模型 | 输入 | 输出 | 纯推理 | cycle | MAC/cycle | 备注 |
|------|------|------|--------|-------|-----------|------|
| LeNet | 28×28×1 | 10 类 | 0.097 ms | 63,724 | — | 基本都是固定开销，不具参考性 |
| ShuffleNetV2 | 224×224×3 | 1000 类 | 2.9 ms | — | 早期测量 |
| ResNet50 | 224×224×3 | 1000 类 | **7.24 ms** | 7,279,439 | 561 | |
| YOLOv5s | 640×640×3 | 检测 | **24.8 ms** | 24,764,533 | 332 | depthwise + stride 吃利用率 |
| **EDSR x2** | 640×360×3 | 1280×720×3 | **233.8 ms** | 235,528,418 | **1343** | memory pool 88,474,624 B |

### 这张表怎么读

**峰值 ≥ 2.71 TOPS。** EDSR 是均匀的 3×3 堆叠、无 stride、无 depthwise，是最接近满负荷的
测试模型：1343 MAC/cycle = 1008 MHz 下 1353 GMAC/s = **2.71 TOPS**。
这是**下界** —— 峰值只会更高（等价 1.2 GHz 下 ≥3.2 TOPS）。

**★利用率与模型结构强相关，别拿 CNN 分类模型当尺子标峰值。**
同一块硅上 MAC/cycle 从 332 到 1343 差了 4 倍，差距全在模型结构（YOLOv5s 的
depthwise/stride/focus/concat）。想标定峰值要用 EDSR 这种「全分辨率均匀卷积堆叠」。

**★★NPU 吞吐强烈依赖 DDR 频率 —— 测之前必须固定它。**
同一个 EDSR NBG，只改 DDR 频率：

| DDR | EDSR | ResNet50 | YOLOv5s |
|-----|------|----------|---------|
| **1800 MHz** | **233.8 ms** | 7.24 ms | 24.8 ms |
| 1200 MHz | 246.2 ms (+5%) | 7.62 ms (+5%) | 27.9 ms (+13%) |
| 800 MHz | 328.6 ms (**+41%**) | 9.19 ms (+27%) | 37.6 ms (+52%) |

DDR governor 是 `performance` 时**不会自动跑满** —— 它把 min/max 都设成 `max_freq`，
而 `max_freq` 可能是被上一次实验改小的值。改动前先读
`/sys/class/devfreq/a020000.dmcfreq/{cur_freq,max_freq}`。

> **一次踩坑实录**：同一份 EDSR NBG 先测出 256.5 ms、后测出 233.8 ms，差 9%，
> 一度以为是内存池物理落点的随机性。排除了「每进程重新分配」（8 个独立进程
> 波动仅 0.08%）之后，才定位到是 DDR 频率不同。上面那张 DDR 表就是结论。
>
> 顺带撤回一个更早的说法：曾用「降 NPU 时钟」的数据拟合出
> `耗时 = 固定开销 1.62 ms + 算力周期/时钟`。**降 NPU 时钟同时会让 DDR 相对更宽裕、
> 停顿减少**，两种解释是混淆的，这个 1.62 ms 站不住，不要引用。

> 早期版 README 写过「360×640 → 720×1280 纯 NPU 延迟 236ms」。这个数字站得住
> —— 现在固定 DDR 后测得 233.8 ms。但当时它是由**随机噪声校准**的 NBG 产出的
> （见踩坑 17），数值巧合而已。

### 历史对照（未复测，仅供参考）

本仓库的起点是作者原有的 **Win Server 核显超分工具**（同一个 `realesr-animevideov3` x4，
走 ncnn-Vulkan fp16）。它在 1280×536 上的历史记录是 **~2.9 s/帧**（2026-09 测量）。
**该机器目前已下线，无法复测** —— 这个数字只作为历史参照，不是本轮对照实验的结果。
等 A733 上跑通 animevideov3 之后，才能给出同等条件下的比较。

---

## 仓库结构

```
├── container.txt                      ★ 转换容器的 digest 引用（唯一不可恢复依赖）
├── requirements.txt                   宿主机依赖（全部 pin 住）
├── .github/workflows/build_nbg.yml    validate（便宜）+ convert（贵）两个 job
├── scripts/
│   ├── env.sh                         v1/v2/v3 → VSIMULATOR_CONFIG，NPU 型号映射
│   ├── model_config.py                model.json → pegasus argv（每行一个 token，不用 eval）
│   ├── check_config.py                L1 校验：docker pull 之前把错误全挡下来
│   ├── prepare_calib.py               校准图 + dataset.txt（div2k / random / dir:）
│   ├── patch_inputmeta.py             归一化参数写进 pegasus 的 inputmeta.yml
│   ├── pipeline.sh                    主流水线，支持 --stop-after
│   └── make_build_info.py             产出 build_info.json（板端 dequant 参数从这里取）
└── models/
    ├── edsr_x2/                       super-image EDSR x2
    │   ├── model.json
    │   └── export.py
    └── animevideov3/                  Real-ESRGAN realesr-animevideov3 x4
        ├── model.json
        └── export.py
```

---

## 快速开始

### 走 CI（推荐）

Actions → **Build NBG** → Run workflow：

| 输入 | 说明 |
|------|------|
| `model` | `models/` 下的目录名，如 `edsr_x2` |
| `npu` | `v3` = A733（覆盖 model.json 里的 npu） |
| `upload_release` | 把 NBG 挂到滚动 Release |

产出的下载地址**长期固定**（同名附件每次重建覆盖）：

```
https://github.com/northwindlight/a733-npu/releases/download/nbg-<model>-<npu>/network_binary.nb
```

> CI artifact 只保留 90 天，Release 才是长期地址。第一次建仓库时没有 Release，
> 过期的 artifact 就再也找不回来了。

### 本地（需要 x86_64 Linux + docker）

```bash
NAME=edsr_x2
IMG=$(grep -vE '^\s*#|^\s*$' container.txt | head -1)

# 1. 生成 ONNX（若 model.json 的 onnx.from=script）
python3 models/$NAME/export.py --shape 3,360,640 --output models/$NAME/$NAME.onnx

# 2. 校准图（CWD 必须是模型目录，见下方「路径约定」）
docker run --rm -v "$PWD:/ws" -w "/ws/models/$NAME" "$IMG" \
    python3 /ws/scripts/prepare_calib.py --n 8 --h 360 --w 640 --source div2k

# 3. 编译
docker run --rm -v "$PWD:/workspace" -w /workspace "$IMG" \
    bash scripts/pipeline.sh $NAME v3

# 产物
ls models/$NAME/wksp/*/network_binary.nb
cat models/$NAME/build_info.json
```

---

## 加一个模型

1. **建目录**：`models/<name>/`（名字只允许小写字母/数字/下划线）
2. **写 `model.json`**：照着 `models/edsr_x2/model.json` 改
3. **提供 ONNX**，三种来源选一：
   - `"onnx": {"from": "script", "route": "export.py"}` —— 本目录放一个导出脚本，
     接口固定为 `--shape CHW --output PATH`
   - `"onnx": {"from": "url", "url": "...", "sha256": "..."}` —— sha256 必填
   - `"onnx": {"from": "file", "path": "xxx.onnx"}` —— 文件已在本目录（CI 里需要已提交）

   无论哪种，最终都会物化成 `models/<name>/<name>.onnx`。
4. **本地先验一遍**：`python3 scripts/check_config.py --model <name>`
5. **dispatch** workflow，`model` 填 `<name>`

> ### ★ 加模型第 1 号坑：`.gitignore`
> `models/*/*.json` 是为了挡住 pegasus 生成的 acuity IR。新建的 `model.json` 会被
> **静默吞掉** —— 本地 `git add` 看着成功，CI 上才发现文件不存在。
> `.gitignore` 里必须留着 `!models/*/model.json` 这条放行规则。
> 自查（别用 `git check-ignore -v`，它连 negation 规则也会打印出来，容易看反）：
> ```bash
> git add --dry-run models/<name>/model.json   # 有输出 = 会被加入
> ```

---

## `model.json` 字段

```jsonc
{
  "schema": 1,
  "license": "Apache-2.0",          // 模型权重的许可证。编成 NBG 挂公开 Release 前想清楚
  "onnx":    { "from": "script", "route": "export.py" },
  "inputs":  [ { "name": "input", "shape": [1, 3, 360, 640] } ],   // ★数组，支持多输入
  "outputs": [ "output" ],                                          // ★数组，支持多输出
  "normalize": { "mean": [0,0,0], "scale": [1,1,1], "reverse_channel": false },
  "quant":   { "qtype": "uint8", "quantizer": "asymmetric_affine" },
  "npu":     "v3",
  "calib":   { "n": 8, "source": "div2k", "div2k_start": 801,
               "size": [360, 640], "seed": 0, "allow_fallback": false }
}
```

- **`inputs` 必须是数组**，即使只有一个。`shape` 是 NCHW 且 batch 必须为 1（NBG 不支持动态 batch）。
- **`outputs` 必须是数组**。YOLO 那类多输出模型直接列出所有名字。
- **`normalize.mean` / `scale` 的长度必须等于输入通道数**，`reverse_channel` **没有默认值**
  —— 拼错成 `revers_channel` 会让它静默按 RGB 编译出通道颠倒的模型，所以缺了就硬失败。
- **`calib.size` 是 `[H, W]`**（不是 CHW！），且必须等于输入形状的 H、W。
- **`calib.source`**：`"div2k"` / `"random"` / `"dir:<相对路径>"`。
  `allow_fallback: true` 才会在 DIV2K 拉不到时降级成随机噪声（**产出的 NBG 质量不可信，
  而且从产物上看不出来**）—— 只在验证流程时开。
- **`quant.qtype`**：`uint8` 或 `int16`。这片的 NN 引擎 INT8 和 INT16 都是 8 核
  （见 `~/a733-work/bsp/drivers/npu/` 的 feature database）。

### 归一化预设（贴进 `normalize` 段）

| 模型 | mean | scale | reverse_channel |
|------|------|-------|-----------------|
| super-image / EDSR | `[0,0,0]` | `[1,1,1]` | `false` |
| Real-ESRGAN | `[0,0,0]` | `[0.00392,0.00392,0.00392]` | `false` |
| YOLO v3/v5 | `[0,0,0]` | `[0.0039,0.0039,0.0039]` | `false` |
| torchvision ImageNet | `[123.675,116.28,103.53]` | `[0.0171,0.0175,0.0174]` | `false` |
| caffe / OpenCV (BGR) | 同上 | 同上 | `true` |

---

## 板端

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/viplite-tina/lib/aarch64-none-linux-gnu/v2.0

# 快速验证某个 NBG
vpm_run -s sample.txt -l 1 -b 1
```

板端做 dequant 需要的 `scale` / `zero_point`，从 **`build_info.json`** 取，不要从 README 抄：

```c
awnn_init();
Awnn_Context_t *ctx = awnn_create(nbg_path);   // 一次 load
for (each frame) {
    awnn_set_input_buffers(ctx, input_buffers);
    awnn_run(ctx);
    void *raw_out = awnn_get_output_buffer(ctx, 0);   // uint8 量化值
}
awnn_destroy(ctx);
awnn_uninit();
```

> `build_info.json` 里的 `output_quant` 目前需要**人工回填** —— 从 pegasus 的
> `.quantize` 产物里自动提取还没验证过。空值表示还没填，板端遇到 `null` 应当拒绝执行，
> 不要默认成 `1.0 / 0`。

---

## 验证

### 便宜的回归（分钟级）

改了流水线之后不要每轮都跑完整编译。用 `--stop-after` 停在中间阶段：

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace "$IMG" \
    bash scripts/pipeline.sh edsr_x2 v3 --stop-after inputmeta
diff <(git show HEAD:models/edsr_x2/edsr_x2_inputmeta.yml 2>/dev/null || echo) \
     models/edsr_x2/edsr_x2_inputmeta.yml
```

`patch_inputmeta.py` 同时接受旧的扁平 `normalize.json` 和新的嵌套 `model.json`，
所以两条路径能在同一棵树里跑、生成物**逐字节对比**。

### 数值比对闸（★新增模型必做）

`DepthToSpace` 的坑值得单独说：PyTorch 的 `pixel_shuffle` ≡ ONNX
`DepthToSpace(mode="CRD")`，而 **ONNX 默认是 `"DCR"`**。任何忽略这个属性的后端会给出
**形状正确、通道错乱**的输出 —— 表现成网格/拼贴伪影，而且**不报错**。

所以换后端/换模型后，必须拿同一个输入比一次 NBG 输出和 ONNX Runtime 参考输出，
算 PSNR 或最大偏差。**这道闸要故意破坏一次确认它会响**（把 mode 改成 DCR 应该能测出来）。

### 构建记录

每次编译都会写 `models/<name>/build_info.json`：ONNX 与 NBG 的 sha256、镜像 digest、
git sha、NPU 目标字符串、量化参数、校准来源（含**是否降级成随机噪声**）。
比对两次构建的 `build_info.json` 就能知道产物变没变。

---

## 踩坑记录

全流程中**非显然的陷阱**。前 10 条来自最初的 EDSR 摸索（每条都浪费了 4h+ 的 CI 跑），
11 之后是 2026-09-27 改造时补的。

### pegasus / acuity 侧

**1. ONNX 有 dynamic_axes 时必须显式传 `--input-size-list`**
acuity 不会自己解析尺寸，报 `Miss input size list`。必须传 `--input-size-list 3,360,640`（CHW）。

**2. inputmeta 的 `lid` 是 import 后才确定的**
不要 checkin 固定 lid 的 inputmeta.yml。`pegasus generate inputmeta` 自动拿正确的 lid
（如 `input_103`），然后用 `normalize` 段覆盖归一化字段即可。

**3. LD_LIBRARY_PATH 顺序：torch/lib 必须在 vsimulator/lib 前面**
vsimulator/lib 里有自己的 `libc10.so` / `libtorch_cpu.so`，会跟 Python torch 冲突。
顺序反了在 import 阶段就崩。

**4. `--pack-nbg-unify` 会自动给 output-path 追加 `_nbg_unify` 后缀**
产物在 `wksp/<name>_<qtype>_nbg_unify/network_binary.nb`。

### 板端

**5. vpm_run 输入用 `.dat` 后缀**（纯 binary uint8 NCHW，无 header）
`.tensor` 是 ASCII 文本格式（每行一个数），vpm_run 据此判定格式。`.raw` → segfault。

**6. vpm_run 输出必写 56MB ASCII 文本**
`--save_txt 1` 把 2.76M 个 float 写成 ASCII 字符串，单帧额外 ~10s。
循环推理用 `-b 1` 跳过输出保存，或 C 直调 VIP API。

**7. 推理输出是 uint8 量化值，显示前要 dequant**
`pixel = (uint8_value - zero_point) * scale`。不做 dequant 画面发白或发暗。
参数从 `build_info.json` 取。

**8. viplite-tina API 使用模式**：`awnn_create` 一次 load，循环里只换输入、跑、拿 raw buffer。

**9. LD_LIBRARY_PATH 要包含 viplite-tina 路径**。

### Git / 仓库侧

**10. 大二进制不要进 git**
2.7GB 的 docker 镜像用 ghcr 分发；`.gitignore` 挡住 `*.zip/*.tar/*.nb/*.onnx/*.data/*.quantize`。

### 改造时补的

**11. `find ... | head -1` 在 `set -euo pipefail` 下会静默中止脚本**
`find` 多输出一行就被 SIGPIPE 杀掉（退出码 141），于是脚本在 `if [ -z "$NB" ]` 守卫
**之前**就退出 —— 没有诊断、看起来像"没找到文件"。改用 `mapfile` 收数组再显式判定。

**12. 容器是 Python 3.8，`socket.timeout` 还不是 `TimeoutError` 的别名**（3.10 才合并）
旧代码 `except (urllib.error.URLError, TimeoutError)` 读超时时异常直接逃逸，
`--fallback-random` 那条分支**永远走不到**。用 `except OSError`（覆盖 URLError、
ConnectionError、3.8 的 socket.timeout）。且**只包住 fetch 段** —— 把 `crop()` 包在
try 里会让"图下载残缺导致解码失败"被静默降级成随机噪声，那是把硬失败换成更坏的产物。

**13. 形状不要声明在两个地方**
旧 workflow 有个 `resolution` 输入：ONNX 按它导出，但调 `pipeline.sh` 时没传下去，
于是走默认值。dispatch 成别的分辨率就得到一个形状对不上的 NBG。
现在形状只在 `model.json` 里声明一次。

**14. `models/*/*.json` 会吞掉新建的 `model.json`**（见上文「加模型第 1 号坑」）。

**15. `DepthToSpace` 的 `mode` 必须是 `CRD`**（见上文「数值比对闸」）。

**16. GitHub 托管 runner 的作业硬上限是 6 小时**
写 `timeout-minutes: 720` 不会延长。一个 job 只编一个模型。

**17. ★DIV2K 的逐文件下载路径早就失效了，而 `--fallback-random` 把失败吞了**
原脚本按 `validation_release/DIV2K_valid_HR/0801.png` 逐张下载，实测 **404** ——
上游现在**只发 zip**。而 workflow 一直传着 `--fallback-random`，于是 `URLError`
被当成"网络抖了一下"，静默降级成随机噪声。
**后果：这之前的每一个 NBG 都是用噪声校准的，从产物上完全看不出来。**
（也解释了为什么 README 早期记的 `output scale=2.646` 和后来复现的 `3.105` 对不上
—— 两次都是噪声，只是构建不同。）

现在改成拉 `DIV2K_valid_LR_bicubic_X4.zip`（30 MB，缓存在 `_raw/`）只解出需要的几张，
并且 `calib.allow_fallback` 默认 **false** —— 拉不到就硬失败。
校准出来的图是真照片还是噪声，可以用相邻像素差一眼分辨（真照片 4~10，噪声 ~85）。

> 顺带一提：DIV2K 的许可是**仅限非商业研究**，所以不要把它的裁切图提交进仓库 ——
> 运行时下载、用完即弃是可以的，重分发不行。

**18. ★板端测性能前必须固定 DDR 频率，`performance` 不代表跑满**
`/sys/class/devfreq/a020000.dmcfreq` 的 `performance` governor 是把 min/max 都设成
**当前的 `max_freq`**。如果之前有实验把 `max_freq` 调小过，切回 `performance`
会把它**钉在那个小值上**，而 `cur_freq` 会如实反映 —— 但你不去读就不知道。
实测 DDR 800 MHz vs 1800 MHz，EDSR 慢 41%、YOLOv5s 慢 52%。详见上文「这张表怎么读」。

---

## 已知未解

- 驱动 feature database 里 `vip_sram_size = 0x80000` = **512 KB**，
  但早期 README 记的是 **296 KB** —— 两者对不上，296 的来源没查到。
- 峰值是从 EDSR 的实测**下界**反推的（≥2.47 TOPS），不是直接读到的硬件常量。
  驱动 feature database 里的 `NNMadPerCore(64) × NNCoreCount(8)` 无法解释实测吞吐
  （差 2.4 倍），那个字段的单位还没搞清。**实测数字不受影响。**

## 许可

Apache-2.0（见 `LICENSE`）。`models/animevideov3/export.py` 内联的 `SRVGGNetCompact`
来自 [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)（Apache-2.0）。
各模型权重的许可证见对应 `model.json` 的 `license` 字段 —— **YOLOv5 是 AGPL-3.0**，
把它编成 NBG 挂到公开 Release 前请先确认。
