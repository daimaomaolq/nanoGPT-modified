# nanoGPT Modern Components + KV Cache 实验报告

## 1. 项目目标

本项目基于 Karpathy 的 nanoGPT 做两阶段改造：

1. `ModernComponents0.1`：在原始 GPT-2 风格 Block 中加入现代大模型常见组件，包括 RMSNorm、SwiGLU、RoPE。
2. `ModernKVCache0.2`：在现代组件版本基础上加入 KV Cache 推理路径，并提供 generation benchmark 工具。

目标不是完整复现 GPT-2，也不是追求在小数据集上单点刷分，而是验证以下问题：

- 现代 Transformer 组件能否在 nanoGPT 中以可配置方式接入，并保持训练闭环正常。
- 在更接近真实预训练分布的 OpenWebText 子集上，现代组件是否能改善短程语言建模 loss。
- KV Cache 是否能在同一个 checkpoint 上带来可测量的自回归推理吞吐提升。
- 整个项目是否能形成可复现实验、可对比日志、可解释 trade-off 的工程闭环。

## 2. 版本与 Git 标签

| 节点 | Commit/Tag | 作用 |
| --- | --- | --- |
| original | `original` | 原始 nanoGPT baseline。 |
| ModernComponents0.1 | `MordenComponents0.1` | 加入 RMSNorm、SwiGLU、RoPE。注意 tag 名保持当时拼写。 |
| tools | `ModernComponents0.1-tools` | 增加日志对比工具，修复 char-level sampling 对 `tiktoken` 的硬依赖。 |
| ModernKVCache0.2 | `ModernKVCache0.2` | 加入 KV Cache generation path 与 benchmark 脚本。 |
| ModernKVCache0.2-hotfix | `ModernKVCache0.2-hotfix` | 修复 benchmark 脚本从 `scripts/` 运行时找不到 `model.py` 的导入路径问题。 |
| AnalysisPlots0.2 | `AnalysisPlots0.2` | 支持合并多段训练日志，增加 zoom/delta/log-scale loss 图。 |
| SamplingBPEMeta0.1 | `SamplingBPEMeta0.1` | 修复 BPE 数据集 `meta.pkl` 没有 `stoi/itos` 时采样失败的问题。 |

当前远程最新提交：

```text
d1461fe Support BPE metadata in sampling tools
```

## 3. 为什么选择这些改进

### 3.1 为什么加入 RMSNorm

原始 nanoGPT 使用 GPT-2 风格 LayerNorm。RMSNorm 是许多现代 LLM 使用的归一化方式，它只归一化 root mean square，不减均值，计算更简单，实践中常用于 LLaMA 等架构。

预期收益：

- 更接近现代 decoder-only LLM 的架构设计。
- 减少归一化计算中的部分操作。
- 为后续扩展到 LLaMA-like block 打基础。

本项目中 RMSNorm 通过配置项启用：

```python
norm_type = 'rmsnorm'
```

默认仍为：

```python
norm_type = 'layernorm'
```

因此原始 checkpoint 和原始训练命令仍然兼容。

### 3.2 为什么加入 SwiGLU

原始 nanoGPT 的 MLP 是：

```text
Linear -> GELU -> Linear
```

现代 LLM 常用 gated MLP，例如 SwiGLU：

```text
Linear -> split(value, gate) -> value * SiLU(gate) -> Linear
```

预期收益：

- 门控机制提高非线性表达能力。
- 与 RMSNorm、RoPE 一起形成更现代的 Transformer Block。
- 在相近参数量下可能带来更好的 validation loss。

代价：

- SwiGLU 有额外 gate 分支，训练计算更重。
- 当前实现未使用融合 kernel，因此吞吐会下降。

配置项：

```python
mlp_type = 'swiglu'
swiglu_hidden_mult = 8/3
```

其中 `8/3` 是常见的近似设置，用来让 SwiGLU MLP 参数量接近传统 `4 * n_embd` GELU MLP。

### 3.3 为什么加入 RoPE

原始 GPT-2 使用 learned absolute position embedding。RoPE 通过旋转 q/k 向量注入位置信息，是现代 LLM 常见的位置编码方式。

预期收益：

- 更接近 LLaMA / DeepSeek 等现代模型的位置编码形式。
- 更自然支持 KV Cache 下的增量位置。
- 避免 learned position embedding 对固定表的依赖。

代价：

- 当前实现每次 forward 动态计算 `cos/sin`，没有预缓存。
- 训练吞吐会下降。
- 若 KV Cache 下不处理 `pos_offset`，生成会出错。

配置项：

```python
position_embedding_type = 'rope'
rope_base = 10000.0
```

### 3.4 为什么加入 KV Cache

原始 nanoGPT 的 `generate()` 每生成一个 token 都会重新 forward 整个上下文：

```text
token 1: forward prompt
token 2: forward prompt + token1
token 3: forward prompt + token1 + token2
...
```

这会重复计算历史 token 的 K/V。KV Cache 的核心是缓存每层 attention 的历史 key/value，只对新增 token 计算新的 q/k/v。

预期收益：

- 减少自回归 decode 阶段重复计算。
- 在长上下文、大模型、长生成时提升 tokens/s。
- 是 LLM 推理系统中的基础工程能力。

重要说明：

- KV Cache 不参与训练，因此不能用训练 loss 证明 KV Cache 有效。
- KV Cache 的有效性必须通过同一个 checkpoint 下的 no-cache vs cache generation benchmark 验证。

## 4. 文件架构与代码改动

### 4.1 `model.py`

主要新增：

- `RMSNorm`
- `build_norm(config)`
- `SwiGLUMLP`
- `build_mlp(config)`
- `apply_rope(q, k, inv_freq, pos_offset=0)`
- `GPT.generate(..., use_kv_cache=False)`
- `GPT._generate_with_kv_cache(...)`

`GPTConfig` 新增字段：

```python
norm_type: str = 'layernorm'
mlp_type: str = 'gelu'
position_embedding_type: str = 'learned'
rope_base: float = 10000.0
swiglu_hidden_mult: float = 8/3
```

KV Cache 相关 forward 改动：

```python
CausalSelfAttention.forward(x, past_kv=None, use_cache=False, pos_offset=0)
Block.forward(x, past_kv=None, use_cache=False, pos_offset=0)
GPT.forward(idx, targets=None, past_kv=None, use_cache=False)
```

设计原则：

- 训练路径默认不启用 cache，返回值仍保持 `(logits, loss)`。
- 只有显式 `use_cache=True` 时才返回 `present_kv`。
- RoPE 在 cache 下使用 `pos_offset`，保证增量 token 的位置正确。
- 当 cache 长度达到 `block_size` 后，回退到最近窗口重建 cache，保持与原始 sliding window 生成行为一致。

### 4.2 `train.py`

新增现代组件配置项：

```python
norm_type = 'layernorm'
mlp_type = 'gelu'
position_embedding_type = 'learned'
rope_base = 10000.0
swiglu_hidden_mult = 8/3
```

这些参数会进入 checkpoint 的 `model_args`。

兼容性处理：

- 新 checkpoint 会保存这些字段。
- 旧 checkpoint 没有这些字段时，resume 会回退到原始 nanoGPT 默认配置：

```text
layernorm / gelu / learned position embedding
```

### 4.3 `sample.py`

新增：

```python
use_kv_cache = False
```

命令行启用：

```bash
--use_kv_cache=True
```

BPE meta 修复：

- char-level 数据集的 `meta.pkl` 有 `stoi/itos`，按字符表解码。
- OpenWebText subset 的 `meta.pkl` 只有 tokenizer 信息，没有 `stoi/itos`，此时回退到 GPT-2 BPE `tiktoken`。

这个修复对应 tag：

```text
SamplingBPEMeta0.1
```

### 4.4 `scripts/benchmark_generate.py`

新增 generation benchmark 工具，用于同一 checkpoint 下对比：

```text
use_kv_cache=False
use_kv_cache=True
```

输出指标：

- prompt tokens
- total time
- average sample time
- tokens/s
- speedup

该脚本也支持 BPE meta，能用于 OpenWebText subset checkpoint。

### 4.5 `scripts/compare_runs.py`

新增/增强日志对比能力：

- 解析 `step N: train loss ..., val loss ...`
- 解析 `iter N: loss ..., time ...ms, mfu ...%`
- 生成 `summary.md`
- 生成 `metrics.json`
- 生成 `loss_curves.png`
- 生成 `loss_curves_log.png`
- 生成 `val_loss_zoom.png`
- 生成 `loss_delta.png`
- 生成 `iter_time.png`

后续修复：

- 支持多个 `--baseline-log` 合并，因为 original baseline 曾经被拆成 `0-10000` 和 `10000-18000` 两段日志。
- `loss_delta.png` 从 step 2500 开始画，避免早期过大的 loss 掩盖后期差异。

## 5. 数据集设计

### 5.1 为什么没有使用完整 OpenWebText

完整 OpenWebText 不适合本项目当前阶段。Karpathy README 中复现 GPT-2 124M 的建议配置是：

```text
8 x A100 40GB
约 4 天
OpenWebText 全量
```

当前实验环境是：

```text
1 x RTX 5090 32GB
```

完整 OpenWebText 下载和预处理也非常重：

- HuggingFace cache 可能几十 GB。
- 全量 `train.bin` 约 17GB。
- 中间缓存和 tokenization 可能占更多空间。
- 曾经尝试全量 OpenWebText，遇到 `datasets` 依赖缺失、多进程 timeout、磁盘占用迅速增长等问题。

因此最终改为：

```text
OpenWebText streaming subset 1M docs
```

### 5.2 OpenWebText subset 1M docs

自定义脚本：

```text
data/openwebtext_subset/prepare.py
```

准备参数：

```bash
--dataset=openwebtext
--num_docs=1000000
--val_docs=5000
--out_dir=data/openwebtext_subset_1000k
```

实际生成结果：

```text
train_docs: 1,000,000
val_docs: 5,000
train_tokens: 约 1.13B
存储格式: uint16 train.bin / val.bin
tokenizer: GPT-2 BPE
```

100k 子集实际生成过：

```text
train_docs = 100,000
train_tokens = 113,016,947
train.bin = 215.56 MB
```

据此估算 1M docs 约为：

```text
train_tokens ≈ 1.13B
train.bin ≈ 2.1GB
```

最终选择 1M docs 的原因：

- 比 Shakespeare char/BPE 更接近真实预训练分布。
- 比完整 OpenWebText 小很多，准备和训练成本可控。
- 规模足够大，不容易像 tiny Shakespeare 那样快速过拟合。
- 适合做 baseline vs modern_v2 的短中程 benchmark。

## 6. 实验环境

服务器环境：

```text
GPU: RTX 5090 32GB * 1
CPU: 25 vCPU Intel Xeon Platinum 8470Q
Memory: 90GB
Python: 3.12
PyTorch: 2.8.0
CUDA: 12.8
```

数据盘：

```text
/root/autodl-tmp
Size: 150G
Available after expansion: about 139G
```

训练使用：

```text
device=cuda
compile=False
dtype=bfloat16
```

为什么 `compile=False`：

- Windows/服务器环境和新 GPU 上 `torch.compile` 可能引入额外不确定性。
- 本项目当前重点是公平对比和稳定复现。
- baseline 与 modern_v2 均使用 `compile=False`，保证对比公平。

## 7. 实验一：Shakespeare char smoke test

### 7.1 目的

第一阶段不追求最终效果，只验证：

- 现代组件能否正确接入。
- forward/backward 是否正常。
- loss 是否下降。
- checkpoint/sample 是否能跑通。

### 7.2 modern_v1 smoke test 结果

配置：

```text
dataset = shakespeare_char
n_layer = 6
n_head = 6
n_embd = 384
block_size = 256
batch_size = 64
dropout = 0.2
norm_type = rmsnorm
mlp_type = swiglu
position_embedding_type = rope
```

20 iter smoke test：

| step | train loss | val loss |
| --- | ---: | ---: |
| 0 | 4.3649 | 4.3577 |
| 10 | 3.2938 | 3.3399 |
| 20 | 2.7857 | 2.7933 |

结论：

- 现代组件版本可以稳定训练。
- loss 快速下降，说明 forward/backward 和 optimizer 路径正常。
- 该阶段目标达成。

### 7.3 Shakespeare char original vs modern_v1

在 Shakespeare char 上完成 original 与 modern_v1 对比：

| metric | original | modern_v1 | change |
| --- | ---: | ---: | ---: |
| params_m | 10.6500 | 10.6500 | +0.0000 |
| final_train_loss | 0.6246 | 0.4635 | +25.79% |
| final_val_loss | 1.7077 | 1.9052 | -11.57% |
| avg_tail_iter_time_ms | 92.4449 | 123.5483 | -33.65% |
| avg_tail_mfu_percent | 24.8801 | 18.8644 | -24.18% |
| avg_tail_tokens_per_sec | 1,094,275.3287 | 817,043.4666 | -25.33% |

分析：

- modern_v1 train loss 更低，但 val loss 更高，说明在 tiny Shakespeare char 上更容易过拟合。
- 训练速度下降，主要来自 SwiGLU 和 RoPE 的额外计算。
- 这不是失败，而是说明现代组件已接入并可训练，但小数据集不能充分体现泛化优势。

## 8. 实验二：OpenWebText subset 1M large benchmark

### 8.1 目的

使用更真实、更大的 BPE 预训练数据，对比：

```text
original baseline
modern_v2 = RMSNorm + SwiGLU + RoPE + KV Cache code path
```

注意：

- 训练对比主要验证 RMSNorm/SwiGLU/RoPE 的效果。
- KV Cache 不参与训练，因此训练 loss 不能证明 KV Cache 有效。
- KV Cache 需通过 generation benchmark 单独验证。

### 8.2 模型与训练参数

最终正式配置：

```text
dataset = openwebtext_subset_1000k
block_size = 512
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1
batch_size = 8
gradient_accumulation_steps = 8
tokens_per_iter = 8 * 8 * 512 = 32,768
max_iters = 18,000
eval_interval = 1,000
eval_iters = 100
log_interval = 20
compile = False
device = cuda
```

为什么这样设置：

- `block_size=512`：比前期 `256` 更接近长上下文，也能提高 GPU 利用率。
- `n_layer=8, n_head=8, n_embd=512`：模型约 50M 参数，明显大于 smoke test 的 10M，小到单卡 5090 可稳定训练。
- `batch_size=8, grad_accum=8`：每 iter 32,768 tokens，兼顾吞吐和显存。
- `max_iters=18,000`：由于 original eval-only 误用导致继续训练到 18k，最终将 modern_v2 也训练到 18k，保证公平。
- `eval_interval=1000`：提供足够密集的曲线点，同时避免频繁 eval 拖慢训练。
- `eval_iters=100`：比 smoke test 更稳定，但不会过慢。

### 8.3 original baseline 训练

original baseline 使用：

```text
LayerNorm + GELU MLP + learned absolute position embedding
```

训练过程：

- 首先训练到 10,000 iter。
- 之后原计划单独 eval，但原始 nanoGPT 的 `eval_only` 在 resume checkpoint 时不会退出，导致继续训练到 18,000 iter。
- 该现象被记录并保留，最终将 modern_v2 也训练到 18,000 iter 做公平对比。

关键结果：

```text
step 10000: train loss 4.0108, val loss 4.0403
checkpoint later reached iter_num = 18000
best_val_loss = 3.9126
```

最终用于对比的 baseline：

```text
original_owt_subset_1000k_large_18000iter
```

### 8.4 modern_v2 训练

modern_v2 使用：

```text
RMSNorm + SwiGLU + RoPE
```

训练到 18,000 iter：

```text
step 18000: train loss 3.8176, val loss 3.8229
```

### 8.5 original vs modern_v2 对比结果

| metric | original | modern_v2 | change |
| --- | ---: | ---: | ---: |
| params_m | 50.9100 | 50.9000 | -0.0100 |
| final_train_loss | 3.9162 | 3.8176 | +2.52% |
| final_val_loss | 3.9126 | 3.8229 | +2.29% |
| avg_tail_iter_time_ms | 205.1182 | 392.3937 | -91.30% |
| avg_tail_mfu_percent | 16.9399 | 10.5723 | -37.59% |
| avg_tail_tokens_per_sec | 159,821.3309 | 99,940.2181 | -37.47% |

### 8.6 结果分析

质量侧：

- modern_v2 在几乎相同参数量下，将 final val loss 从 `3.9126` 降到 `3.8229`。
- 相对改善约 `2.29%`。
- final train loss 也降低约 `2.52%`。
- 说明 RMSNorm + SwiGLU + RoPE 在 OpenWebText subset 上带来更好的短中程建模效果。

效率侧：

- modern_v2 的 tail tokens/s 从 `159.8k` 降到 `99.9k`。
- 训练吞吐下降约 `37.47%`。
- average iter time 从 `205ms` 增加到 `392ms`。

这说明：

```text
modern_v2 用训练效率换来了更低的 validation loss。
```

主要原因：

- SwiGLU 比 GELU MLP 多 gate 分支。
- RoPE 当前每次 forward 动态计算 `sin/cos`，没有预缓存。
- 当前实现没有使用 fused kernel。
- PyTorch eager 模式下小型自定义组件开销更明显。

因此该结果应该被解释为：

```text
现代组件带来质量改善，但当前实现不是训练性能优化版。
```

## 9. 实验三：KV Cache 推理 benchmark

### 9.1 目的

验证 KV Cache 本身是否有效。该实验使用同一个 modern_v2 checkpoint：

```text
result/openwebtext_subset_1000k_large/modern_v2/out_18000
```

只改变 generation 是否启用 cache：

```text
use_kv_cache=False
use_kv_cache=True
```

这样可以隔离 KV Cache 对推理速度的影响，避免与模型结构、训练步数混在一起。

### 9.2 推理参数

```text
checkpoint = modern_v2 out_18000
dataset = OpenWebText subset 1M docs
model params = 50.90M
block_size = 512
dtype = bfloat16
prompt = "The future of artificial intelligence is"
prompt_tokens = 6
num_samples = 5
max_new_tokens = 400 / 500
```

为什么选择 400 和 500：

- `block_size=512`。
- prompt 约 6 tokens。
- 400 tokens 是较长但不贴近上限的生成长度。
- 500 tokens 接近上下文窗口上限，更能观察长生成下的 cache 表现。

### 9.3 KV Cache 结果

| max_new_tokens | no-cache tokens/s | kv-cache tokens/s | speedup |
| --- | ---: | ---: | ---: |
| 400 | 237.30 | 246.11 | 1.0371x |
| 500 | 232.66 | 241.35 | 1.0373x |

换算为提升比例：

```text
400 tokens: +3.71%
500 tokens: +3.73%
```

### 9.4 KV Cache 结果分析

结论：

```text
KV Cache 在同一个 modern_v2 checkpoint 上带来稳定正向推理吞吐提升，约 3.7%。
```

提升不大的原因：

- 模型规模仍较小，约 50.9M 参数。
- 上下文窗口只有 512。
- 当前 KV Cache 实现偏教学清晰版，不是 fused kernel 优化版。
- generation 端到端时间中还包括 Python 循环、采样、kernel launch 等开销。
- 当生成长度接近 `block_size` 时，cache 接近窗口上限，收益受限。

因此该实验结论应写为：

```text
KV Cache 路径已实现并验证有效，但当前小模型/短上下文设置下端到端加速有限。
```

而不是夸大为：

```text
KV Cache 大幅提升推理速度。
```

## 10. 遇到的问题与解决方案

### 10.1 OpenWebText 全量下载不现实

问题：

- 全量 OpenWebText 下载和 tokenization 重。
- `datasets` 依赖缺失。
- 多进程下载/映射触发 `httpx client closed`、`multiprocess TimeoutError`。
- 数据盘快速占用到 80GB 以上。

解决：

- 放弃完整 OpenWebText。
- 使用 streaming 方式构造 OpenWebText subset。
- 最终选用 1M docs，生成约 1.13B tokens。

价值：

- 仍然比 Shakespeare 更接近真实预训练语料。
- 成本可控。
- 足以做 baseline vs modern_v2 对比。

### 10.2 AutoDL 数据盘容量理解问题

问题：

- 初看 `/root/autodl-tmp` 似乎只有 12G。
- 实际 `du -sh` 显示的是已用空间，不是总容量。

解决：

使用：

```bash
df -h /root/autodl-tmp
```

确认扩容后：

```text
Size = 150G
Avail = 139G
```

### 10.3 `/autodl-pub` 没有 OpenWebText

问题：

- 试图在公共盘寻找 OpenWebText 或 nanoGPT `train.bin/val.bin`。

检查命令：

```bash
find /autodl-pub -iname "*openwebtext*"
find /autodl-pub -iname "train.bin"
find /autodl-pub -iname "val.bin"
```

结果：

```text
没有可直接使用的 OpenWebText。
```

解决：

- 自己使用 streaming 构造 subset。

### 10.4 `sample.py` 强依赖 `tiktoken`

问题：

- char-level checkpoint 实际有 `meta.pkl`，不需要 `tiktoken`。
- 但原脚本顶部直接 `import tiktoken`，导致没安装时 sample 失败。

解决：

- 将 `tiktoken` 改为懒加载，只在需要 GPT-2 BPE tokenizer 时导入。

### 10.5 BPE 数据集 meta 没有 `stoi/itos`

问题：

OpenWebText subset 的 `meta.pkl` 是：

```text
vocab_size = 50257
tokenizer = gpt2
```

没有 char-level 的：

```text
stoi / itos
```

原 `sample.py` 看到 `meta.pkl` 就默认按 char-level 解码，导致：

```text
KeyError: 'stoi'
```

解决：

- 如果 meta 有 `stoi/itos`，走 char-level。
- 否则回退 GPT-2 BPE `tiktoken`。
- 同步修复 `scripts/benchmark_generate.py`。

### 10.6 `eval_only` resume 后继续训练

问题：

原始 nanoGPT 逻辑：

```python
if iter_num == 0 and eval_only:
    break
```

如果从 checkpoint resume：

```text
iter_num = 10000
```

则：

```python
iter_num == 0
```

为 false，导致 `--eval_only=True` 在 resume 后继续训练。

影响：

- original baseline 从 10,000 iter 继续训练到了 18,000 iter。

解决：

- 不再对 resume checkpoint 使用原始 `train.py --eval_only=True` 做单独评估。
- 使用训练日志中自动 eval 的 `step N: train loss, val loss`。
- 为保证公平，将 modern_v2 也训练到 18,000 iter。
- 重命名 original 结果目录和日志，避免误解。

建议后续改进：

- 新增独立 `scripts/eval_checkpoint.py`，只加载 checkpoint 评估，不进入训练循环。
- 或修改 `train.py`：

```python
if eval_only:
    break
```

但这会改变原始行为，应单独提交。

### 10.7 baseline 日志被拆成两段

问题：

original baseline 分为：

```text
train_first_10000.log
train_18000_continued_from_eval_command.log
```

对比脚本最初只读取后半段，导致 `val_loss_zoom.png` 中 original 蓝线只有半截。

解决：

- `scripts/compare_runs.py` 支持多个 `--baseline-log`。
- 自动按 step 合并 eval 点。
- 新增 `loss_delta.png` 和 `loss_curves_log.png`。

### 10.8 KV Cache benchmark 初期提升不明显

问题：

在 Shakespeare char 小模型上：

```text
speedup ≈ 1.0x
```

原因：

- 模型仅 10.65M。
- `block_size=256`。
- prompt 太短或超过 block_size 后被裁剪。
- cache 很快达到窗口上限并重建。

解决：

- 在更大模型、更大数据、更大 `block_size=512` 的 modern_v2 checkpoint 上重新 benchmark。
- 最终得到稳定约 `3.7%` 的推理吞吐提升。

## 11. 当前结论

### 11.1 Modern components 结论

在 OpenWebText streaming subset 1M docs 上：

```text
original val loss = 3.9126
modern_v2 val loss = 3.8229
relative improvement = 2.29%
```

说明：

```text
RMSNorm + SwiGLU + RoPE 在相近参数量下改善了短中程语言建模效果。
```

代价：

```text
training tokens/s 下降约 37.47%
```

因此：

```text
现代组件版本质量更好，但训练效率更低。
```

### 11.2 KV Cache 结论

在同一个 modern_v2 checkpoint 上：

```text
400-token generation speedup = 1.0371x
500-token generation speedup = 1.0373x
```

说明：

```text
KV Cache 推理路径有效，带来稳定正向吞吐提升。
```

但由于模型规模和上下文长度有限，提升幅度约为 3.7%，不应夸大。

### 11.3 综合结论

本项目完成了从原始 nanoGPT 到 modern_v2 的系统改造：

- 模型结构从 GPT-2 风格 Block 扩展到现代 LLM Block。
- 推理路径支持 KV Cache。
- 数据集从 tiny Shakespeare 扩展到 OpenWebText streaming subset 1M docs。
- 建立了训练 loss、val loss、吞吐、MFU、generation speedup 的实验分析工具。
- 记录并解决了数据集、meta、采样、日志合并、eval-only resume 等实际工程问题。

该项目可以作为大模型算法实习项目的一部分，核心亮点是：

```text
从底层实现现代 Transformer 组件与 KV Cache，并构建可复现实验对照，量化质量和效率 trade-off。
```

## 12. 可以写进简历的表述

建议表述：

```text
基于 nanoGPT 实现现代 LLM 组件改造与推理优化：将原始 GPT-2 Block 扩展为可配置的 RMSNorm、SwiGLU、RoPE 架构，并实现 KV Cache 自回归生成路径；构建 OpenWebText streaming subset 1M docs 训练基准和日志分析工具。实验显示，在约 50M 参数、18k iter 设置下，modern_v2 相比 original 将 validation loss 从 3.9126 降至 3.8229（相对降低 2.29%），并在同一 checkpoint 的 generation benchmark 中通过 KV Cache 获得约 3.7% tokens/s 提升。
```

更稳健的补充：

```text
同时分析了现代组件带来的训练吞吐下降问题，定位主要来自未融合 SwiGLU/RoPE 实现，并通过 loss delta、zoomed validation curve、log-scale curve 等图表展示质量-效率 trade-off。
```

不建议写：

```text
复现 GPT-2。
```

也不建议写：

```text
KV Cache 大幅提升推理速度。
```

更准确的说法是：

```text
实现并验证 KV Cache 在当前设置下带来稳定但有限的推理吞吐提升。
```

## 13. 后续优化方向

### 13.1 RoPE 缓存优化

当前 RoPE 每次 forward 动态计算 `cos/sin`。后续可：

- 在模块初始化时预计算最大 `block_size` 的 cos/sin。
- 按 `pos_offset` 切片。
- 减少训练和推理开销。

预期：

```text
提升 modern_v2 训练吞吐。
```

### 13.2 KV Cache 更严格消融

当前 benchmark 是端到端 generation。后续可增加：

- decode-only latency
- prefill latency
- per-token latency
- 不同 prompt length
- 不同 `max_new_tokens`
- 更大 `block_size=1024`

### 13.3 独立 checkpoint eval 脚本

新增：

```text
scripts/eval_checkpoint.py
```

避免 `train.py --eval_only=True --init_from=resume` 继续训练的问题。

### 13.4 训练效率优化

可尝试：

- `torch.compile=True`
- 预缓存 RoPE
- fused SwiGLU
- 更大的 batch
- 更大的 `block_size`

但这些应作为新实验，不与当前结果混淆。

### 13.5 mini-GRPO

下一阶段可在 modern_v2 基础上加入 mini-GRPO：

- 构造可自动判分任务。
- 实现 group rollout。
- 实现 rule-based reward。
- 实现 group relative advantage。
- 加入 KL penalty。
- 评估 reward、accuracy、format pass rate。

该方向用于证明强化学习对齐理解，不应声称复现 DeepSeek-R1。
