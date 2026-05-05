# nanoGPT-modified 新线程交接说明

本文用于在新 Codex 线程中快速恢复项目上下文，继续实现第三版本：mini-GRPO 强化学习对齐。

## 当前仓库状态

远程仓库：

```text
https://github.com/daimaomaolq/nanoGPT-modified
```

当前主线已完成两个版本：

| 版本 | Tag | 说明 |
| --- | --- | --- |
| original | `original` | 原始 nanoGPT baseline。 |
| ModernComponents0.1 | `MordenComponents0.1` | 加入 RMSNorm、SwiGLU、RoPE。 |
| ModernKVCache0.2 | `ModernKVCache0.2` | 在 modern components 基础上加入 KV Cache generation path。 |

后续工具/修复标签：

```text
ModernComponents0.1-tools
ModernKVCache0.2-hotfix
AnalysisPlots0.1
AnalysisPlots0.2
SamplingBPEMeta0.1
```

重要报告：

```text
PROJECT_IMPROVEMENT_ROADMAP.md
MODERN_KVCACHE_EXPERIMENT_REPORT.md
```

`MODERN_KVCACHE_EXPERIMENT_REPORT.md` 是 v1/v2 的完整阶段报告，包含代码改动、实验设置、结果、问题复盘和简历表述。

## 已完成的核心代码改动

### `model.py`

新增现代组件：

- `RMSNorm`
- `SwiGLUMLP`
- RoPE：`apply_rope(q, k, inv_freq, pos_offset=0)`

新增配置字段：

```python
norm_type
mlp_type
position_embedding_type
rope_base
swiglu_hidden_mult
```

新增 KV Cache 推理路径：

```python
CausalSelfAttention.forward(..., past_kv=None, use_cache=False, pos_offset=0)
Block.forward(..., past_kv=None, use_cache=False, pos_offset=0)
GPT.forward(..., past_kv=None, use_cache=False)
GPT.generate(..., use_kv_cache=False)
```

训练路径默认不启用 cache，保持 `(logits, loss)` 返回格式。

### `train.py`

支持现代组件配置项，并保存到 checkpoint 的 `model_args`。

注意：原始 nanoGPT 的 `eval_only` 在 resume checkpoint 时存在坑：

```python
if iter_num == 0 and eval_only:
    break
```

如果 `init_from=resume` 且 `iter_num > 0`，`--eval_only=True` 会继续训练。后续如果需要单独评估 checkpoint，建议新增独立脚本 `scripts/eval_checkpoint.py`，不要复用 `train.py --eval_only=True --init_from=resume`。

### `sample.py`

支持：

```bash
--use_kv_cache=True
```

并修复了两类采样问题：

- char-level checkpoint 不再强制顶部 import `tiktoken`。
- BPE 数据集 `meta.pkl` 没有 `stoi/itos` 时，自动回退到 GPT-2 BPE `tiktoken`。

### `scripts/benchmark_generate.py`

用于同一个 checkpoint 下 benchmark：

```text
no-cache generation
kv-cache generation
```

输出：

- prompt tokens
- total time
- avg time
- tokens/s
- speedup

### `scripts/compare_runs.py`

用于训练日志对比，支持多日志合并。

输出：

```text
summary.md
metrics.json
loss_curves.png
loss_curves_log.png
val_loss_zoom.png
loss_delta.png
iter_time.png
```

## 已完成实验结果

### Shakespeare char smoke test

modern components 20 iter smoke：

| step | train loss | val loss |
| --- | ---: | ---: |
| 0 | 4.3649 | 4.3577 |
| 10 | 3.2938 | 3.3399 |
| 20 | 2.7857 | 2.7933 |

结论：RMSNorm + SwiGLU + RoPE 能正常训练。

### Shakespeare char original vs modern_v1

| metric | original | modern_v1 | change |
| --- | ---: | ---: | ---: |
| params_m | 10.6500 | 10.6500 | +0.0000 |
| final_train_loss | 0.6246 | 0.4635 | +25.79% |
| final_val_loss | 1.7077 | 1.9052 | -11.57% |
| avg_tail_iter_time_ms | 92.4449 | 123.5483 | -33.65% |
| avg_tail_tokens_per_sec | 1,094,275.3287 | 817,043.4666 | -25.33% |

结论：小数据集上 modern_v1 更易过拟合，训练更慢。

### OpenWebText streaming subset 1M docs

完整 OpenWebText 不现实，因此使用 streaming 方式构造了子集：

```text
dataset = openwebtext
num_docs = 1,000,000
val_docs = 5,000
tokenizer = GPT-2 BPE
train_tokens ≈ 1.13B
```

正式 large benchmark 配置：

```text
block_size = 512
n_layer = 8
n_head = 8
n_embd = 512
batch_size = 8
gradient_accumulation_steps = 8
tokens_per_iter = 32,768
max_iters = 18,000
eval_interval = 1,000
eval_iters = 100
dropout = 0.1
compile = False
```

original baseline 曾因误用 resume eval 继续训练，所以最终统一比较 18,000 iter。

OpenWebText subset 1M large 18k 结果：

| metric | original | modern_v2 | change |
| --- | ---: | ---: | ---: |
| params_m | 50.9100 | 50.9000 | -0.0100 |
| final_train_loss | 3.9162 | 3.8176 | +2.52% |
| final_val_loss | 3.9126 | 3.8229 | +2.29% |
| avg_tail_iter_time_ms | 205.1182 | 392.3937 | -91.30% |
| avg_tail_mfu_percent | 16.9399 | 10.5723 | -37.59% |
| avg_tail_tokens_per_sec | 159,821.3309 | 99,940.2181 | -37.47% |

结论：

- modern_v2 在相近参数量下 val loss 降低约 2.29%。
- 训练吞吐下降约 37.47%。
- 现代组件带来更好建模效果，但当前未融合实现牺牲训练效率。

### KV Cache benchmark

同一个 modern_v2 checkpoint：

```text
result/openwebtext_subset_1000k_large/modern_v2/out_18000
```

prompt：

```text
The future of artificial intelligence is
```

结果：

| max_new_tokens | no-cache tokens/s | kv-cache tokens/s | speedup |
| --- | ---: | ---: | ---: |
| 400 | 237.30 | 246.11 | 1.0371x |
| 500 | 232.66 | 241.35 | 1.0373x |

结论：KV Cache 推理路径有效，带来稳定约 3.7% 的 generation 吞吐提升。由于模型约 50.9M、`block_size=512`、实现未融合，端到端加速幅度有限。

## 服务器结果目录

服务器项目路径：

```text
/root/autodl-tmp/nanoGPT-modern-v1
```

重要结果目录：

```text
result/original
result/modern_v1
result/modern_kvcache_v2
result/openwebtext_subset_1000k_large
```

OpenWebText large 结果：

```text
result/openwebtext_subset_1000k_large/original
result/openwebtext_subset_1000k_large/modern_v2
result/openwebtext_subset_1000k_large/comparisons/original_vs_modern_v2_18000iter
```

注意 original 的日志拆成两段：

```text
result/openwebtext_subset_1000k_large/original/logs/train_first_10000.log
result/openwebtext_subset_1000k_large/original/logs/train_18000_continued_from_eval_command.log
```

对比时需要两个都传给 `scripts/compare_runs.py`。

## 已知坑

1. 不要用原始 `train.py --eval_only=True --init_from=resume` 做 checkpoint eval，会继续训练。
2. BPE 数据集 `meta.pkl` 没有 `stoi/itos`，采样工具必须回退 tiktoken。
3. KV Cache 不参与训练，不能用训练 loss 证明 KV Cache 有效。
4. OpenWebText 全量对单 RTX 5090 三天项目不现实，使用 streaming subset 更合理。
5. modern_v2 训练更慢，主要来自 SwiGLU 和当前动态 RoPE 实现。

## 下一版本目标：mini-GRPO

第三版本建议命名：

```text
ModernKVCacheGRPO0.3
```

目标不是复现 DeepSeek-R1，而是在 nanoGPT 上实现一个边界清楚的 mini-GRPO 对齐实验。

推荐实验设计：

1. 构造可自动判分小任务：
   - 一位/两位整数加减乘
   - 输出格式 `<answer>...</answer>`
2. 先做 SFT，让模型学会格式和基本答案。
3. 再做 GRPO：
   - 对每个 prompt 采样 G 个 responses。
   - 使用 rule-based reward。
   - 组内 reward 标准化，计算 relative advantage。
   - 加 KL penalty，约束当前 policy 不偏离 reference policy。
   - 不引入 critic。
4. 评估：
   - reward
   - accuracy
   - format pass rate
   - KL
   - response length

推荐目录：

```text
grpo/
  data.py
  rewards.py
  train_sft.py
  train_grpo.py
  evaluate.py
  README.md
```

推荐先用小模型，不要直接用 OpenWebText large checkpoint：

```text
n_layer = 4
n_head = 4
n_embd = 256
block_size = 128 或 256
```

原因：

- GRPO 需要多 response sampling，计算成本高。
- 小任务更容易自动判分。
- 简历重点是展示 RL 对齐流程，而不是大规模推理能力。

建议简历表述：

```text
在 nanoGPT 上实现 mini-GRPO 强化学习对齐流程，包含 group rollout、rule-based reward、relative advantage、KL penalty 和 policy gradient 更新；在可自动判分任务上评估 reward、accuracy 与 format pass rate。
```

不要写：

```text
复现 DeepSeek-R1。
```
