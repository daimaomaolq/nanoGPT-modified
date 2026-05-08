# ModernKVCacheGRPO0.3 实验报告

本文只记录 nanoGPT-modified 第三阶段 v3：`ModernKVCacheGRPO0.3`，即在已有 modern components + KV Cache 版本基础上实现 mini-GRPO 强化学习对齐流程，并在可自动判分的算术任务上验证 SFT、GRPO、SFT-continued 和 group-size 消融。

本阶段目标不是复现 DeepSeek-R1，也不是训练通用数学推理模型，而是在 nanoGPT 这样的小型代码库中实现一个边界清楚、可复现、可分析的 RL 对齐实验闭环。

## 1. 实验目标

### 1.1 目标定义

v3 的目标是实现一个 mini-GRPO 对齐流程，包含：

- SFT warmup：先让模型学会固定输出格式和基础算术答案。
- Group rollout：每个 prompt 采样多个 responses。
- Rule-based reward：用规则判断 `<answer>...</answer>` 格式和答案正确性。
- Relative advantage：在同一 prompt 的 group 内做 reward 标准化。
- KL penalty：约束当前 policy 不要过度偏离 SFT reference policy。
- No critic：不训练 value model 或 critic。
- Evaluation：评估 accuracy、average reward、format pass rate、invalid answer rate、response length、KL。

推荐表述：

```text
在 nanoGPT 上实现 mini-GRPO 强化学习对齐流程，包含 group rollout、rule-based reward、relative advantage、KL penalty 和无 critic 策略优化；在可自动判分任务上评估 reward、accuracy、format pass rate、invalid answer rate 和 KL。
```

不建议表述：

```text
复现 DeepSeek-R1。
```

### 1.2 为什么选择算术任务

GRPO 需要自动 reward。算术任务有几个优点：

- 可自动生成无限训练样本。
- 有精确 gold answer，便于 rule-based reward。
- 格式简单，方便判断 reward hacking。
- 小模型也有机会学到非随机能力。
- 不需要人工标注 reward model。

但它也有明显局限：

- 它非常适合 SFT，因为每个样本都有精确 token-level gold answer。
- 答案很短，SFT-continued 会成为很强 baseline。
- synthetic 模板分布较窄，不代表真实开放推理。
- reward 容易饱和或全错，导致 GRPO advantage 信号不足。

因此本实验重点是验证 mini-GRPO 机制和分析 trade-off，而不是证明 RL 一定优于 SFT。

## 2. 代码结构

新增目录：

```text
grpo/
  README.md
  __init__.py
  data.py
  rewards.py
  policy.py
  train_sft.py
  train_grpo.py
  evaluate.py
```

各文件职责：

| 文件 | 作用 |
| --- | --- |
| `grpo/data.py` | 合成 arithmetic 数据、task-local char tokenizer、public benchmark 懒加载。 |
| `grpo/rewards.py` | 解析 `<answer>...</answer>`，计算 rule reward。 |
| `grpo/policy.py` | 计算 response-token logprobs 和 sampled KL。 |
| `grpo/train_sft.py` | SFT warmup 和 SFT-continued baseline。 |
| `grpo/train_grpo.py` | GRPO group rollout、advantage、KL penalty 和 policy update。 |
| `grpo/evaluate.py` | synthetic/public benchmark 评估。 |

核心设计原则：

- 不大改 `model.py`，复用已有 `GPT.forward()` 和 `GPT.generate()`。
- GRPO 训练独立于原始 `train.py`，避免污染语言模型预训练路径。
- checkpoint 保存 `model_args`、`tokenizer` 和训练参数，便于独立评估。

## 3. 数据与 Reward 设计

### 3.1 Synthetic Arithmetic

合成任务分为三个 stage：

| Stage | 范围 | 操作 | 用途 |
| --- | --- | --- | --- |
| `easy` | 0-9 | `+`, `-` | 简单分布 sanity / 泛化检查。 |
| `medium` | 0-99 | `+`, `-` | SFT 主训练任务。 |
| `hard` | 0-99 | `+`, `-`, `*` | GRPO 主训练和主评估任务。 |

样本格式：

```text
Question: What is 37 * 8?
Answer: <answer>296</answer>
```

SFT 输入是完整 prompt + response，但 loss 只计算 response token。prompt token 使用 `-1` mask 掉。

### 3.2 Tokenizer

本阶段使用 task-local character tokenizer：

- 特殊 token：`<pad>`, `<bos>`, `<eos>`, `<unk>`。
- 字符集：ASCII printable 字符 + newline。
- checkpoint 内保存 tokenizer state。

选择 char-level tokenizer 的原因：

- 实现简单。
- 不依赖 GPT-2 BPE 或 tiktoken。
- 合成算术任务可完全覆盖。
- 便于在小模型上快速验证 RL loop。

代价：

- 英文 word-problem benchmark 在 char-level 下 prompt 很长。
- `block_size=256` 对 MAWPS/SVAMP/GSM8K 不够友好。
- 后续公开 benchmark 需要 `block_size=512/768` 或 BPE tokenizer 版本。

### 3.3 Reward 函数

当前 reward：

```text
format ok: +0.2
answer correct: +1.0
invalid format: -0.2
too long response: -0.2
```

单样本最高 reward：

```text
1.2
```

解析规则：

```text
<answer>[-+]?\d+</answer>
```

只接受严格 answer tag。这样可以降低 reward hacking 风险，例如输出一堆数字但没有合法 tag 不会得高分。

## 4. 模型与训练参数

### 4.1 模型配置

正式 v3 使用小模型：

```text
n_layer = 4
n_head = 4
n_embd = 256
block_size = 256
norm_type = rmsnorm
mlp_type = swiglu
position_embedding_type = rope
dtype = bfloat16
compile = False
```

训练日志：

```text
number of parameters: 3.17M
```

为什么不用 v2 的 50M OpenWebText checkpoint：

- GRPO 需要多 response sampling，计算成本随 `batch_size * group_size * max_new_tokens` 增长。
- 算术任务和 OpenWebText 语言建模分布不同。
- v3 简历重点是 RL 对齐流程，而不是大模型规模。
- 小模型更便于快速消融和排错。

### 4.2 SFT 参数

SFT full run：

```bash
python grpo/train_sft.py \
  --out_dir result/grpo_v3/sft \
  --device cuda \
  --dtype bfloat16 \
  --max_iters 3000 \
  --eval_interval 250 \
  --eval_iters 50 \
  --batch_size 64 \
  --block_size 256 \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 256 \
  --norm_type rmsnorm \
  --mlp_type swiglu \
  --position_embedding_type rope \
  --compile False
```

SFT 使用 `stage=medium`，即两位数加减。

SFT 训练结果片段：

```text
step 2500: val_sft_loss 0.0074
step 2750: val_sft_loss 0.0088
```

解释：

- loss 已经非常低。
- 后期 `0.0074 -> 0.0088` 是低 loss 区间的随机 eval 抖动，不代表训练失败。
- SFT 已经学会格式和 medium 加减。

### 4.3 GRPO-G4 参数

主 GRPO 实验：

```bash
python grpo/train_grpo.py \
  --init_from result/grpo_v3/sft/ckpt.pt \
  --reference_from result/grpo_v3/sft/ckpt.pt \
  --out_dir result/grpo_v3/grpo \
  --stage hard \
  --device cuda \
  --dtype bfloat16 \
  --max_iters 1000 \
  --eval_interval 100 \
  --batch_size 16 \
  --group_size 4 \
  --max_new_tokens 32 \
  --kl_coef 0.02 \
  --learning_rate 1e-5 \
  --compile False
```

每步 rollout response 数：

```text
16 * 4 = 64 responses / iter
```

总 rollout response budget：

```text
1000 * 64 = 64,000 responses
```

### 4.4 SFT-continued 参数

公平 baseline：

```bash
python grpo/train_sft.py \
  --init_from result/grpo_v3/sft/ckpt.pt \
  --out_dir result/grpo_v3/sft_continued \
  --stage hard \
  --device cuda \
  --dtype bfloat16 \
  --max_iters 1000 \
  --eval_interval 250 \
  --eval_iters 50 \
  --batch_size 64 \
  --compile False
```

为什么必须有这个 baseline：

- GRPO 相比 SFT-only 提升不一定说明 RL 有独特价值。
- 可能只是因为模型多训练了 1000 步。
- SFT-continued 控制了“额外训练”这个变量。

### 4.5 GRPO-G8-250 参数

Group-size 消融：

```bash
python grpo/train_grpo.py \
  --init_from result/grpo_v3/sft/ckpt.pt \
  --reference_from result/grpo_v3/sft/ckpt.pt \
  --out_dir result/grpo_v3/grpo_g8_250 \
  --stage hard \
  --device cuda \
  --dtype bfloat16 \
  --max_iters 250 \
  --eval_interval 50 \
  --batch_size 32 \
  --group_size 8 \
  --max_new_tokens 32 \
  --kl_coef 0.02 \
  --learning_rate 1e-5 \
  --compile False
```

每步 rollout response 数：

```text
32 * 8 = 256 responses / iter
```

为了和 GRPO-G4 做 same rollout budget 对比：

```text
G4: 1000 * 64 = 64,000 responses
G8: 250 * 256 = 64,000 responses
```

注意：same rollout budget 不等于 same optimizer updates。

```text
G4: 1000 optimizer updates
G8: 250 optimizer updates
```

这成为后续分析中的一个重要点。

## 5. GRPO 理论与参数优化逻辑

### 5.1 GRPO Objective

对每个 prompt 采样 `G` 个 responses：

```text
r_1, r_2, ..., r_G
```

组内标准化 advantage：

```text
A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
```

训练目标：

```text
loss = -mean(A_i * logprob(response_i)) + kl_coef * KL(policy || reference)
```

本实现不引入 critic，不训练 value function。

### 5.2 为什么要从 G4 尝试到 G8

GRPO 依赖组内 reward 差异。如果一个 group 全对或全错：

```text
reward = [1.2, 1.2, 1.2, 1.2]
```

或：

```text
reward = [-0.2, -0.2, -0.2, -0.2]
```

那么：

```text
std ≈ 0
advantage ≈ 0
policy gradient ≈ 0
```

假设单条 response 正确概率 `p = 0.63`，一个 group 中同时出现正确和错误 response 的概率：

```text
P(useful group) = 1 - p^G - (1-p)^G
```

当 `G=4`：

```text
P = 1 - 0.63^4 - 0.37^4
  ≈ 1 - 0.1575 - 0.0187
  ≈ 0.8238
```

当 `G=8`：

```text
P = 1 - 0.63^8 - 0.37^8
  ≈ 1 - 0.0248 - 0.00035
  ≈ 0.9749
```

理论上，G8 更容易产生有效组内对比，因此可能提供更稳定 advantage。

### 5.3 为什么增大 batch_size

每步 response 数：

```text
N = batch_size * group_size
```

G4：

```text
N = 16 * 4 = 64
```

G8：

```text
N = 32 * 8 = 256
```

梯度估计方差近似满足：

```text
std(gradient estimate) ∝ 1 / sqrt(N)
```

因此从 64 responses 增加到 256 responses：

```text
std_new / std_old = sqrt(64 / 256) = 1/2
```

理论上每步梯度噪声下降约一半。

### 5.4 为什么不优先调 learning rate

learning rate 控制更新步长：

```text
theta <- theta - lr * gradient
```

但如果 advantage 本身经常接近 0 或噪声大，直接调大 learning rate 只会放大噪声。

因此本阶段优先调整：

```text
group_size / batch_size
```

再考虑：

```text
learning_rate, kl_coef, temperature
```

## 6. 实验结果

### 6.1 SFT-only

SFT medium test：

```text
accuracy = 0.971
format_pass_rate = 1.000
average_reward = 1.171
invalid_answer_rate = 0.000
```

SFT hard test：

```text
accuracy = 0.631
format_pass_rate = 0.992
average_reward = 0.8278
invalid_answer_rate = 0.008
```

SFT easy test：

```text
accuracy = 0.700
format_pass_rate = 1.000
average_reward = 0.900
invalid_answer_rate = 0.000
```

分析：

- SFT 在 medium 上很强，说明格式和两位数加减已经学会。
- hard 加入乘法后 accuracy 降到 63.1%，说明有 GRPO 优化空间。
- easy 反而只有 70.0%，说明 easy 虽然表面简单，但它是 medium 的分布子集，短答案、负数、0、一位数边界样例比例不同，SFT 并未自动泛化到满分。

### 6.2 GRPO-G4

GRPO-G4 hard test：

```text
accuracy = 0.652
format_pass_rate = 1.000
average_reward = 0.852
invalid_answer_rate = 0.000
average_kl = 0.022831
```

对比 SFT hard：

```text
accuracy: 0.631 -> 0.652  (+2.1 points)
reward:   0.8278 -> 0.852 (+0.0242)
format:   0.992 -> 1.000
invalid:  0.008 -> 0.000
```

GRPO-G4 medium test：

```text
accuracy = 0.999
format_pass_rate = 1.000
average_reward = 1.199
invalid_answer_rate = 0.000
average_kl = 0.001456
```

GRPO-G4 easy test：

```text
accuracy = 0.891
format_pass_rate = 1.000
average_reward = 1.091
invalid_answer_rate = 0.000
average_kl = 0.013125
```

分析：

- GRPO-G4 相比 SFT-only 在 hard 上有稳定小幅提升。
- GRPO-G4 在 easy 上提升非常明显：`0.700 -> 0.891`。
- GRPO-G4 在 medium 上接近满分，没有灾难性遗忘。
- 格式稳定性提升：format 全部为 1.0，invalid 为 0。
- KL 很低，说明 policy 没有严重偏离 reference。

### 6.3 SFT-continued

SFT-continued hard test：

```text
accuracy = 0.659
format_pass_rate = 1.000
average_reward = 0.859
invalid_answer_rate = 0.000
```

SFT-continued medium test：

```text
accuracy = 0.974
format_pass_rate = 1.000
average_reward = 1.174
invalid_answer_rate = 0.000
```

SFT-continued easy test：

```text
accuracy = 0.871
format_pass_rate = 0.989
average_reward = 1.0666
invalid_answer_rate = 0.011
```

分析：

- SFT-continued 在 hard 上略优于 GRPO-G4：`0.659 vs 0.652`。
- 这说明在有 gold answer 的 synthetic hard 上，继续 SFT 是很强 baseline。
- SFT-continued 在 easy 上低于 GRPO-G4，并出现 1.1% invalid answers。
- GRPO-G4 在格式稳定性上优于 SFT-continued。

### 6.4 GRPO-G8-250

GRPO-G8-250 hard test：

```text
accuracy = 0.651
format_pass_rate = 1.000
average_reward = 0.851
invalid_answer_rate = 0.000
average_kl = 0.025154
```

GRPO-G8-250 medium test：

```text
accuracy = 0.997
format_pass_rate = 1.000
average_reward = 1.197
invalid_answer_rate = 0.000
average_kl = 0.001567
```

GRPO-G8-250 easy test：

```text
accuracy = 0.864
format_pass_rate = 1.000
average_reward = 1.064
invalid_answer_rate = 0.000
average_kl = 0.025303
```

分析：

- G8-250 没有超过 G4-1000。
- 在 hard 上，G8-250 几乎等于 G4：`0.651 vs 0.652`。
- 在 easy 上，G8-250 低于 G4：`0.864 vs 0.891`。
- 理论上 G8 应该提供更稳定 advantage，但实际没有更好。

可能原因：

1. G8-250 虽然 sample budget 与 G4-1000 相同，但 optimizer updates 更少。
2. G4 进行了 1000 次参数更新，G8-250 只有 250 次。
3. Larger group 降低了每步 advantage 噪声，但减少 update 次数可能抵消收益。
4. 当前任务中 G4 的 group signal 已经足够，继续增大 G 没有带来新信息。
5. GRPO 的主要瓶颈不是 group size，而是 reward 稀疏和 SFT baseline 太强。

### 6.5 GRPO-noKL

GRPO-noKL 参数与 GRPO-G4 完全一致，只将：

```text
kl_coef = 0.02 -> 0.0
```

其余保持不变：

```text
stage = hard
batch_size = 16
group_size = 4
max_iters = 1000
learning_rate = 1e-5
max_new_tokens = 32
```

GRPO-noKL hard test：

```text
accuracy = 0.652
format_pass_rate = 1.000
average_reward = 0.852
invalid_answer_rate = 0.000
average_kl = 0.068690
```

GRPO-noKL medium test：

```text
accuracy = 0.999
format_pass_rate = 1.000
average_reward = 1.199
invalid_answer_rate = 0.000
average_kl = 0.001462
```

GRPO-noKL easy test：

```text
accuracy = 0.903
format_pass_rate = 1.000
average_reward = 1.103
invalid_answer_rate = 0.000
average_kl = 0.018381
```

对比 GRPO-G4：

| Method | Easy Acc | Medium Acc | Hard Acc | Easy KL | Medium KL | Hard KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRPO-G4, `kl_coef=0.02` | 0.891 | 0.999 | 0.652 | 0.013125 | 0.001456 | 0.022831 |
| GRPO-noKL, `kl_coef=0.0` | 0.903 | 0.999 | 0.652 | 0.018381 | 0.001462 | 0.068690 |

分析：

- no-KL 在 easy 上进一步提升：`0.891 -> 0.903`。
- no-KL 在 medium/hard 上与 GRPO-G4 基本持平。
- no-KL 没有造成 format 崩坏，三档任务均保持 100% format pass rate 与 0 invalid。
- hard KL 从 `0.022831` 增至 `0.068690`，约为 3 倍。
- 这说明在当前 synthetic arithmetic 上，KL penalty 主要起限制 policy drift 的作用；去掉 KL 可以让 policy 更自由移动，并在 easy split 上带来额外收益，但没有转化为 hard split 的提升。
- 当前任务 reward 规则较强、输出格式较短，因此短程 no-KL 未出现明显格式崩坏；但从稳定性和泛化风险看，保留 KL 仍更稳健。

### 6.6 总表

| Method | Easy Acc | Medium Acc | Hard Acc | Easy Reward | Medium Reward | Hard Reward | Format / Invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SFT-only | 0.700 | 0.971 | 0.631 | 0.9000 | 1.1710 | 0.8278 | invalid up to 0.008 |
| SFT-continued | 0.871 | 0.974 | 0.659 | 1.0666 | 1.1740 | 0.8590 | easy invalid 0.011 |
| GRPO-G4 | 0.891 | 0.999 | 0.652 | 1.0910 | 1.1990 | 0.8520 | invalid 0.000 |
| GRPO-G8-250 | 0.864 | 0.997 | 0.651 | 1.0640 | 1.1970 | 0.8510 | invalid 0.000 |
| GRPO-noKL | 0.903 | 0.999 | 0.652 | 1.1030 | 1.1990 | 0.8520 | invalid 0.000 |

核心结论：

- GRPO-noKL 在 easy 上 accuracy 最高，但 hard KL 明显更高。
- GRPO-G4 是当前更稳健的 GRPO 配置，accuracy 提升与 KL 约束更平衡。
- GRPO-G4/GRPO-noKL 在 medium 上均接近满分。
- GRPO-G4 在 hard 上提升 SFT-only，但略低于 SFT-continued。
- G8-250 未超过 G4，说明更大 group size 不一定自动转化为更好 test accuracy。
- 所有 GRPO 变体的格式稳定性都很好，synthetic split 上 invalid 为 0。

## 7. 问题复盘与改进过程

### 7.1 问题一：SFT 太强，GRPO 初始 reward 饱和

现象：

SFT 在 medium 上已经达到：

```text
accuracy = 0.971
format_pass_rate = 1.000
```

GRPO smoke 在 medium batch 中出现：

```text
reward = 1.2000
acc = 1.0000
policy_loss = -0.0000
```

原因：

GRPO 依赖组内 reward 差异。如果所有 response 都正确：

```text
reward = [1.2, 1.2, 1.2, 1.2]
advantage ≈ [0, 0, 0, 0]
policy gradient ≈ 0
```

改进：

将 GRPO 训练 stage 从 `medium` 改为 `hard`，加入乘法任务。

改进效果：

SFT hard test：

```text
accuracy = 0.631
```

这说明 hard 上模型“会格式但答案不稳定”，正好适合 GRPO。

结论：

这是一次合理的实验难度重设。GRPO 需要既不是全对也不是全错的区域，才能产生有效 relative advantage。

### 7.2 问题二：GRPO 提升 SFT-only，但没有超过 SFT-continued

现象：

```text
SFT-only hard       acc = 0.631
GRPO-G4 hard        acc = 0.652
SFT-continued hard  acc = 0.659
```

分析：

synthetic hard 有精确 gold answer，因此 SFT-continued 得到的是 dense token-level supervision：

```text
<answer> 2 9 6 </answer>
```

每个 token 都有梯度。

GRPO 得到的是 sparse answer-level reward：

```text
correct answer: +1.2
wrong answer: low reward
```

它不知道错在哪一位，也不知道答案差多少。

因此在有标准答案的任务上，SFT-continued 是天然强 baseline。

结论：

当前结果并不说明 GRPO 失败，而是说明：

- GRPO 可以稳定改善 SFT-only。
- 在有 gold answer 的 synthetic arithmetic 上，SFT-continued 更样本高效。
- GRPO 的优势需要更偏 reward-only、弱监督或多解偏好任务来体现。

### 7.3 问题三：理论上 G8 应该更好，但实验没有明显提升

理论预期：

G8 更容易让 group 内出现正确和错误混合，从而产生有效 advantage。

实际结果：

```text
GRPO-G4 hard       acc = 0.652
GRPO-G8-250 hard   acc = 0.651
GRPO-G4 easy       acc = 0.891
GRPO-G8-250 easy   acc = 0.864
```

原因分析：

1. G8-250 是 sample-matched，不是 update-matched。

```text
G4: 1000 updates, 64 responses/update
G8: 250 updates, 256 responses/update
```

两者 rollout samples 都是 64k，但 optimizer updates 不同。

2. G8 降低了单步梯度方差，但减少了更新次数。

这可能导致：

```text
更稳定的每步估计
但总参数移动次数不足
```

3. 当前任务中 G4 signal 已经足够。

当 SFT hard accuracy 约 0.63 时，G4 产生有用组内差异的概率已经约 82.4%，G8 提升到约 97.5%，但这个理论增益没有成为最终 accuracy 增益。

4. 主要瓶颈可能不是 group size，而是 reward 稀疏。

即使 group 更大，reward 仍只判断最终答案是否正确，没有更细粒度的过程反馈。

未来改进：

- 跑 `G8-1000` 做 larger-budget 实验，但耗时约为 G4 的 4 倍。
- 保持 G8，同时增加 learning rate，例如 `2e-5`。
- 降低 KL，例如 `kl_coef=0.01`。
- 使用更丰富 reward，例如数字距离、格式多样惩罚、运算类型分项 reward。

### 7.4 问题四：GPU 利用率和显存利用率低

现象：

AutoDL panel 显示 GPU utilization 大约 20%，显存约 1-2GB。

原因：

模型很小：

```text
params = 3.17M
```

参数相关显存粗估：

```text
fp32 params + grads + Adam states ≈ 16 * P bytes
P = 3.17M
16P ≈ 50.7MB
```

显存主要来自 CUDA context、PyTorch cache、临时 tensors，而不是模型本身。

当前 rollout 实现是逐条生成：

```python
for ex in examples:
    for _ in range(group_size):
        model.generate(prompt)
```

这意味着即使 `batch_size * group_size` 变大，也只是增加 generate 调用次数。单次 GPU kernel 仍很小。

因此 G8 没有显著提高 GPU 利用率：

```text
G4: 64 responses/iter
G8: 256 responses/iter
```

理论计算量 4 倍，但当前实现接近逐条 generate，单次矩阵没有变大，Python loop 和 kernel launch overhead 仍然明显。

改进方向：

实现 batched rollout：

```text
当前：64 或 256 次 batch=1 generate
改进：1 次 batch=(batch_size * group_size) generate
```

将输入从：

```text
[1, T, C]
```

变成：

```text
[B * G, T, C]
```

这才会真正提高 GPU 饱和度。

### 7.5 问题五：Public benchmark 无法直接评估

原计划：

- MAWPS
- SVAMP
- GSM8K

实际问题：

1. Hugging Face 网络不稳定。

最初出现：

```text
RuntimeError: Cannot send a request, as the client has been closed.
```

改进：

设置 HF mirror：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
```

小下载测试成功。

2. block_size 与 public benchmark prompt 不匹配。

当前 checkpoint：

```text
block_size = 256
max_new_tokens = 32
```

因此 prompt 最大长度：

```text
256 - 32 = 224 char tokens
```

MAWPS 在 char-level tokenizer 下 prompt 超过限制，导致：

```text
ValueError: no examples fit inside the checkpoint block_size after reserving max_new_tokens
```

即使将 `max_new_tokens` 降到 8，仍无法匹配。

复盘：

一开始设计 v3 时重点关注 synthetic arithmetic，没有充分提前评估 public word-problem benchmark 的 prompt 长度与 char-level `block_size=256` 的匹配问题。

改进方向：

- v3.1 训练 `block_size=512` 或 `768` checkpoint。
- 使用 BPE tokenizer 版本，降低英文 prompt token 长度。
- 增加短公开 benchmark 或自建短 word-problem benchmark。
- 在训练前增加 benchmark length profiling 脚本：

```text
统计 prompt_len 分布、p50/p90/p95/max
根据 max_new_tokens 选择 block_size
```

## 8. 当前实验结论

### 8.1 正向结论

1. mini-GRPO pipeline 跑通。

实现了：

- group rollout
- rule-based reward
- relative advantage
- KL penalty
- no critic policy optimization
- SFT/SFT-continued/GRPO/G8 评估

2. GRPO-G4 提升 SFT-only。

hard：

```text
0.631 -> 0.652
```

easy：

```text
0.700 -> 0.891
```

3. GRPO 保持格式稳定。

GRPO-G4 在 easy/medium/hard 上：

```text
format_pass_rate = 1.000
invalid_answer_rate = 0.000
```

4. GRPO 没有灾难性遗忘 medium。

medium：

```text
GRPO-G4 accuracy = 0.999
```

5. 实验包含强 baseline。

不仅比较 SFT-only，也比较了 SFT-continued。

### 8.2 未达预期的地方

1. GRPO-G4 hard 没超过 SFT-continued。

```text
GRPO-G4: 0.652
SFT-continued: 0.659
```

2. G8-250 没超过 G4。

```text
G8 hard: 0.651
G4 hard: 0.652
```

3. Public benchmark 没能在当前 checkpoint 上直接完成。

原因：

- char-level prompt 太长。
- block_size=256 设计不足。
- 前期没有做 benchmark length profiling。

4. GPU 利用率低。

原因：

- 小模型。
- 逐条 rollout。
- Python autoregressive loop。
- batch/group 增大没有变成真正 batched generation。

### 8.3 最重要的阶段性判断

当前 v3 可以作为：

```text
可运行、可分析、带强 baseline 和消融的 mini-GRPO 对齐实验。
```

但不能夸大为：

```text
GRPO 显著超过 SFT。
```

更准确的结论：

```text
在 synthetic arithmetic 上，mini-GRPO 相比 SFT-only 带来稳定小幅提升，并改善格式稳定性；但在有精确 gold answer 的任务上，继续 SFT 仍是强 baseline。G8 group-size 消融显示，增大 group size 在 same rollout budget 下没有自动转化为更高 test accuracy，主要受 optimizer update 次数、reward 稀疏性和当前 rollout 实现限制影响。
```

## 9. 后续改进方向

### 9.1 实现 Batched Rollout

当前最大工程瓶颈。

目标：

```text
prompts = repeat_interleave(prompts, group_size)
generate(prompts_batch)
```

预期：

- 提升 GPU utilization。
- 减少 Python loop overhead。
- 让增大 batch/group 真正转化为吞吐提升。

### 9.2 增大 Context 以支持 Public Benchmark

训练新 checkpoint：

```text
block_size = 512 或 768
```

或者改为 BPE tokenizer。

在训练前先做：

```text
MAWPS/SVAMP/GSM8K prompt length profiling
```

避免再次出现 public benchmark 全部被过滤。

### 9.3 Reward Shaping

当前 reward 太稀疏。可加入：

- 数字距离 reward，例如 `-abs(pred - gold)` 的归一化版本。
- 运算类型分项 reward。
- 多 answer tag 惩罚。
- 过短/过长 response 更细化惩罚。

注意：reward shaping 要避免泄漏太多 SFT-like 标准答案信号，否则会弱化 GRPO 实验意义。

### 9.4 Hyperparameter Ablation

已完成：

| 实验 | 参数 | 结论 |
| --- | --- | --- |
| no-KL | `kl_coef=0.0` | easy accuracy 从 GRPO-G4 的 89.1% 提升至 90.3%，medium/hard 与 GRPO-G4 持平，但 hard KL 从 0.0228 增至 0.0687，说明 KL 主要限制 policy drift。 |

建议后续补：

| 实验 | 参数 | 目的 |
| --- | --- | --- |
| lower KL | `kl_coef=0.01` | 观察是否提升 accuracy。 |
| higher LR | `learning_rate=2e-5` | 观察 policy 是否移动更多。 |
| higher temp | `temperature=1.0` | 增强 exploration。 |
| G8-1000 | `group_size=8`, `max_iters=1000` | larger-budget 最佳模型实验。 |

### 9.5 更适合 RL 的任务

Synthetic hard 适合验证 pipeline，但不一定适合证明 RL 超过 SFT。

后续可以考虑：

- 只有 final answer 可验证、没有标准解法 token 的任务。
- 多步推理但只 reward final answer。
- 程序执行类任务。
- 格式/约束遵循任务。
- 偏好排序任务。

## 10. 简历表述建议

当前简历中 nanoGPT 项目可补充对齐训练内容。

建议放在第三条：

```text
对齐训练：在 nanoGPT 上实现 mini-GRPO 强化学习对齐流程，包含 group rollout、rule-based reward、relative advantage、KL penalty 和无 critic 策略优化；构建 SFT-only、SFT-continued、GRPO-G4、GRPO-G8-250、GRPO-noKL 等对照与消融实验，在 synthetic easy/medium/hard 三档可自动判分任务上评估 accuracy、reward、format pass rate、invalid answer rate 和 KL。GRPO-G4 相比 SFT-only 将 accuracy 从 70.0%/97.1%/63.1% 提升至 89.1%/99.9%/65.2%，并保持 100% format pass rate 与 0 invalid response；no-KL 消融进一步将 easy accuracy 提升至 90.3%，但 hard KL 从 0.0228 增至 0.0687，说明 KL 主要限制 policy drift。
```

如果简历篇幅有限，可压缩为：

```text
对齐训练：实现 mini-GRPO 对齐流程（group rollout、rule-based reward、relative advantage、KL penalty、no critic），构建 SFT-only / SFT-continued / GRPO-G4 / GRPO-G8-250 / GRPO-noKL 消融；GRPO-G4 相比 SFT-only 将 easy/medium/hard accuracy 从 70.0%/97.1%/63.1% 提升至 89.1%/99.9%/65.2%，保持 100% format pass rate 与 0 invalid response；no-KL 将 easy accuracy 进一步提升至 90.3%，但 hard KL 约增至 3 倍。
```

更稳健、不夸大的版本：

```text
对齐训练：在 nanoGPT 中实现 mini-GRPO 强化学习对齐实验，覆盖 SFT warmup、group rollout、rule-based reward、relative advantage、KL penalty 与无 critic policy update；通过 SFT-only、SFT-continued、G4/G8 group-size 消融和 no-KL 消融评估 reward、accuracy、format pass rate 与 KL，验证 GRPO 可稳定提升 SFT-only 并改善格式稳定性，同时分析其在有 gold answer 任务上未超过继续 SFT 的原因。
```

## 11. 面试复盘话术

如果被问“GRPO 为什么没明显超过 SFT-continued”，可以回答：

```text
因为当前 synthetic arithmetic 有精确 gold answer，继续 SFT 能获得 dense token-level supervision，而 GRPO 只有 answer-level sparse reward。实验中 GRPO 相比 SFT-only 有提升，并且格式更稳定，但 SFT-continued 在 hard split 上仍略强。这说明这个任务更 supervised-learning friendly，也促使我加入 SFT-continued 作为强 baseline，而不是只和 SFT-only 对比。
```

如果被问“为什么 G8 没比 G4 好”，可以回答：

```text
理论上 group size 增大能提高同组中正确/错误样本同时出现的概率，从而改善 relative advantage 估计。但我的 G8 实验是 same rollout budget：G4 是 1000 updates * 64 samples，G8 是 250 updates * 256 samples。虽然样本数相同，但 G8 的 optimizer update 次数更少；同时当前 reward 很稀疏，G4 已经能产生足够组内差异，所以 G8 没有转化为更高 test accuracy。
```

如果被问“KL 消融说明了什么”，可以回答：

```text
no-KL 消融显示，去掉 KL 后 easy accuracy 从 GRPO-G4 的 89.1% 提升到 90.3%，medium/hard 与 GRPO-G4 基本持平，且短程训练中没有出现 format 崩坏；但 hard split 上 average KL 从 0.0228 增至 0.0687，约为 3 倍。这说明在当前 synthetic arithmetic 任务上，KL penalty 不是 accuracy 的主要瓶颈，主要作用是限制 policy drift。no-KL 可以让 policy 更自由移动，但稳定性和泛化风险更高。
```

如果被问“GPU 利用率为什么低”，可以回答：

```text
瓶颈不在模型 FLOPs，而在 rollout 实现。模型只有 3.17M 参数，且当前 group rollout 是逐 prompt、逐 sample 调用 autoregressive generate，导致 Python loop 和 kernel launch overhead 占比很高。增大 batch/group 增加了总 samples，但没有把单次矩阵变大。后续应实现 batched rollout，把 batch_size * group_size 个 prompts 合并生成，才能真正提高 GPU utilization。
```

如果被问“公开 benchmark 为什么没跑通”，可以回答：

```text
v3 初版使用 char-level tokenizer 和 block_size=256，synthetic prompt 很短所以足够。但 MAWPS/SVAMP/GSM8K 这类英文 word problem 在 char-level 下 prompt 很长，预留 max_new_tokens 后没有样本能放进 context。这暴露出前期没有做 benchmark length profiling。后续 v3.1 会训练 block_size=512/768 或 BPE tokenizer 版本，并在训练前统计 public benchmark prompt length 分布。
```

## 12. 本阶段最终判断

v3 达成了以下目标：

- 实现 mini-GRPO 代码闭环。
- 完成 SFT、GRPO、SFT-continued、G8 group-size 消融和 no-KL 消融。
- 得到可解释的正向提升。
- 识别并分析了未超过 SFT-continued 的原因。
- 发现并复盘了 GPU 利用率和 public benchmark context mismatch 问题。

当前最准确的项目定位：

```text
这是一个教学清晰、实验完整、边界诚实的 mini-GRPO 对齐系统，而不是大规模推理模型复现。
```
