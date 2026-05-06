# mini-GRPO for nanoGPT

This directory implements the v3 `ModernKVCacheGRPO0.3` experiment.

It is a compact GRPO-style alignment pipeline for arithmetic tasks. It is meant
to demonstrate the mechanics of reinforcement learning alignment in nanoGPT,
not to reproduce large-scale reasoning models.

## Files

```text
data.py          Synthetic arithmetic data and a task-local character tokenizer.
rewards.py       Strict answer parsing and rule-based rewards.
policy.py        Response-token log probability and KL helper functions.
train_sft.py     Supervised fine-tuning warmup.
train_grpo.py    Group rollout and GRPO-style policy optimization.
evaluate.py      Evaluation for reward, accuracy, format, KL, and length.
```

## Workflow

1. Train an SFT checkpoint so the model learns the prompt/answer format.
2. Copy the SFT checkpoint as a frozen reference policy.
3. Run GRPO from the SFT checkpoint.
4. Evaluate SFT-only, SFT-continued, GRPO, and no-KL ablations.

## Baselines

The primary baseline is `SFT-only`, not the old language-modeling checkpoint.
The v3 task is a reward-optimized answer-format task, so the fair comparison is
between checkpoints that share the same model architecture and SFT starting
point.

Recommended comparisons:

```text
SFT-only
SFT-continued with similar extra steps
SFT + GRPO
SFT + GRPO without KL
```

## Output Format

Prompts ask for an arithmetic answer. Responses must contain exactly one strict
answer tag:

```text
<answer>41</answer>
```

The evaluator extracts the first answer tag and compares it to the gold answer.

## Quick CPU Smoke Test

```bash
python grpo/train_sft.py \
  --out_dir out-grpo-sft-smoke \
  --device cpu \
  --dtype float32 \
  --max_iters 5 \
  --eval_interval 5 \
  --batch_size 4 \
  --block_size 256 \
  --n_layer 2 \
  --n_head 2 \
  --n_embd 64 \
  --compile False
```

```bash
python grpo/evaluate.py \
  --checkpoint out-grpo-sft-smoke/ckpt.pt \
  --device cpu \
  --dtype float32 \
  --num_examples 16
```

## Server SFT Run

```bash
python grpo/train_sft.py \
  --out_dir out-grpo-sft \
  --device cuda \
  --dtype bfloat16 \
  --max_iters 3000 \
  --eval_interval 250 \
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

## Server GRPO Run

```bash
python grpo/train_grpo.py \
  --init_from out-grpo-sft/ckpt.pt \
  --reference_from out-grpo-sft/ckpt.pt \
  --out_dir out-grpo-v3 \
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

## Evaluation

```bash
python grpo/evaluate.py \
  --checkpoint out-grpo-v3/ckpt.pt \
  --reference_checkpoint out-grpo-sft/ckpt.pt \
  --device cuda \
  --dtype bfloat16 \
  --num_examples 1000 \
  --out_file out-grpo-v3/eval_test.json
```

Public benchmark example:

```bash
python grpo/evaluate.py \
  --checkpoint out-grpo-v3/ckpt.pt \
  --reference_checkpoint out-grpo-sft/ckpt.pt \
  --benchmark gsm8k \
  --split test \
  --device cuda \
  --dtype bfloat16 \
  --num_examples 1000 \
  --out_file out-grpo-v3/eval_gsm8k.json
```

The main metrics are:

- `accuracy`
- `format_pass_rate`
- `average_reward`
- `invalid_answer_rate`
- `average_kl`
- `average_response_length`

## Public Benchmarks

The first implementation is optimized for synthetic arithmetic and public
word-problem evaluation. Recommended public datasets:

- `MU-NLPC/Calc-mawps`
- `MU-NLPC/Calc-svamp`
- `openai/gsm8k`

Public benchmark loading is intentionally kept outside the smoke path so local
verification does not require network access.
