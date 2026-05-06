import argparse
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import GPT, GPTConfig
from grpo.data import CharTokenizer, build_prompt_tensor, generate_examples, pad_sequences, sample_examples
from grpo.policy import response_mask_from_prompt_lengths, sampled_kl, token_logprobs
from grpo.rewards import score_response


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="mini-GRPO training for nanoGPT arithmetic alignment.")
    parser.add_argument("--init_from", required=True)
    parser.add_argument("--reference_from", required=True)
    parser.add_argument("--out_dir", default="out-grpo-v3")
    parser.add_argument("--stage", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--train_examples", type=int, default=20000)
    parser.add_argument("--val_examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--kl_coef", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=None, choices=[None, "float32", "bfloat16", "float16"])
    parser.add_argument("--compile", type=str2bool, default=False)
    return parser.parse_args()


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    tokenizer = CharTokenizer.from_state_dict(checkpoint["tokenizer"])
    model = GPT(GPTConfig(**checkpoint["model_args"]))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key, _ in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    return model, tokenizer, checkpoint


@torch.no_grad()
def rollout(model, examples, tokenizer, args, ctx):
    sequences = []
    prompt_lengths = []
    rewards = []
    rows = []
    model.eval()
    for ex in examples:
        prompt_ids = tokenizer.encode(ex.prompt)
        for _ in range(args.group_size):
            prompt = build_prompt_tensor(ex.prompt, tokenizer, args.device)
            with ctx:
                out = model.generate(
                    prompt,
                    args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    use_kv_cache=True,
                )
            ids = out[0].tolist()
            response = tokenizer.decode(ids[len(prompt_ids):])
            score = score_response(response, ex.answer)
            sequences.append(ids)
            prompt_lengths.append(len(prompt_ids))
            rewards.append(score.reward)
            rows.append((ex, response, score))
    model.train()
    return sequences, prompt_lengths, torch.tensor(rewards, dtype=torch.float32, device=args.device), rows


def filter_rollout_examples(examples, tokenizer, block_size: int, max_new_tokens: int):
    max_prompt_len = block_size - max_new_tokens
    return [ex for ex in examples if len(tokenizer.encode(ex.prompt)) <= max_prompt_len]


def group_advantages(rewards: torch.Tensor, batch_size: int, group_size: int):
    grouped = rewards.view(batch_size, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False)
    adv = (grouped - mean) / (std + 1e-6)
    return adv.view(-1), mean.mean().item(), std.mean().item()


def summarize_rows(rows):
    if not rows:
        return {}
    rewards = [row[2].reward for row in rows]
    correct = [row[2].correct for row in rows]
    fmt = [row[2].format_ok for row in rows]
    invalid = [row[2].invalid for row in rows]
    lengths = [row[2].length for row in rows]
    return {
        "avg_reward": sum(rewards) / len(rewards),
        "accuracy": sum(correct) / len(correct),
        "format_pass_rate": sum(fmt) / len(fmt),
        "invalid_answer_rate": sum(invalid) / len(invalid),
        "avg_response_length": sum(lengths) / len(lengths),
    }


def main():
    args = parse_args()
    if args.group_size < 2:
        raise ValueError("GRPO requires group_size >= 2 for within-group advantage normalization")
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = args.dtype
    if dtype is None:
        dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    model, tokenizer, checkpoint = load_model(args.init_from, args.device)
    reference, ref_tokenizer, _ = load_model(args.reference_from, args.device)
    if tokenizer.tokens != ref_tokenizer.tokens:
        raise ValueError("policy and reference tokenizers differ")
    model.to(args.device)
    reference.to(args.device)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    raw_model = model
    optimizer = raw_model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)
    if args.compile:
        model = torch.compile(raw_model)
    train_data = generate_examples(args.train_examples, split="train", stage=args.stage, seed=args.seed)
    train_data = filter_rollout_examples(train_data, tokenizer, model.config.block_size, args.max_new_tokens)
    if not train_data:
        raise ValueError(
            "no training examples fit inside block_size after reserving max_new_tokens; "
            "increase block_size or reduce max_new_tokens"
        )
    if len(train_data) < args.batch_size:
        raise ValueError("not enough fitting training examples for one GRPO batch")
    rng = random.Random(args.seed + 23)
    metrics_path = Path(args.out_dir, "metrics.jsonl")
    t0 = time.time()

    for iter_num in range(args.max_iters):
        batch = sample_examples(train_data, args.batch_size, rng)
        sequences, prompt_lengths, rewards, rows = rollout(model, batch, tokenizer, args, ctx)
        advantages, group_reward, group_reward_std = group_advantages(rewards, args.batch_size, args.group_size)
        padded = pad_sequences(sequences, tokenizer.pad_id, args.device)
        response_mask = response_mask_from_prompt_lengths(padded, prompt_lengths, tokenizer.pad_id)

        current = token_logprobs(model, padded, response_mask)
        with torch.no_grad():
            ref = token_logprobs(reference, padded, response_mask)
        kl_mean, _ = sampled_kl(current["token_logprobs"], ref["token_logprobs"], response_mask)
        policy_loss = -(advantages.detach() * current["sum_logprobs"]).mean()
        loss = policy_loss + args.kl_coef * kl_mean

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        optimizer.step()

        behavior = summarize_rows(rows)
        record = {
            "iter": iter_num,
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "kl": kl_mean.item(),
            "group_reward": group_reward,
            "group_reward_std": group_reward_std,
            **behavior,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if iter_num % 10 == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(
                f"iter {iter_num}: loss {loss.item():.4f}, policy {policy_loss.item():.4f}, "
                f"kl {kl_mean.item():.4f}, reward {behavior['avg_reward']:.4f}, "
                f"acc {behavior['accuracy']:.4f}, format {behavior['format_pass_rate']:.4f}, "
                f"time {dt * 1000:.2f}ms"
            )

        if iter_num % args.eval_interval == 0 or iter_num == args.max_iters - 1:
            save = {
                "model": raw_model.state_dict(),
                "model_args": checkpoint["model_args"],
                "tokenizer": tokenizer.state_dict(),
                "iter_num": iter_num,
                "args": vars(args),
            }
            torch.save(save, os.path.join(args.out_dir, "ckpt.pt"))
            Path(args.out_dir, "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
