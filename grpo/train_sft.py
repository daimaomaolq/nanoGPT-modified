import argparse
import json
import math
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
from grpo.data import CharTokenizer, build_sft_batch, generate_examples, sample_examples


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
    parser = argparse.ArgumentParser(description="SFT warmup for mini-GRPO arithmetic tasks.")
    parser.add_argument("--out_dir", default="out-grpo-sft")
    parser.add_argument("--init_from", default=None, help="Optional checkpoint path to continue SFT.")
    parser.add_argument("--stage", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--train_examples", type=int, default=20000)
    parser.add_argument("--val_examples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--max_iters", type=int, default=3000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", type=str2bool, default=False)
    parser.add_argument("--norm_type", default="rmsnorm", choices=["layernorm", "rmsnorm"])
    parser.add_argument("--mlp_type", default="swiglu", choices=["gelu", "swiglu"])
    parser.add_argument("--position_embedding_type", default="rope", choices=["learned", "rope"])
    parser.add_argument("--rope_base", type=float, default=10000.0)
    parser.add_argument("--swiglu_hidden_mult", type=float, default=8 / 3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=None, choices=[None, "float32", "bfloat16", "float16"])
    parser.add_argument("--compile", type=str2bool, default=False)
    return parser.parse_args()


def load_checkpoint(path, device):
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
def estimate_loss(model, examples, tokenizer, args, ctx, rng):
    model.eval()
    losses = []
    for _ in range(args.eval_iters):
        batch = sample_examples(examples, args.batch_size, rng)
        x, y = build_sft_batch(batch, tokenizer, args.block_size, args.device)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    args = parse_args()
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

    if args.init_from:
        model, tokenizer, checkpoint = load_checkpoint(args.init_from, args.device)
        model_args = checkpoint["model_args"]
        args.block_size = model_args["block_size"]
    else:
        tokenizer = CharTokenizer()
        model_args = dict(
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            block_size=args.block_size,
            bias=args.bias,
            vocab_size=tokenizer.vocab_size,
            dropout=args.dropout,
            norm_type=args.norm_type,
            mlp_type=args.mlp_type,
            position_embedding_type=args.position_embedding_type,
            rope_base=args.rope_base,
            swiglu_hidden_mult=args.swiglu_hidden_mult,
        )
        model = GPT(GPTConfig(**model_args))

    model.to(args.device)
    raw_model = model
    optimizer = raw_model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)
    if args.compile:
        model = torch.compile(raw_model)

    train_data = generate_examples(args.train_examples, split="train", stage=args.stage, seed=args.seed)
    val_data = generate_examples(args.val_examples, split="val", stage=args.stage, seed=args.seed)
    train_rng = random.Random(args.seed + 7)
    eval_rng = random.Random(args.seed + 11)

    best_val_loss = math.inf
    t0 = time.time()
    for iter_num in range(args.max_iters + 1):
        if iter_num % args.eval_interval == 0:
            val_loss = estimate_loss(model, val_data, tokenizer, args, ctx, eval_rng)
            print(f"step {iter_num}: val_sft_loss {val_loss:.4f}")
            if val_loss < best_val_loss or iter_num == args.max_iters:
                best_val_loss = val_loss
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "model_args": model_args,
                    "tokenizer": tokenizer.state_dict(),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "args": vars(args),
                }
                torch.save(checkpoint, os.path.join(args.out_dir, "ckpt.pt"))
                Path(args.out_dir, "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        if iter_num == args.max_iters:
            break

        batch = sample_examples(train_data, args.batch_size, train_rng)
        x, y = build_sft_batch(batch, tokenizer, args.block_size, args.device)
        with ctx:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        optimizer.step()

        if iter_num % 20 == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"iter {iter_num}: sft_loss {loss.item():.4f}, time {dt * 1000:.2f}ms")


if __name__ == "__main__":
    main()
