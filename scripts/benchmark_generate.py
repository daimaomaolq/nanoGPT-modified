import argparse
import os
import pickle
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from model import GPTConfig, GPT


def load_model(out_dir, device, compile_model):
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, _ in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    if compile_model:
        model = torch.compile(model)
    return model, checkpoint


def build_codec(checkpoint):
    load_meta = False
    if "config" in checkpoint and "dataset" in checkpoint["config"]:
        meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
        load_meta = os.path.exists(meta_path)
    if load_meta:
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[i] for i in l])
    else:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    return encode, decode


def synchronize(device_type):
    if device_type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark_one(model, x, args, ctx, device_type, use_kv_cache):
    # warmup
    with ctx:
        model.generate(x, min(args.warmup_tokens, args.max_new_tokens), temperature=args.temperature,
                       top_k=args.top_k, use_kv_cache=use_kv_cache)
    synchronize(device_type)

    times = []
    for _ in range(args.num_samples):
        synchronize(device_type)
        t0 = time.time()
        with ctx:
            model.generate(x, args.max_new_tokens, temperature=args.temperature,
                           top_k=args.top_k, use_kv_cache=use_kv_cache)
        synchronize(device_type)
        times.append(time.time() - t0)

    total_tokens = args.num_samples * args.max_new_tokens
    total_time = sum(times)
    return {
        "use_kv_cache": use_kv_cache,
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "total_time_s": total_time,
        "avg_time_s": total_time / len(times),
        "tokens_per_sec": total_tokens / total_time,
    }


def format_result(name, result):
    return (
        f"{name}: use_kv_cache={result['use_kv_cache']} "
        f"num_samples={result['num_samples']} "
        f"max_new_tokens={result['max_new_tokens']} "
        f"total_time_s={result['total_time_s']:.4f} "
        f"avg_time_s={result['avg_time_s']:.4f} "
        f"tokens_per_sec={result['tokens_per_sec']:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark nanoGPT generation with and without KV cache.")
    parser.add_argument("--out_dir", default="out")
    parser.add_argument("--start", default="\n")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--warmup_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--out_log", default=None)
    args = parser.parse_args()

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

    model, checkpoint = load_model(args.out_dir, args.device, args.compile)
    encode, _ = build_codec(checkpoint)

    start = args.start
    if start.startswith("FILE:"):
        with open(start[5:], "r", encoding="utf-8") as f:
            start = f.read()
    x = torch.tensor(encode(start), dtype=torch.long, device=args.device)[None, ...]

    no_cache = benchmark_one(model, x, args, ctx, device_type, use_kv_cache=False)
    kv_cache = benchmark_one(model, x, args, ctx, device_type, use_kv_cache=True)
    speedup = kv_cache["tokens_per_sec"] / no_cache["tokens_per_sec"]

    lines = [
        f"out_dir={args.out_dir}",
        f"device={args.device}",
        f"dtype={dtype}",
        f"prompt_tokens={x.size(1)}",
        format_result("no_cache", no_cache),
        format_result("kv_cache", kv_cache),
        f"speedup={speedup:.4f}x",
    ]
    output = "\n".join(lines)
    print(output)

    if args.out_log is not None:
        out_path = Path(args.out_log)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
