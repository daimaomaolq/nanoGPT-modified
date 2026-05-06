import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import GPT, GPTConfig
from grpo.data import CharTokenizer, build_prompt_tensor, generate_examples, load_public_examples, pad_sequences
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
    parser = argparse.ArgumentParser(description="Evaluate mini-GRPO arithmetic checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference_checkpoint", default=None)
    parser.add_argument("--benchmark", default="synthetic", choices=["synthetic", "gsm8k", "mawps", "svamp"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--stage", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--num_examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=None, choices=[None, "float32", "bfloat16", "float16"])
    parser.add_argument("--compile", type=str2bool, default=False)
    parser.add_argument("--out_file", default=None)
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
    model.eval()
    model.to(device)
    return model, tokenizer


@torch.no_grad()
def generate_one(model, tokenizer, prompt, args, ctx):
    x = build_prompt_tensor(prompt, tokenizer, args.device)
    with ctx:
        if args.temperature == 0.0:
            out = greedy_generate(model, x, args.max_new_tokens, tokenizer.eos_id)
        else:
            out = model.generate(
                x,
                args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                use_kv_cache=True,
            )
    ids = out[0].tolist()
    prompt_len = x.size(1)
    return ids, prompt_len, tokenizer.decode(ids[prompt_len:])


@torch.no_grad()
def greedy_generate(model, idx, max_new_tokens, eos_id):
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
        if int(idx_next.item()) == eos_id:
            break
    return idx


def main():
    args = parse_args()
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

    model, tokenizer = load_model(args.checkpoint, args.device)
    reference = None
    if args.reference_checkpoint:
        reference, ref_tokenizer = load_model(args.reference_checkpoint, args.device)
        if tokenizer.tokens != ref_tokenizer.tokens:
            raise ValueError("policy and reference tokenizers differ")

    if args.benchmark == "synthetic":
        examples = generate_examples(args.num_examples, split=args.split, stage=args.stage, seed=args.seed)
    else:
        examples = load_public_examples(args.benchmark, split=args.split, num_examples=args.num_examples, seed=args.seed)
    requested_examples = len(examples)
    max_prompt_len = model.config.block_size - args.max_new_tokens
    examples = [ex for ex in examples if len(tokenizer.encode(ex.prompt)) <= max_prompt_len]
    if not examples:
        raise ValueError(
            "no examples fit inside the checkpoint block_size after reserving max_new_tokens; "
            "use a larger block_size SFT checkpoint, reduce max_new_tokens, or evaluate a shorter benchmark"
        )
    if args.compile:
        model = torch.compile(model)
    rows = []
    sequences = []
    prompt_lengths = []
    for ex in examples:
        ids, prompt_len, response = generate_one(model, tokenizer, ex.prompt, args, ctx)
        score = score_response(response, ex.answer)
        rows.append({
            "prompt": ex.prompt,
            "gold_answer": ex.answer,
            "response": response,
            "reward": score.reward,
            "correct": score.correct,
            "format_ok": score.format_ok,
            "invalid": score.invalid,
            "length": score.length,
        })
        sequences.append(ids)
        prompt_lengths.append(prompt_len)

    rewards = [row["reward"] for row in rows]
    correct = [row["correct"] for row in rows]
    fmt = [row["format_ok"] for row in rows]
    invalid = [row["invalid"] for row in rows]
    lengths = [row["length"] for row in rows]
    result = {
        "checkpoint": args.checkpoint,
        "reference_checkpoint": args.reference_checkpoint,
        "benchmark": args.benchmark,
        "split": args.split,
        "stage": args.stage,
        "num_examples": len(rows),
        "requested_examples": requested_examples,
        "accuracy": sum(correct) / len(correct),
        "format_pass_rate": sum(fmt) / len(fmt),
        "average_reward": sum(rewards) / len(rewards),
        "invalid_answer_rate": sum(invalid) / len(invalid),
        "average_response_length": sum(lengths) / len(lengths),
    }

    if reference is not None and sequences:
        padded = pad_sequences(sequences, tokenizer.pad_id, args.device)
        mask = response_mask_from_prompt_lengths(padded, prompt_lengths, tokenizer.pad_id)
        current = token_logprobs(model, padded, mask)
        ref = token_logprobs(reference, padded, mask)
        kl_mean, _ = sampled_kl(current["token_logprobs"], ref["token_logprobs"], mask)
        result["average_kl"] = kl_mean.item()

    output = json.dumps(result, indent=2)
    print(output)
    if args.out_file:
        out_path = Path(args.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"summary": result, "examples": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
