import random
import re
import string
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


@dataclass(frozen=True)
class ArithmeticExample:
    prompt: str
    response: str
    answer: str


class CharTokenizer:
    """Task-local character tokenizer for small arithmetic alignment runs."""

    def __init__(self, chars: Iterable[str] = None):
        if chars is None:
            chars = ["\n"] + list(string.printable[:95])
        deduped = []
        seen = set()
        for ch in chars:
            if ch not in seen:
                deduped.append(ch)
                seen.add(ch)
        self.tokens = SPECIAL_TOKENS + deduped
        self.stoi = {ch: i for i, ch in enumerate(self.tokens)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.pad_id = self.stoi[PAD_TOKEN]
        self.bos_id = self.stoi[BOS_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        ids = [self.stoi.get(ch, self.unk_id) for ch in text]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True, stop_at_eos: bool = True) -> str:
        out = []
        for idx in ids:
            idx = int(idx)
            if stop_at_eos and idx == self.eos_id:
                break
            tok = self.itos.get(idx, UNK_TOKEN)
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        return "".join(out)

    def state_dict(self) -> dict:
        return {"tokens": self.tokens}

    @classmethod
    def from_state_dict(cls, state: dict) -> "CharTokenizer":
        obj = cls(chars=[])
        obj.tokens = list(state["tokens"])
        obj.stoi = {ch: i for i, ch in enumerate(obj.tokens)}
        obj.itos = {i: ch for ch, i in obj.stoi.items()}
        obj.pad_id = obj.stoi[PAD_TOKEN]
        obj.bos_id = obj.stoi[BOS_TOKEN]
        obj.eos_id = obj.stoi[EOS_TOKEN]
        obj.unk_id = obj.stoi[UNK_TOKEN]
        return obj


def solve(a: int, op: str, b: int) -> int:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    raise ValueError(f"unknown op: {op}")


def make_example(a: int, op: str, b: int) -> ArithmeticExample:
    ans = str(solve(a, op, b))
    prompt = f"Question: What is {a} {op} {b}?\nAnswer: "
    response = f"<answer>{ans}</answer>"
    return ArithmeticExample(prompt=prompt, response=response, answer=ans)


def generate_examples(num_examples: int, split: str = "train", stage: str = "medium", seed: int = 1337) -> List[ArithmeticExample]:
    split_offsets = {"train": 0, "val": 100_000, "test": 200_000}
    rng = random.Random(seed + split_offsets.get(split, 300_000))
    if stage == "easy":
        ops = ["+", "-"]
        max_value = 9
    elif stage == "medium":
        ops = ["+", "-"]
        max_value = 99
    elif stage == "hard":
        ops = ["+", "-", "*"]
        max_value = 99
    else:
        raise ValueError(f"unknown stage: {stage}")

    examples = []
    for _ in range(num_examples):
        a = rng.randint(0, max_value)
        b = rng.randint(0, max_value)
        op = rng.choice(ops)
        examples.append(make_example(a, op, b))
    return examples


def _extract_number(text) -> str:
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", str(text))
    if not matches:
        return ""
    value = matches[-1]
    if value.endswith(".0"):
        value = value[:-2]
    return value


def _field(row: dict, names: Sequence[str], default: str = "") -> str:
    lowered = {str(k).lower(): k for k in row.keys()}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None:
            return str(row[key])
    return default


def load_public_examples(benchmark: str, split: str = "test", num_examples: int = 1000, seed: int = 1337) -> List[ArithmeticExample]:
    """Load public arithmetic word-problem benchmarks with lazy dependencies.

    Supported benchmark names are `gsm8k`, `mawps`, and `svamp`. This helper is
    intentionally flexible about field names because small arithmetic datasets on
    Hugging Face do not all use the same schema.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("public benchmark loading requires `datasets`") from exc

    benchmark = benchmark.lower()
    if benchmark == "gsm8k":
        dataset_name = ("openai/gsm8k", "main")
    elif benchmark == "mawps":
        dataset_name = ("MU-NLPC/Calc-mawps", None)
    elif benchmark == "svamp":
        dataset_name = ("MU-NLPC/Calc-svamp", None)
    else:
        raise ValueError(f"unknown benchmark: {benchmark}")

    name, config = dataset_name
    try:
        dataset = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    except ValueError:
        dataset_dict = load_dataset(name, config) if config else load_dataset(name)
        if split in dataset_dict:
            dataset = dataset_dict[split]
        elif "test" in dataset_dict:
            dataset = dataset_dict["test"]
        elif "validation" in dataset_dict:
            dataset = dataset_dict["validation"]
        else:
            first_split = next(iter(dataset_dict.keys()))
            dataset = dataset_dict[first_split]

    rows = list(dataset)
    rng = random.Random(seed)
    rng.shuffle(rows)
    examples = []
    for row in rows[:num_examples]:
        question = _field(row, ["question", "body", "problem", "text", "input"])
        if benchmark == "svamp":
            body = _field(row, ["body"])
            q = _field(row, ["question"])
            if body and q:
                question = f"{body} {q}"
        answer_text = _field(row, ["answer", "target", "output", "label"])
        if benchmark == "gsm8k" and "####" in answer_text:
            answer_text = answer_text.split("####")[-1]
        answer = _extract_number(answer_text)
        if not question or not answer:
            continue
        prompt = f"Question: {question}\nAnswer: "
        response = f"<answer>{answer}</answer>"
        examples.append(ArithmeticExample(prompt=prompt, response=response, answer=answer))
    return examples


def build_sft_batch(examples: Sequence[ArithmeticExample], tokenizer: CharTokenizer, block_size: int, device: str):
    xs, ys = [], []
    for ex in examples:
        prompt_ids = tokenizer.encode(ex.prompt)
        response_ids = tokenizer.encode(ex.response, add_eos=True)
        full = prompt_ids + response_ids
        if len(full) > block_size + 1:
            full = full[:block_size + 1]
        x = full[:-1]
        y = full[1:]

        # Mask prompt positions. The last prompt token predicts the first response token.
        supervised_start = max(len(prompt_ids) - 1, 0)
        y = [-1 if i < supervised_start else tok for i, tok in enumerate(y)]

        pad_len = block_size - len(x)
        if pad_len > 0:
            x = x + [tokenizer.pad_id] * pad_len
            y = y + [-1] * pad_len
        xs.append(torch.tensor(x, dtype=torch.long))
        ys.append(torch.tensor(y, dtype=torch.long))

    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def build_prompt_tensor(prompt: str, tokenizer: CharTokenizer, device: str):
    ids = tokenizer.encode(prompt)
    return torch.tensor(ids, dtype=torch.long, device=device)[None, :]


def pad_sequences(sequences: Sequence[Sequence[int]], pad_id: int, device: str):
    max_len = max(len(seq) for seq in sequences)
    out = torch.full((len(sequences), max_len), pad_id, dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        out[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return out


def sample_examples(pool: Sequence[ArithmeticExample], batch_size: int, rng: random.Random) -> List[ArithmeticExample]:
    return [pool[rng.randrange(len(pool))] for _ in range(batch_size)]
