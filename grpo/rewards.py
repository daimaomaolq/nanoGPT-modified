import re
from dataclasses import dataclass
from typing import Optional


ANSWER_RE = re.compile(r"<answer>\s*([-+]?\d+)\s*</answer>")


@dataclass(frozen=True)
class RewardResult:
    reward: float
    correct: bool
    format_ok: bool
    extracted_answer: Optional[str]
    invalid: bool
    length: int


def extract_answer(text: str) -> Optional[str]:
    match = ANSWER_RE.search(text)
    if match is None:
        return None
    return match.group(1)


def score_response(response: str, gold_answer: str, max_response_chars: int = 80) -> RewardResult:
    extracted = extract_answer(response)
    format_ok = extracted is not None
    correct = extracted == str(gold_answer)
    invalid = not format_ok

    reward = 0.0
    if format_ok:
        reward += 0.2
    if correct:
        reward += 1.0
    if len(response) > max_response_chars:
        reward -= 0.2
    if invalid:
        reward -= 0.2

    return RewardResult(
        reward=reward,
        correct=correct,
        format_ok=format_ok,
        extracted_answer=extracted,
        invalid=invalid,
        length=len(response),
    )
