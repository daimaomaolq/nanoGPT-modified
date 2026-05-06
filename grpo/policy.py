import torch
from torch.nn import functional as F


def response_mask_from_prompt_lengths(sequences: torch.Tensor, prompt_lengths, pad_id: int):
    """Build a mask over target positions used by next-token logprobs."""
    targets = sequences[:, 1:]
    mask = targets.ne(pad_id)
    for i, prompt_len in enumerate(prompt_lengths):
        supervised_start = max(int(prompt_len) - 1, 0)
        if supervised_start > 0:
            mask[i, :supervised_start] = False
    return mask


def token_logprobs(model, sequences: torch.Tensor, response_mask: torch.Tensor):
    idx = sequences[:, :-1]
    targets = sequences[:, 1:]
    lm_targets = targets.masked_fill(~response_mask, -1)
    # Passing targets keeps nanoGPT on the full-sequence logits path. With
    # targets=None the model optimizes inference by returning only the last
    # position, which is not enough for response-token policy gradients.
    logits, _ = model(idx, lm_targets)
    log_probs = F.log_softmax(logits, dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    gathered = gathered * response_mask.to(gathered.dtype)
    lengths = response_mask.sum(dim=-1).clamp_min(1)
    return {
        "token_logprobs": gathered,
        "sum_logprobs": gathered.sum(dim=-1),
        "mean_logprobs": gathered.sum(dim=-1) / lengths,
        "lengths": lengths,
    }


def sampled_kl(current_token_logprobs: torch.Tensor, reference_token_logprobs: torch.Tensor, response_mask: torch.Tensor):
    mask = response_mask.to(current_token_logprobs.dtype)
    lengths = response_mask.sum(dim=-1).clamp_min(1).to(current_token_logprobs.dtype)
    log_ratio = reference_token_logprobs - current_token_logprobs
    # Non-negative sampled KL approximation used in several RLHF implementations:
    # exp(log r) - log r - 1, where r = pi_ref / pi_current.
    per_token = torch.exp(log_ratio) - log_ratio - 1.0
    per_seq = (per_token * mask).sum(dim=-1) / lengths
    return per_seq.mean(), per_seq
