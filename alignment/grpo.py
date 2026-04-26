from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict[str, Tensor]:
    """Tokenize prompt/output pairs and build a response mask over the labels."""
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have the same length")

    prompt_token_ids = [
        tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompt_strs
    ]
    output_token_ids = [
        tokenizer.encode(output, add_special_tokens=False) for output in output_strs
    ]
    full_sequences = [
        prompt_ids + output_ids
        for prompt_ids, output_ids in zip(prompt_token_ids, output_token_ids, strict=True)
    ]

    if not full_sequences:
        return {
            "input_ids": torch.empty((0, 0), dtype=torch.long),
            "labels": torch.empty((0, 0), dtype=torch.long),
            "response_mask": torch.empty((0, 0), dtype=torch.bool),
        }

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")

    max_len = max(len(sequence) - 1 for sequence in full_sequences)
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    response_mask: list[list[bool]] = []

    for prompt_ids, output_ids, sequence in zip(
        prompt_token_ids,
        output_token_ids,
        full_sequences,
        strict=True,
    ):
        sequence_len = len(sequence) - 1
        padding_len = max_len - sequence_len

        input_ids.append(sequence[:-1] + [pad_token_id] * padding_len)
        labels.append(sequence[1:] + [pad_token_id] * padding_len)
        response_mask.append(
            [False] * (len(prompt_ids) - 1)
            + [True] * len(output_ids)
            + [False] * padding_len
        )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype=torch.bool),
    }


def compute_entropy(logits: Tensor) -> Tensor:
    """Compute per-token entropies over the vocabulary dimension."""
    raise NotImplementedError


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Score conditional log-probabilities for a batch of prompt/response examples."""
    raise NotImplementedError


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> Tensor:
    """Sum over masked elements and normalize by the provided constant."""
    raise NotImplementedError


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Compute raw rewards and per-group normalized advantages for GRPO."""
    raise NotImplementedError


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the per-token GRPO-Clip loss."""
    raise NotImplementedError


def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    advantages: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Backpropagate a single GRPO microbatch loss."""
    raise NotImplementedError


def log_generations(
    prompts: Sequence[str],
    responses: Sequence[str],
    ground_truths: Sequence[str],
    reward_infos: Sequence[dict[str, float]],
    token_entropies: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Create serializable generation logs for debugging training runs."""
    raise NotImplementedError


def train_grpo(*args, **kwargs) -> dict[str, Any]:
    """Run the full GRPO training loop from Section 3.5."""
    raise NotImplementedError
