from __future__ import annotations

from collections.abc import Callable, Sequence
import json
from pathlib import Path
from typing import Any

from .prompts import COT_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE
from .rewards import answer_tag_reward_fn, majority_vote_tagged_answers


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_VALIDATION_SIZE = 256


def _extract_gsm8k_target(example: dict[str, Any]) -> str:
    answer = str(example["answer"])
    if "####" not in answer:
        return answer.strip()
    return answer.rsplit("####", maxsplit=1)[-1].strip()


def _request_output_text(output: Any) -> str:
    if hasattr(output, "outputs") and output.outputs:
        return output.outputs[0].text
    if isinstance(output, dict):
        if "outputs" in output and output["outputs"]:
            first = output["outputs"][0]
            return first["text"] if isinstance(first, dict) else first.text
        if "text" in output:
            return output["text"]
    if isinstance(output, str):
        return output
    raise TypeError(f"Unsupported vLLM output type: {type(output)!r}")


def load_gsm8k_examples(split: str) -> list[dict[str, Any]]:
    """Load GSM8K examples from HuggingFace datasets."""
    from datasets import load_dataset

    dataset = load_dataset("gsm8k", "main")
    if split == "validation":
        return list(dataset["train"].select(range(DEFAULT_VALIDATION_SIZE)))
    if split not in dataset:
        raise ValueError(f"Unsupported GSM8K split: {split}")
    return list(dataset[split])


def build_prompts(examples: Sequence[dict[str, Any]], prompt_template: str) -> list[str]:
    """Format raw GSM8K examples into prompt strings."""
    return [prompt_template.format(question=example["question"]) for example in examples]


def evaluate_vllm(
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: Sequence[str],
    eval_sampling_params,
    ground_truths: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Generate model outputs, score them, and return serializable evaluation artifacts."""
    generations = vllm_model.generate(list(prompts), eval_sampling_params)
    responses = [_request_output_text(generation) for generation in generations]

    scored_examples: list[dict[str, Any]] = []
    aggregate_scores: dict[str, float] = {}

    if ground_truths is None:
        ground_truths = [""] * len(prompts)

    for prompt, response, ground_truth in zip(prompts, responses, ground_truths, strict=True):
        scores = reward_fn(response, ground_truth)
        scored_examples.append(
            {
                "prompt": prompt,
                "response": response,
                "ground_truth": ground_truth,
                "scores": scores,
            }
        )
        for key, value in scores.items():
            aggregate_scores[key] = aggregate_scores.get(key, 0.0) + float(value)

    num_examples = len(scored_examples)
    average_scores = {
        key: value / num_examples for key, value in aggregate_scores.items()
    } if num_examples else {}
    return {
        "num_examples": num_examples,
        "average_scores": average_scores,
        "examples": scored_examples,
    }


def write_evaluation_results(results: dict[str, Any], output_path: Path) -> None:
    """Serialize generations and scores for later analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def run_direct_baseline(output_path: Path) -> None:
    """Evaluate the direct-prediction GSM8K baseline from Section 3.1."""
    from vllm import LLM, SamplingParams

    examples = load_gsm8k_examples("test")
    prompts = build_prompts(examples, DIRECT_PROMPT_TEMPLATE)
    ground_truths = [_extract_gsm8k_target(example) for example in examples]

    llm = LLM(model=DEFAULT_MODEL_NAME)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        stop=["</answer>"],
    )
    results = evaluate_vllm(
        llm,
        answer_tag_reward_fn,
        prompts,
        sampling_params,
        ground_truths=ground_truths,
    )
    write_evaluation_results(results, output_path)


def run_cot_baseline(output_path: Path) -> None:
    """Evaluate the chain-of-thought baseline from Section 3.2."""
    from vllm import LLM, SamplingParams

    examples = load_gsm8k_examples("test")
    prompts = build_prompts(examples, str(COT_PROMPT_TEMPLATE))
    ground_truths = [_extract_gsm8k_target(example) for example in examples]

    llm = LLM(model=DEFAULT_MODEL_NAME)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop=["</answer>"],
    )
    results = evaluate_vllm(
        llm,
        answer_tag_reward_fn,
        prompts,
        sampling_params,
        ground_truths=ground_truths,
    )
    write_evaluation_results(results, output_path)


def run_self_consistency_baseline(output_path: Path, k: int = 5) -> None:
    """Evaluate the self-consistency baseline from Section 3.2."""
    from vllm import LLM, SamplingParams

    examples = load_gsm8k_examples("test")
    prompts = build_prompts(examples, str(COT_PROMPT_TEMPLATE))
    ground_truths = [_extract_gsm8k_target(example) for example in examples]

    llm = LLM(model=DEFAULT_MODEL_NAME)
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=512,
        n=k,
        stop=["</answer>"],
    )

    generations = llm.generate(prompts, sampling_params)
    scored_examples: list[dict[str, Any]] = []
    total_reward = 0.0

    for prompt, generation, ground_truth in zip(prompts, generations, ground_truths, strict=True):
        responses = [candidate.text for candidate in generation.outputs]
        voted_answer = majority_vote_tagged_answers(responses)
        final_response = f"<answer>{voted_answer}</answer>" if voted_answer is not None else ""
        scores = answer_tag_reward_fn(final_response, ground_truth)
        total_reward += scores["reward"]
        scored_examples.append(
            {
                "prompt": prompt,
                "responses": responses,
                "majority_answer": voted_answer,
                "ground_truth": ground_truth,
                "scores": scores,
            }
        )

    results = {
        "num_examples": len(scored_examples),
        "average_scores": {
            "reward": total_reward / len(scored_examples) if scored_examples else 0.0,
        },
        "examples": scored_examples,
    }
    write_evaluation_results(results, output_path)


def get_prompt_template(use_cot: bool) -> str:
    return COT_PROMPT_TEMPLATE if use_cot else DIRECT_PROMPT_TEMPLATE
