from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
import argparse
import json
from pathlib import Path
from typing import Any

from .prompts import COT_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE
from .rewards import answer_tag_reward_fn, majority_vote_tagged_answers


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_VALIDATION_SIZE = 256
DEFAULT_GPU_MEMORY_UTILIZATION = 0.65
DEFAULT_MAX_NUM_SEQS = 8


def load_vllm_model(model_name: str = DEFAULT_MODEL_NAME):
    """Load a vLLM model with conservative defaults for Colab GPUs."""
    from vllm import LLM

    return LLM(
        model=model_name,
        gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
        max_num_seqs=DEFAULT_MAX_NUM_SEQS,
        enforce_eager=True,
    )


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


def _score_category(scores: dict[str, float]) -> str:
    format_reward = float(scores.get("format_reward", 0.0))
    answer_reward = float(scores.get("answer_reward", 0.0))
    if format_reward == 1.0 and answer_reward == 1.0:
        return "correct_format_and_answer"
    if format_reward == 1.0 and answer_reward == 0.0:
        return "formatted_wrong_answer"
    if format_reward == 0.0 and answer_reward == 0.0:
        return "unformatted"
    return "other"


def summarize_evaluation_results(
    results: dict[str, Any],
    examples_per_category: int = 10,
) -> dict[str, Any]:
    """Count reward categories and keep a few examples for qualitative analysis."""
    counts: Counter[str] = Counter()
    examples_by_category: dict[str, list[dict[str, Any]]] = {}

    for example in results["examples"]:
        category = _score_category(example["scores"])
        counts[category] += 1
        category_examples = examples_by_category.setdefault(category, [])
        if len(category_examples) < examples_per_category:
            category_examples.append(example)

    return {
        "category_counts": dict(counts),
        "examples_by_category": examples_by_category,
    }


def run_direct_baseline(output_path: Path, split: str = "test") -> None:
    """Evaluate the direct-prediction GSM8K baseline from Section 3.1."""
    from vllm import SamplingParams

    examples = load_gsm8k_examples(split)
    prompts = build_prompts(examples, DIRECT_PROMPT_TEMPLATE)
    ground_truths = [_extract_gsm8k_target(example) for example in examples]

    llm = load_vllm_model()
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    results = evaluate_vllm(
        llm,
        answer_tag_reward_fn,
        prompts,
        sampling_params,
        ground_truths=ground_truths,
    )
    results["split"] = split
    results["category_summary"] = summarize_evaluation_results(results)
    write_evaluation_results(results, output_path)


def run_cot_baseline(output_path: Path) -> None:
    """Evaluate the chain-of-thought baseline from Section 3.2."""
    from vllm import SamplingParams

    examples = load_gsm8k_examples("test")
    prompts = build_prompts(examples, str(COT_PROMPT_TEMPLATE))
    ground_truths = [_extract_gsm8k_target(example) for example in examples]

    llm = load_vllm_model()
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
    from vllm import SamplingParams

    examples = load_gsm8k_examples("test")
    prompts = build_prompts(examples, str(COT_PROMPT_TEMPLATE))
    ground_truths = [_extract_gsm8k_target(example) for example in examples]

    llm = load_vllm_model()
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GSM8K vLLM evaluation baselines.")
    parser.add_argument(
        "--baseline",
        choices=["direct"],
        default="direct",
        help="Which baseline to run.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "validation"],
        default="test",
        help="GSM8K split to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/direct_baseline.json"),
        help="Path to write JSON results.",
    )
    args = parser.parse_args()

    if args.baseline == "direct":
        run_direct_baseline(args.output, split=args.split)


if __name__ == "__main__":
    main()
