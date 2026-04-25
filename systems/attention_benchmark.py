from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class AttentionBenchmarkConfig:
    head_dims: tuple[int, ...] = (16, 32, 64, 128)
    sequence_lengths: tuple[int, ...] = (64, 128, 256, 512, 1024)
    batch_size: int = 8
    forward_passes: int = 100
    backward_passes: int = 100
    compile_attention: bool = False


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark attention implementations.")
    parser.add_argument("--compile-attention", action="store_true")
    return parser


def iter_benchmark_shapes(config: AttentionBenchmarkConfig) -> Iterable[tuple[int, int]]:
    for head_dim in config.head_dims:
        for sequence_length in config.sequence_lengths:
            yield head_dim, sequence_length


def make_qkv(batch_size: int, sequence_length: int, head_dim: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Create random Q, K, and V tensors for the attention benchmark."""
    Q = torch.rand(batch_size, sequence_length, head_dim, device=device, requires_grad=True)
    K = torch.rand(batch_size, sequence_length, head_dim, device=device, requires_grad=True)
    V = torch.rand(batch_size, sequence_length, head_dim, device=device, requires_grad=True)
    return Q, K, V


def benchmark_attention_once(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    forward_passes: int,
    backward_passes: int,
    compile_attention: bool,
) -> dict[str, float]:
    """Time the forward and backward pass for a single attention configuration."""
    def attention_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        softmax = torch.nn.Softmax(dim=-1)
        return softmax(q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5)) @ v

    if compile_attention:
        attention_fn = torch.compile(attention_fn)

    forward_times = []
    backward_times = []

    for _ in range(forward_passes):
        q.grad = None
        k.grad = None
        v.grad = None
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        _ = attention_fn(q, k, v)
        end_event.record()
        torch.cuda.synchronize()
        forward_times.append(start_event.elapsed_time(end_event))

    for _ in range(backward_passes):
        q.grad = None
        k.grad = None
        v.grad = None
        output = attention_fn(q, k, v)
        loss = output.sum()
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        loss.backward()
        end_event.record()
        torch.cuda.synchronize()
        backward_times.append(start_event.elapsed_time(end_event))

    return {
        "forward_time_ms": sum(forward_times) / len(forward_times),
        "backward_time_ms": sum(backward_times) / len(backward_times),
    }



def benchmark_attention_grid(config: AttentionBenchmarkConfig) -> list[dict[str, float | int | str]]:
    """Run the attention benchmark over the Section 2.7 Cartesian product of scales."""
    results = []
    device = torch.device("cuda")
    for head_dim, sequence_length in iter_benchmark_shapes(config):
        q, k, v = make_qkv(config.batch_size, sequence_length, head_dim, device)
        result = benchmark_attention_once(
            q,
            k,
            v,
            forward_passes=config.forward_passes,
            backward_passes=config.backward_passes,
            compile_attention=config.compile_attention,
        )
        results.append({
            "head_dim": head_dim,
            "sequence_length": sequence_length,
            "compiled": str(config.compile_attention),
            **result,
        })
    return results


def main() -> None:
    args = build_argparser().parse_args()
    config = AttentionBenchmarkConfig(compile_attention=args.compile_attention)
    benchmark_attention_grid(config)


if __name__ == "__main__":
    main()
