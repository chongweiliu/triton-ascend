# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Functional and smoke-performance validation for the AddRmsNorm tutorial."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import torch

try:
    import torch_npu  # noqa: F401
except Exception:  # pragma: no cover - depends on Ascend runtime.
    torch_npu = None

from add_rms_norm import add_rms_norm, add_rms_norm_reference


PUBLIC_BH_SHAPES = [
    (1, 3584),
    (1, 4096),
    (1, 5120),
    (1, 8192),
    (8, 3584),
    (8, 4096),
    (8, 5120),
    (8, 8192),
    (16, 3584),
    (16, 4096),
    (16, 5120),
    (16, 8192),
    (32, 3584),
    (32, 4096),
    (32, 5120),
    (32, 8192),
    (64, 3584),
    (64, 4096),
    (64, 5120),
    (64, 8192),
]

PUBLIC_SEQUENCE_LENGTHS = [1, 8, 32, 128]


@dataclass(frozen=True)
class Case:
    bsz: int
    seq: int
    hidden: int
    kind: str = "public"

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.bsz, self.seq, self.hidden)


def public_cases() -> list[Case]:
    return [Case(bsz, seq, hidden) for seq in PUBLIC_SEQUENCE_LENGTHS for bsz, hidden in PUBLIC_BH_SHAPES]


def generalization_cases() -> list[Case]:
    return [
        Case(1, 3, 64, "boundary"),
        Case(2, 5, 257, "boundary"),
        Case(7, 2, 3584, "near-public"),
        Case(8, 3, 4097, "near-public"),
        Case(16, 5, 5130, "near-public"),
        Case(32, 2, 8064, "near-public"),
        Case(1, 32, 8320, "near-public"),
        Case(16, 128, 8320, "near-public"),
        Case(64, 1, 8192, "boundary"),
        Case(5, 33, 4608, "near-public"),
    ]


def make_inputs(shape: tuple[int, int, int], seed: int, device: str) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x1_cpu = torch.empty(shape, dtype=torch.float32).uniform_(-1.0, 1.0, generator=gen)
    x2_cpu = torch.empty(shape, dtype=torch.float32).uniform_(-1.0, 1.0, generator=gen)
    gamma_cpu = torch.empty(shape, dtype=torch.float32).uniform_(0.5, 1.5, generator=gen)
    return (
        x1_cpu.to(device=device, dtype=torch.bfloat16).contiguous(),
        x2_cpu.to(device=device, dtype=torch.bfloat16).contiguous(),
        gamma_cpu.to(device=device, dtype=torch.bfloat16).contiguous(),
    )


def assert_l1_bf16_accuracy(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual64 = actual.cpu().to(torch.float64)
    expected64 = expected.cpu().to(torch.bfloat16).to(torch.float64)
    diff = torch.abs(actual64 - expected64)
    rel = diff / (torch.abs(expected64) + 1e-7)
    mere = float(rel.mean().item())
    mare = float(rel.max().item())
    max_diff = float(diff.max().item())
    threshold = 2**-7
    if not (mere < threshold and mare < 10 * threshold):
        raise AssertionError(
            f"BF16 L1 accuracy failed: MERE={mere:.6g}, MARE={mare:.6g}, max_diff={max_diff:.6g}"
        )
    return {"mere": mere, "mare": mare, "max_diff": max_diff}


def benchmark_once(func, args, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        func(*args)
    torch_npu.npu.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        func(*args)
    torch_npu.npu.synchronize()
    return (time.perf_counter() - start) * 1_000_000 / repeat


def run_case(case: Case, seed: int, device: str, benchmark: bool, warmup: int, repeat: int) -> dict[str, float]:
    x1, x2, gamma = make_inputs(case.shape, seed, device)
    actual = add_rms_norm(x1, x2, gamma)
    expected = add_rms_norm_reference(x1, x2, gamma)
    metrics = assert_l1_bf16_accuracy(actual, expected)
    if benchmark:
        metrics["latency_us"] = benchmark_once(add_rms_norm, (x1, x2, gamma), warmup, repeat)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AddRmsNorm Triton-Ascend implementation.")
    parser.add_argument("--device", default="npu", help="Torch device, default: npu")
    parser.add_argument("--public", action="store_true", help="Run all 80 public B/S/H cases.")
    parser.add_argument("--generalization", action="store_true", help="Run additional non-public shape cases.")
    parser.add_argument("--benchmark", action="store_true", help="Print simple wall-clock latency smoke numbers.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    if torch_npu is None:
        raise RuntimeError("torch_npu is required for NPU validation")

    selected = []
    if args.public or not args.generalization:
        selected.extend(public_cases())
    if args.generalization:
        selected.extend(generalization_cases())

    max_mere = 0.0
    max_mare = 0.0
    max_diff = 0.0
    latencies = []
    for index, case in enumerate(selected):
        metrics = run_case(case, 20260612 + index, args.device, args.benchmark, args.warmup, args.repeat)
        max_mere = max(max_mere, metrics["mere"])
        max_mare = max(max_mare, metrics["mare"])
        max_diff = max(max_diff, metrics["max_diff"])
        if "latency_us" in metrics:
            latencies.append(metrics["latency_us"])
        print(
            f"[PASS] {index + 1:03d}/{len(selected):03d} "
            f"{case.kind} shape={case.shape} "
            f"MERE={metrics['mere']:.3e} MARE={metrics['mare']:.3e} max_diff={metrics['max_diff']:.3e}"
        )

    print(
        f"SUMMARY passed={len(selected)} threshold={2**-7:.8f} "
        f"max_mere={max_mere:.3e} max_mare={max_mare:.3e} max_diff={max_diff:.3e}"
    )
    if latencies:
        geo = math.exp(sum(math.log(max(v, 1e-9)) for v in latencies) / len(latencies))
        print(f"LATENCY wall_clock_us_geomean={geo:.3f} min={min(latencies):.3f} max={max(latencies):.3f}")


if __name__ == "__main__":
    main()
