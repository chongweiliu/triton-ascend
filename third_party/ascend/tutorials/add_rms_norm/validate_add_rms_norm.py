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


def add_rms_norm_torch_npu(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    op = getattr(torch_npu, "npu_add_rms_norm", None)
    if not callable(op):
        raise RuntimeError("torch_npu.npu_add_rms_norm is not available")
    out = op(x1, x2, gamma, float(epsilon))
    if isinstance(out, (tuple, list)):
        if not out:
            raise RuntimeError("torch_npu.npu_add_rms_norm returned no outputs")
        return out[0]
    return out


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(max(v, 1e-9)) for v in values) / len(values))


def format_latency_us(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f} us"


def format_speedup(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6f}x"


def run_case(case: Case, seed: int, device: str, benchmark: bool, warmup: int, repeat: int) -> dict[str, object]:
    x1, x2, gamma = make_inputs(case.shape, seed, device)
    actual = add_rms_norm(x1, x2, gamma)
    expected = add_rms_norm_reference(x1, x2, gamma)
    metrics = assert_l1_bf16_accuracy(actual, expected)
    if benchmark:
        triton_latency = benchmark_once(add_rms_norm, (x1, x2, gamma), warmup, repeat)
        torch_latency = benchmark_once(add_rms_norm_reference, (x1, x2, gamma), warmup, repeat)
        metrics["triton_latency_us"] = triton_latency
        metrics["torch_latency_us"] = torch_latency
        metrics["speedup_vs_torch"] = torch_latency / triton_latency
        try:
            torch_npu_actual = add_rms_norm_torch_npu(x1, x2, gamma)
            try:
                torch_npu_acc = assert_l1_bf16_accuracy(torch_npu_actual, expected)
                metrics["torch_npu_accuracy"] = "PASS"
                metrics["torch_npu_mare"] = torch_npu_acc["mare"]
            except AssertionError:
                metrics["torch_npu_accuracy"] = "FAIL"
                metrics["torch_npu_mare"] = float("nan")
            torch_npu_latency = benchmark_once(add_rms_norm_torch_npu, (x1, x2, gamma), warmup, repeat)
            metrics["torch_npu_latency_us"] = torch_npu_latency
            metrics["speedup_vs_torch_npu"] = torch_npu_latency / triton_latency
        except Exception as exc:
            metrics["torch_npu_accuracy"] = "ERROR"
            metrics["torch_npu_error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"
            metrics["torch_npu_latency_us"] = None
            metrics["speedup_vs_torch_npu"] = None
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
    triton_latencies = []
    torch_latencies = []
    speedups_vs_torch = []
    torch_npu_latencies = []
    speedups_vs_torch_npu = []
    speedups_vs_valid_torch_npu = []
    torch_npu_accuracy_counts = {"PASS": 0, "FAIL": 0, "ERROR": 0}
    for index, case in enumerate(selected):
        metrics = run_case(case, 20260612 + index, args.device, args.benchmark, args.warmup, args.repeat)
        max_mere = max(max_mere, metrics["mere"])
        max_mare = max(max_mare, metrics["mare"])
        max_diff = max(max_diff, metrics["max_diff"])
        latency_text = ""
        if "triton_latency_us" in metrics:
            triton_latency = metrics["triton_latency_us"]
            torch_latency = metrics["torch_latency_us"]
            speedup_vs_torch = metrics["speedup_vs_torch"]
            triton_latencies.append(triton_latency)
            torch_latencies.append(torch_latency)
            speedups_vs_torch.append(speedup_vs_torch)
            torch_npu_accuracy = str(metrics["torch_npu_accuracy"])
            torch_npu_accuracy_counts[torch_npu_accuracy] += 1
            torch_npu_latency = metrics["torch_npu_latency_us"]
            speedup_vs_torch_npu = metrics["speedup_vs_torch_npu"]
            if torch_npu_latency is not None:
                torch_npu_latencies.append(torch_npu_latency)
                speedups_vs_torch_npu.append(speedup_vs_torch_npu)
                if torch_npu_accuracy == "PASS":
                    speedups_vs_valid_torch_npu.append(speedup_vs_torch_npu)
            latency_text = (
                f" triton={format_latency_us(triton_latency)}"
                f" torch={format_latency_us(torch_latency)}"
                f" speedup_vs_torch={format_speedup(speedup_vs_torch)}"
                f" torch_npu={format_latency_us(torch_npu_latency)}"
                f" speedup_vs_torch_npu={format_speedup(speedup_vs_torch_npu)}"
                f" torch_npu_accuracy={torch_npu_accuracy}"
            )
            if "torch_npu_error" in metrics:
                latency_text += f" torch_npu_error={metrics['torch_npu_error']}"
        print(
            f"[PASS] {index + 1:03d}/{len(selected):03d} "
            f"{case.kind} shape={case.shape} "
            f"MERE={metrics['mere']:.3e} MARE={metrics['mare']:.3e} max_diff={metrics['max_diff']:.3e}"
            f"{latency_text}"
        )

    print(
        f"SUMMARY passed={len(selected)} threshold={2**-7:.8f} "
        f"max_mere={max_mere:.3e} max_mare={max_mare:.3e} max_diff={max_diff:.3e}"
    )
    if triton_latencies:
        print(
            f"LATENCY triton_geomean={geomean(triton_latencies):.3f} us "
            f"triton_min={min(triton_latencies):.3f} us triton_max={max(triton_latencies):.3f} us "
            f"torch_geomean={geomean(torch_latencies):.3f} us "
            f"torch_min={min(torch_latencies):.3f} us torch_max={max(torch_latencies):.3f} us "
            f"speedup_vs_torch_geomean={geomean(speedups_vs_torch):.6f}x"
        )
        valid_torch_npu_speedup = (
            f"{geomean(speedups_vs_valid_torch_npu):.6f}x" if speedups_vs_valid_torch_npu else "N/A"
        )
        print(
            f"LATENCY_TORCH_NPU timed_cases={len(torch_npu_latencies)} "
            f"accuracy_pass={torch_npu_accuracy_counts['PASS']} "
            f"accuracy_fail={torch_npu_accuracy_counts['FAIL']} "
            f"accuracy_error={torch_npu_accuracy_counts['ERROR']} "
            f"torch_npu_geomean={format_latency_us(geomean(torch_npu_latencies) if torch_npu_latencies else None)} "
            f"torch_npu_min={format_latency_us(min(torch_npu_latencies) if torch_npu_latencies else None)} "
            f"torch_npu_max={format_latency_us(max(torch_npu_latencies) if torch_npu_latencies else None)} "
            f"speedup_vs_torch_npu_timed_geomean={format_speedup(geomean(speedups_vs_torch_npu) if speedups_vs_torch_npu else None)} "
            f"speedup_vs_torch_npu_accuracy_pass_geomean={valid_torch_npu_speedup}"
        )


if __name__ == "__main__":
    main()
