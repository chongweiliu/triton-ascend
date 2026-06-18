# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Validation helper for the KvRmsNormRopeCache Triton-Ascend tutorial."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

try:
    import torch_npu  # noqa: F401  # pylint: disable=unused-import
except Exception as exc:  # pragma: no cover - requires Ascend runtime.
    raise RuntimeError(f"torch_npu import failed; source the Ascend runtime first: {exc}") from exc

from kv_rms_norm_rope_cache import kv_rms_norm_rope_cache

DOC_MODEL_ROWS = [
    (1, 28),
    (1, 32),
    (1, 40),
    (1, 64),
    (8, 28),
    (8, 32),
    (8, 40),
    (8, 64),
    (16, 28),
    (16, 32),
    (16, 40),
    (16, 64),
    (32, 28),
    (32, 32),
    (32, 40),
    (32, 64),
    (64, 28),
    (64, 32),
    (64, 40),
    (64, 64),
]
SPLIT_CHOICES = [
    (64, 64),
    (32, 96),
    (48, 80),
    (80, 48),
    (96, 32),
    (64, 128),
    (128, 64),
]
BF16_THRESHOLD = 0.02
DEFAULT_EPSILON = 1.0e-5


@dataclass(frozen=True)
class Case:
    case_id: str
    kind: str
    bsz: int
    heads: int
    skv: int
    scache: int
    bcache: int
    d_rope: int
    d_value: int
    seed: int
    note: str

    @property
    def shape(self) -> dict[str, int]:
        return {
            "Bkv": self.bsz,
            "Bcache": self.bcache,
            "N": self.heads,
            "Skv": self.skv,
            "Scache": self.scache,
            "D": self.d_rope + self.d_value,
            "Dk": self.d_rope,
            "Dv": self.d_value,
        }


def public_cases() -> list[Case]:
    cases: list[Case] = []
    for idx, (bsz, heads) in enumerate(DOC_MODEL_ROWS, start=1):
        cases.append(
            Case(
                case_id=f"public_{idx:03d}",
                kind="public",
                bsz=bsz,
                heads=heads,
                skv=1,
                scache=8,
                bcache=bsz,
                d_rope=64,
                d_value=64,
                seed=_seed_from_case_id(f"public_{idx:03d}", 20260617),
                note="documented model row, public 64/64 split",
            ))
    return cases


def random_generalization_cases(count: int, seed: int) -> list[Case]:
    if count < 0:
        raise ValueError("random generalization count must be non-negative")
    rng = random.Random(int(seed))
    cases: list[Case] = []
    seen: set[tuple[int, int, int, int, int, int, int]] = set()
    max_attempts = max(200, count * 200)
    attempts = 0
    while len(cases) < count and attempts < max_attempts:
        attempts += 1
        bsz, heads = DOC_MODEL_ROWS[(rng.randrange(len(DOC_MODEL_ROWS)) + attempts) % len(DOC_MODEL_ROWS)]
        d_rope, d_value = SPLIT_CHOICES[(rng.randrange(len(SPLIT_CHOICES)) + attempts) % len(SPLIT_CHOICES)]
        skv = rng.choice([1, 2, 3, 4])
        scache = rng.choice([4, 5, 8, 9, 12])
        bcache = bsz + rng.choice([0, 0, 1])
        signature = (bsz, heads, skv, scache, bcache, d_rope, d_value)
        if signature in seen:
            continue
        seen.add(signature)
        idx = len(cases) + 1
        case_id = f"random_{idx:03d}"
        cases.append(
            Case(
                case_id=case_id,
                kind="random_generalization",
                bsz=bsz,
                heads=heads,
                skv=skv,
                scache=scache,
                bcache=bcache,
                d_rope=d_rope,
                d_value=d_value,
                seed=_seed_from_case_id(case_id, seed),
                note="seeded docs-row dynamic Dk/Dv sample",
            ))
    if len(cases) != count:
        raise RuntimeError(f"generated {len(cases)} random cases after {attempts} attempts, expected {count}")
    return cases


def _seed_from_case_id(case_id: str, seed: int) -> int:
    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    stable = int.from_bytes(digest[:8], "big") % (2**31)
    return (stable + int(seed)) % (2**31)


def _bf16_uniform(shape: Iterable[int], min_val: float, max_val: float, gen: torch.Generator) -> torch.Tensor:
    value = torch.rand(tuple(shape), dtype=torch.float64, generator=gen)
    return (value * (max_val - min_val) + min_val).to(torch.bfloat16)


def make_inputs(case: Case, device: torch.device) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator()
    gen.manual_seed(int(case.seed))
    d_total = case.d_rope + case.d_value
    kv = _bf16_uniform((case.bsz, case.heads, case.skv, d_total), -1.0, 1.0, gen)
    gamma = _bf16_uniform((case.d_value, ), 0.5, 1.5, gen)
    cos = _bf16_uniform((case.bsz, 1, 1, case.d_rope), -1.0, 1.0, gen)
    sin = _bf16_uniform((case.bsz, 1, 1, case.d_rope), -1.0, 1.0, gen)
    k_cache = _bf16_uniform((case.bcache, case.heads, case.scache, case.d_rope), -0.25, 0.25, gen)
    ckv_cache = _bf16_uniform((case.bcache, case.heads, case.scache, case.d_value), -0.25, 0.25, gen)
    index = torch.arange(case.bsz * case.skv, dtype=torch.int64).reshape(case.bsz, case.skv) % case.scache
    if index.numel() >= 7:
        index.reshape(-1)[6::7] = -1
    return (
        kv.to(device),
        gamma.to(device),
        cos.to(device),
        sin.to(device),
        index.to(device),
        k_cache.to(device),
        ckv_cache.to(device),
    )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = int(x.shape[-1]) // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def reference_impl(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    index: torch.Tensor,
    k_cache: torch.Tensor,
    ckv_cache: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_rope = int(cos.shape[-1])
    d_value = int(gamma.shape[0])
    rope = kv[..., :d_rope].to(torch.float32)
    value = kv[..., d_rope:d_rope + d_value].to(torch.float32)
    k_rope = rope * cos.to(torch.float32) + rotate_half(rope) * sin.to(torch.float32)
    variance = torch.mean(value * value, dim=-1, keepdim=True)
    ckv = value * torch.rsqrt(variance + float(epsilon)) * gamma.to(torch.float32)
    k_out = k_cache.clone()
    ckv_out = ckv_cache.clone()
    index_cpu = index.cpu()
    bsz, _, skv, _ = kv.shape
    scache = int(k_cache.shape[2])
    for b in range(int(bsz)):
        for s in range(int(skv)):
            pos = int(index_cpu[b, s].item())
            if pos == -1:
                continue
            if pos < 0 or pos >= scache:
                raise ValueError(f"cache index out of range: {pos}")
            k_out[b, :, pos, :] = k_rope[b, :, s, :].to(k_out.dtype)
            ckv_out[b, :, pos, :] = ckv[b, :, s, :].to(ckv_out.dtype)
    return k_out, ckv_out


def compare_outputs(actual: tuple[torch.Tensor, torch.Tensor],
                    expected: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float | int | bool]:
    mismatch_count = 0
    total_count = 0
    max_diff = 0.0
    sum_diff = 0.0
    max_rel = 0.0
    sum_rel = 0.0
    for out, ref in zip(actual, expected):
        out_f = out.detach().to(torch.float32).cpu()
        ref_f = ref.detach().to(torch.float32).cpu()
        diff = torch.abs(out_f - ref_f)
        denom = torch.clamp(torch.abs(ref_f), min=1.0e-6)
        rel = diff / denom
        mismatch_count += int((diff > BF16_THRESHOLD).sum().item())
        total_count += int(diff.numel())
        max_diff = max(max_diff, float(diff.max().item()) if diff.numel() else 0.0)
        sum_diff += float(diff.sum().item())
        max_rel = max(max_rel, float(rel.max().item()) if rel.numel() else 0.0)
        sum_rel += float(rel.sum().item())
    mean_diff = sum_diff / max(1, total_count)
    mare = sum_rel / max(1, total_count)
    return {
        "passed": mismatch_count == 0,
        "threshold": BF16_THRESHOLD,
        "mismatch_count": mismatch_count,
        "total_count": total_count,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "mere": max_rel,
        "mare": mare,
    }


def run_candidate(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    kv, gamma, cos, sin, index, k_cache, ckv_cache = inputs
    return kv_rms_norm_rope_cache(
        kv,
        gamma,
        cos,
        sin,
        index,
        k_cache.clone(),
        ckv_cache.clone(),
        DEFAULT_EPSILON,
        "Norm",
    )


def benchmark_case(inputs: tuple[torch.Tensor, ...], warmup: int, repeat: int) -> dict[str, float | int | str]:
    for _ in range(int(warmup)):
        run_candidate(inputs)
    torch.npu.synchronize()
    samples: list[float] = []
    for _ in range(int(repeat)):
        torch.npu.synchronize()
        start = time.perf_counter()
        run_candidate(inputs)
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1.0e6)
    samples_sorted = sorted(samples)
    return {
        "latency_us": samples_sorted[len(samples_sorted) // 2],
        "mean_latency_us": sum(samples_sorted) / len(samples_sorted),
        "min_latency_us": min(samples_sorted),
        "max_latency_us": max(samples_sorted),
        "repeat": int(repeat),
        "warmup": int(warmup),
        "timing_source": "delivery_wall_sync_us",
    }


def run_case(case: Case, device: torch.device, args: argparse.Namespace) -> dict[str, object]:
    inputs = make_inputs(case, device)
    expected = reference_impl(*inputs, DEFAULT_EPSILON)
    actual = run_candidate(inputs)
    torch.npu.synchronize()
    compare = compare_outputs(actual, expected)
    record: dict[str, object] = {
        "case": asdict(case),
        "shape": case.shape,
        "dtype": "bfloat16",
        "cache_mode": "Norm",
        "epsilon": DEFAULT_EPSILON,
        "accuracy": compare,
    }
    if args.benchmark:
        record["benchmark"] = benchmark_case(inputs, args.warmup, args.repeat)
    return record


def print_record(index: int, total: int, record: dict[str, object]) -> None:
    case = record["case"]
    shape = record["shape"]
    accuracy = record["accuracy"]
    bench = record.get("benchmark") or {}
    status = "PASS" if accuracy["passed"] else "FAIL"
    latency = ""
    if bench:
        latency = f" latency={bench['latency_us']:.3f} us timing_source={bench['timing_source']}"
    print(f"[{status}] {index:03d}/{total:03d} {case['kind']} id={case['case_id']} "
          f"shape=B{shape['Bkv']},N{shape['N']},Skv{shape['Skv']},Scache{shape['Scache']},"
          f"Dk{shape['Dk']},Dv{shape['Dv']} mismatches={accuracy['mismatch_count']} "
          f"max_diff={accuracy['max_diff']:.6g} MARE={accuracy['mare']:.6g}{latency}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", action="store_true", help="run the 20 documented public-style cases")
    parser.add_argument("--random-generalization", type=int, default=0,
                        help="number of seeded random dynamic split cases")
    parser.add_argument("--random-seed", type=int, default=20260617)
    parser.add_argument("--benchmark", action="store_true", help="also collect delivery wall-sync candidate timings")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.public and args.random_generalization == 0:
        args.public = True
    npu_id = int(os.environ.get("NPU_ID", "0"))
    device = torch.device(f"npu:{npu_id}")
    cases: list[Case] = []
    if args.public:
        cases.extend(public_cases())
    cases.extend(random_generalization_cases(args.random_generalization, args.random_seed))
    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    with (args.jsonl.open("w", encoding="utf-8") if args.jsonl else open(os.devnull, "w", encoding="utf-8")) as jf:
        for idx, case in enumerate(cases, start=1):
            record = run_case(case, device, args)
            records.append(record)
            jf.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            jf.flush()
            print_record(idx, len(cases), record)

    passed = sum(1 for record in records if record["accuracy"]["passed"])
    latencies = [
        float(record["benchmark"]["latency_us"])
        for record in records
        if "benchmark" in record and record["accuracy"]["passed"]
    ]
    summary: dict[str, object] = {
        "total": len(records),
        "passed": passed,
        "failed": len(records) - passed,
        "public_cases": sum(1 for record in records if record["case"]["kind"] == "public"),
        "random_generalization_cases":
        sum(1 for record in records if record["case"]["kind"] == "random_generalization"),
        "random_seed": args.random_seed,
        "benchmark": bool(args.benchmark),
        "timing_source": "delivery_wall_sync_us" if args.benchmark else "",
    }
    if latencies:
        summary["latency_us"] = {
            "geomean": math.exp(sum(math.log(max(v, 1.0e-9)) for v in latencies) / len(latencies)),
            "mean": sum(latencies) / len(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "count": len(latencies),
        }
    if args.summary_json:
        args.summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"SUMMARY total={summary['total']} passed={summary['passed']} failed={summary['failed']} "
          f"public={summary['public_cases']} random={summary['random_generalization_cases']}")
    if latencies:
        lat = summary["latency_us"]
        print(
            f"LATENCY_DELIVERY_WALL geomean={lat['geomean']:.3f} us mean={lat['mean']:.3f} us "
            f"min={lat['min']:.3f} us max={lat['max']:.3f} us count={lat['count']} timing_source=delivery_wall_sync_us")
    if args.jsonl:
        print(f"CANONICAL_JSONL path={args.jsonl}")
    if args.summary_json:
        print(f"CANONICAL_JSONL_SUMMARY path={args.summary_json}")
    if passed != len(records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
