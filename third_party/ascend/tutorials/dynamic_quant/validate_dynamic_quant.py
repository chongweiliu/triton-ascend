# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Validation helper for the DynamicQuant Triton-Ascend tutorial."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401
except Exception as exc:  # pragma: no cover - depends on Ascend runtime.
    torch_npu = None
    _TORCH_NPU_IMPORT_ERROR = exc
else:
    _TORCH_NPU_IMPORT_ERROR = None

from dynamic_quant import dynamic_quant


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
PUBLIC_VALUE_RANGES = [
    (0.0, 0.0),
    (-0.01, 0.01),
    (-1.0, 1.0),
    (-8.0, 8.0),
    (-64.0, 64.0),
    (-256.0, 256.0),
    (-1024.0, 1024.0),
]
RANDOM_GENERALIZATION_POLICY = "seeded_dynamic_quant_bsh_v1"
SCALE_THRESHOLD = 1.0e-3
INT_THRESHOLD = 1


@dataclass(frozen=True)
class Case:
    case_id: str
    bsz: int
    seq: int
    hidden: int
    dst_type: str
    value_range: tuple[float, float]
    kind: str = "public"
    seed: int = 0
    note: str = ""

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.bsz, self.seq, self.hidden)


def _case_seed(case_id: str, seed: int = 0) -> int:
    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    deterministic_hash = int.from_bytes(digest[:8], byteorder="big") % (2**31)
    return (int(seed) + deterministic_hash) % (2**31)


def public_cases() -> list[Case]:
    cases: list[Case] = []
    for index, (bsz, hidden) in enumerate(PUBLIC_BH_SHAPES, start=1):
        value_range = PUBLIC_VALUE_RANGES[(index - 1) % len(PUBLIC_VALUE_RANGES)]
        case_id = f"custom/dynamic_quant_{index}"
        cases.append(Case(case_id, bsz, 1, hidden, "int8", value_range, seed=_case_seed(case_id)))
    for offset, (bsz, hidden) in enumerate(PUBLIC_BH_SHAPES, start=21):
        value_range = PUBLIC_VALUE_RANGES[(offset - 21) % len(PUBLIC_VALUE_RANGES)]
        case_id = f"custom/dynamic_quant_{offset}"
        cases.append(Case(case_id, bsz, 1, hidden, "int4", value_range, seed=_case_seed(case_id)))
    return cases


def random_generalization_cases(count: int, seed: int) -> list[Case]:
    if count < 0:
        raise ValueError("random generalization count must be non-negative")
    rng = random.Random(int(seed))
    public_shapes = {(case.shape, case.dst_type) for case in public_cases()}
    seen: set[tuple[tuple[int, int, int], str]] = set()
    cases: list[Case] = []
    attempts = 0
    max_attempts = max(200, count * 200)

    while len(cases) < count and attempts < max_attempts:
        attempts += 1
        category = rng.choice(["near_hidden", "boundary_shape", "large_hidden", "small_tail"])
        base_bsz, base_hidden = rng.choice(PUBLIC_BH_SHAPES)
        if category == "near_hidden":
            delta = rng.choice([-640, -512, -384, -257, -128, 64, 127, 256, 384, 640])
            bsz = base_bsz
            seq = rng.choice([1, 2, 3, 4])
            hidden = max(32, base_hidden + delta)
        elif category == "large_hidden":
            bsz = rng.choice([1, 2, 4, 8, 16])
            seq = rng.choice([1, 2, 4, 8])
            hidden = rng.randint(8193, 12288)
        elif category == "small_tail":
            bsz = rng.choice([1, 2, 3, 5, 8, 13, 16])
            seq = rng.choice([1, 2, 3, 5, 7, 11])
            hidden = rng.randint(32, 1024)
        else:
            bsz = rng.randint(1, 64)
            seq = rng.randint(1, 16)
            hidden = 32 * (1 + rng.randrange(256))

        dst_type = "int4" if len(cases) % 2 else "int8"
        value_range = rng.choice(PUBLIC_VALUE_RANGES + [(-512.0, 512.0), (-8.0, 2.0)])
        key = ((bsz, seq, hidden), dst_type)
        if key in public_shapes or key in seen:
            continue
        if bsz * seq * hidden > 8 * 1024 * 1024:
            continue
        seen.add(key)
        case_index = len(cases) + 1
        case_id = f"custom/dynamic_quant_random_{case_index:03d}"
        cases.append(
            Case(
                case_id,
                bsz,
                seq,
                hidden,
                dst_type,
                value_range,
                kind="random_generalization",
                seed=_case_seed(case_id, seed),
                note=f"{RANDOM_GENERALIZATION_POLICY}:{category}",
            ))

    if len(cases) != count:
        raise RuntimeError(f"generated {len(cases)} random cases after {attempts} attempts, expected {count}")
    return cases


def _require_npu(device: str) -> None:
    if _TORCH_NPU_IMPORT_ERROR is not None:
        raise RuntimeError(f"torch_npu import failed: {_TORCH_NPU_IMPORT_ERROR}")
    if device != "npu":
        raise ValueError("this tutorial validates the Triton-Ascend path on device='npu' only")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")


def _make_input(case: Case, device: str) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(int(case.seed))
    low, high = case.value_range
    if low == high:
        x_cpu = torch.full(case.shape, float(low), dtype=torch.float32)
    else:
        x_cpu = torch.rand(case.shape, dtype=torch.float32, generator=gen) * (high - low) + low
    return x_cpu.to(torch.bfloat16).to(device=device).contiguous()


def _reference_dynamic_quant(x: torch.Tensor, dst_type: str) -> tuple[torch.Tensor, torch.Tensor]:
    if dst_type == "int4":
        qmin, qmax, qabs = -8, 7, 7
    elif dst_type == "int8":
        qmin, qmax, qabs = -128, 127, 127
    else:
        raise ValueError(f"unsupported dst_type {dst_type!r}")
    x_compute = x.to(torch.float32)
    abs_max = torch.max(torch.abs(x_compute), dim=-1, keepdim=True)[0]
    scale_out = abs_max.clamp(min=1.0e-12) / float(qabs)
    output = torch.clamp(torch.round(x_compute / scale_out), qmin, qmax).to(torch.int8)
    return output, scale_out.squeeze(-1).to(torch.float32)


def _compare(case: Case, output: torch.Tensor, scale: torch.Tensor, ref_output: torch.Tensor, ref_scale: torch.Tensor) -> dict[str, object]:
    output_cpu = output.detach().cpu()
    scale_cpu = scale.detach().cpu()
    ref_output_cpu = ref_output.detach().cpu()
    ref_scale_cpu = ref_scale.detach().cpu()
    int_diff = (output_cpu.to(torch.int16) - ref_output_cpu.to(torch.int16)).abs()
    scale_diff = (scale_cpu - ref_scale_cpu).abs()
    max_int_diff = int(int_diff.max().item()) if int_diff.numel() else 0
    mismatch_count = int((int_diff > INT_THRESHOLD).sum().item()) if int_diff.numel() else 0
    scale_max_abs_diff = float(scale_diff.max().item()) if scale_diff.numel() else 0.0
    scale_mismatch_count = int((scale_diff > SCALE_THRESHOLD).sum().item()) if scale_diff.numel() else 0
    if case.dst_type == "int4":
        range_ok = bool(((output_cpu >= -8) & (output_cpu <= 7)).all().item())
    else:
        range_ok = bool(((output_cpu >= -128) & (output_cpu <= 127)).all().item())
    passed = mismatch_count == 0 and scale_mismatch_count == 0 and range_ok
    return {
        "passed": passed,
        "mismatch_count": mismatch_count,
        "output_max_diff": max_int_diff,
        "output_threshold": INT_THRESHOLD,
        "scale_mismatch_count": scale_mismatch_count,
        "scale_max_abs_diff": scale_max_abs_diff,
        "scale_threshold": SCALE_THRESHOLD,
        "range_ok": range_ok,
    }


def _debug_host_benchmark(case: Case, x: torch.Tensor, repeat: int, warmup: int) -> dict[str, object]:
    for _ in range(int(warmup)):
        dynamic_quant(x, case.dst_type)
    torch.npu.synchronize()
    times: list[float] = []
    for _ in range(int(repeat)):
        start = time.perf_counter_ns()
        dynamic_quant(x, case.dst_type)
        torch.npu.synchronize()
        end = time.perf_counter_ns()
        times.append((end - start) / 1000.0)
    return {
        "debug_host_median_us": statistics.median(times) if times else None,
        "debug_host_repeat": repeat,
        "debug_host_warmup": warmup,
        "timing_note": "debug host elapsed time; official delivery evidence uses OpForge profiler artifacts",
    }


def run_case(case: Case, args: argparse.Namespace) -> dict[str, object]:
    x = _make_input(case, args.device)
    with torch.inference_mode():
        ref_output, ref_scale = _reference_dynamic_quant(x, case.dst_type)
        output, scale = dynamic_quant(x, case.dst_type)
        torch.npu.synchronize()
    result = _compare(case, output, scale, ref_output, ref_scale)
    record: dict[str, object] = {
        "case_id": case.case_id,
        "kind": case.kind,
        "shape": list(case.shape),
        "dtype": "bfloat16",
        "dst_type": case.dst_type,
        "value_range": list(case.value_range),
        "seed": case.seed,
        "note": case.note,
        "triton_accuracy": "PASS" if result["passed"] else "FAIL",
        **result,
    }
    if args.benchmark:
        record.update(_debug_host_benchmark(case, x, args.repeat, args.warmup))
    return record


def _format_us(value: object) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.3f} us"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", action="store_true", help="run the 40 documented public-style cases")
    parser.add_argument("--random-generalization", type=int, default=0, help="number of seeded random non-public cases")
    parser.add_argument("--random-seed", type=int, default=20260617)
    parser.add_argument("--device", default="npu")
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--benchmark", action="store_true", help="also collect debug host elapsed timing")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    _require_npu(args.device)
    cases: list[Case] = []
    if args.public or args.random_generalization == 0:
        cases.extend(public_cases())
    if args.random_generalization:
        cases.extend(random_generalization_cases(args.random_generalization, args.random_seed))

    args.jsonl.parent.mkdir(parents=True, exist_ok=True) if args.jsonl else None
    passed = 0
    records: list[dict[str, object]] = []
    with (args.jsonl.open("w", encoding="utf-8") if args.jsonl else nullcontext()) as out_file:
        for index, case in enumerate(cases, start=1):
            record = run_case(case, args)
            records.append(record)
            if record["passed"]:
                passed += 1
            if out_file:
                out_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            status = "PASS" if record["passed"] else "FAIL"
            timing = _format_us(record.get("debug_host_median_us"))
            print(
                f"[{status}] {index:03d}/{len(cases):03d} {case.case_id} "
                f"shape={case.shape} dst_type={case.dst_type} "
                f"output_max_diff={record['output_max_diff']} "
                f"scale_max_abs_diff={record['scale_max_abs_diff']:.6g} "
                f"debug_host={timing}")

    print(
        "SUMMARY "
        f"total={len(cases)} passed={passed} failed={len(cases) - passed} "
        f"public={sum(1 for r in records if r['kind'] == 'public')} "
        f"random_generalization={sum(1 for r in records if r['kind'] == 'random_generalization')}")
    return 0 if passed == len(cases) else 1


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc_info):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
