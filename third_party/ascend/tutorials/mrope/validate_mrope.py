# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Self-validation for the MRoPE Triton-Ascend tutorial delivery."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from mrope import mrope

BF16_THRESHOLD = 0.02
RANDOM_GENERALIZATION_POLICY = "seeded_mrope_metadata_v1"


@dataclass(frozen=True)
class Case:
    case_id: str
    kind: str
    input_shape: list[list[int]]
    attrs: dict[str, object]
    value_range: list[list[float]]
    note: str = ""
    random_category: str = ""


def _as_section(mrope_section) -> list[int]:
    if mrope_section is None:
        return [0, 0, 0]
    section = [int(v) for v in list(mrope_section)]
    return section or [0, 0, 0]


def _is_rope(section: Sequence[int]) -> bool:
    return not section or all(int(v) == 0 for v in section)


def _decode_cache(cache: torch.Tensor, cache_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    if str(cache_mode) == "interleave":
        return cache[:, 0::2].to(torch.float32), cache[:, 1::2].to(torch.float32)
    half = int(cache.shape[1]) // 2
    return cache[:, :half].to(torch.float32), cache[:, half:].to(torch.float32)


def _assemble_cos_sin(
    positions: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    section: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    max_seq = int(cos_table.shape[0])
    if positions.numel() and (int(positions.min().item()) < 0 or int(positions.max().item()) >= max_seq):
        raise ValueError("positions values must be within cos_sin_cache max_seq_len")
    if _is_rope(section):
        pos = positions.to(torch.long)
        return cos_table[pos], sin_table[pos]
    cos_parts = []
    sin_parts = []
    start = 0
    for row, length in enumerate(section):
        end = start + int(length)
        pos = positions[row].to(torch.long)
        cos_parts.append(cos_table[pos, start:end])
        sin_parts.append(sin_table[pos, start:end])
        start = end
    return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)


def mrope_reference(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int = 128,
    mrope_section=None,
    rotary_mode: str = "half",
    cache_mode: str = "default",
) -> tuple[torch.Tensor, torch.Tensor]:
    section = _as_section(mrope_section)
    head = int(head_size)
    rotary_dim = int(cos_sin_cache.shape[1])
    half = rotary_dim // 2
    if not _is_rope(section) and sum(section) != half:
        raise ValueError("sum(mrope_section) must equal rotary_dim / 2")
    cos_table, sin_table = _decode_cache(cos_sin_cache, cache_mode)
    cos, sin = _assemble_cos_sin(positions, cos_table, sin_table, section)
    return (
        _apply_to_tensor(query, cos, sin, head, rotary_dim, rotary_mode),
        _apply_to_tensor(key, cos, sin, head, rotary_dim, rotary_mode),
    )


def _apply_to_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    rotary_mode: str,
) -> torch.Tensor:
    token_count = int(x.shape[0])
    heads = int(x.shape[1]) // int(head_size)
    out = x.to(torch.float32).reshape(token_count, heads, int(head_size)).clone()
    rotary = out[:, :, :rotary_dim]
    half = rotary_dim // 2
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    if str(rotary_mode) == "interleaved":
        even = rotary[..., 0::2]
        odd = rotary[..., 1::2]
        rotated = torch.empty_like(rotary)
        rotated[..., 0::2] = even * cos - odd * sin
        rotated[..., 1::2] = odd * cos + even * sin
    else:
        first = rotary[..., :half]
        second = rotary[..., half:]
        rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
    out[:, :, :rotary_dim] = rotated
    return out.reshape_as(x).to(dtype=x.dtype)


def _seed_from_case_id(case_id: str, seed: int = 0) -> int:
    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    deterministic_hash = int.from_bytes(digest[:8], byteorder="big") % (2**31)
    return (int(seed) + deterministic_hash) % (2**31)


def _gen_bf16_uniform(shape: Sequence[int], value_range: Sequence[float], gen: torch.Generator) -> torch.Tensor:
    low, high = float(value_range[0]), float(value_range[1])
    tensor = torch.rand(tuple(int(v) for v in shape), dtype=torch.float64, generator=gen)
    return (tensor * (high - low) + low).to(torch.bfloat16)


def _make_inputs(case: Case, device: str, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    shapes = case.input_shape
    attrs = case.attrs
    head_size = int(attrs.get("head_size", 128))
    rotary_dim = int(shapes[3][1])
    half = rotary_dim // 2
    token_count = int(shapes[1][0])
    max_seq = int(shapes[3][0])

    if len(shapes[0]) == 1:
        base = torch.arange(token_count, dtype=torch.int64) % max(1, max_seq)
        positions_cpu = base
    else:
        rows = int(shapes[0][0])
        base = torch.arange(token_count, dtype=torch.int64) % max(1, max_seq)
        positions_cpu = torch.stack([(base + 7 * row) % max(1, max_seq) for row in range(rows)], dim=0)

    query_cpu = _gen_bf16_uniform(shapes[1], case.value_range[1], gen)
    key_cpu = _gen_bf16_uniform(shapes[2], case.value_range[2], gen)
    idx = torch.arange(max_seq, dtype=torch.float32).unsqueeze(1)
    freqs = torch.arange(half, dtype=torch.float32).unsqueeze(0)
    angles = idx / torch.pow(torch.tensor(10000.0), (2.0 * freqs) / max(1, rotary_dim))
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    if str(attrs.get("cache_mode", "default")) == "interleave":
        cache_cpu = torch.empty((max_seq, rotary_dim), dtype=torch.float32)
        cache_cpu[:, 0::2] = cos
        cache_cpu[:, 1::2] = sin
    else:
        cache_cpu = torch.cat((cos, sin), dim=-1)

    return (
        positions_cpu.to(device=device, dtype=torch.int64).contiguous(),
        query_cpu.to(device=device).contiguous(),
        key_cpu.to(device=device).contiguous(),
        cache_cpu.to(device=device, dtype=torch.bfloat16).contiguous(),
    )


def load_public_cases(evidence_path: Path) -> list[Case]:
    data = json.loads(evidence_path.read_text(encoding="utf-8"))
    cases = []
    for item in data["public_cases"]:
        cases.append(
            Case(
                case_id=str(item["case_id"]),
                kind="public",
                input_shape=[[int(v) for v in shape] for shape in item["input_shape"]],
                attrs=dict(item["attrs"]),
                value_range=[[float(v) for v in rng] for rng in item["value_range"]],
                note=str(item.get("note", "")),
            ))
    return cases


def random_generalization_cases(count: int, seed: int) -> list[Case]:
    rng = random.Random(int(seed))
    templates = [
        ("rope_half_default", [0, 0, 0], "half", "default", 128, 1),
        ("mrope3_half_default", [16, 24, 24], "half", "default", 128, 3),
        ("mrope3_interleaved_default", [24, 20, 20], "interleaved", "default", 128, 3),
        ("mrope4_half_default", [16, 16, 16, 16], "half", "default", 128, 4),
        ("mrope3_half_interleave64", [8, 12, 12], "half", "interleave", 64, 3),
    ]
    cases: list[Case] = []
    seen: set[tuple[object, ...]] = set()
    attempts = 0
    while len(cases) < count and attempts < count * 100:
        attempts += 1
        name, section, rotary_mode, cache_mode, rotary_dim, rows = rng.choice(templates)
        tokens = rng.choice([1, 2, 3, 5, 7, 8, 17, 31, 33, 64])
        heads = rng.choice([4, 8, 16, 28, 32, 40, 64])
        max_seq = rng.choice([256, 512, 1024, 2048])
        shape_key = (name, tokens, heads, max_seq)
        if shape_key in seen:
            continue
        seen.add(shape_key)
        hidden = heads * 128
        pos_shape = [tokens] if rows == 1 else [rows, tokens]
        value_kind = rng.choice(["small", "ordinary", "wide"])
        if value_kind == "small":
            q_range = [-0.01, 0.01]
            k_range = [-0.01, 0.01]
        elif value_kind == "ordinary":
            q_range = [-1.0, 1.0]
            k_range = [-1.0, 1.0]
        else:
            q_range = [-128.0, 128.0]
            k_range = [-64.0, 64.0]
        idx = len(cases) + 1
        cases.append(
            Case(
                case_id=f"custom/mrope_random_{idx:03d}",
                kind="random_generalization",
                input_shape=[pos_shape, [tokens, hidden], [tokens, hidden], [max_seq, rotary_dim]],
                attrs={
                    "head_size": 128,
                    "mrope_section": list(section),
                    "rotary_mode": rotary_mode,
                    "cache_mode": cache_mode,
                },
                value_range=[[0, max_seq - 1], q_range, k_range, [-1.0, 1.0]],
                note="seeded non-public MRoPE metadata sample",
                random_category=name,
            ))
    if len(cases) != count:
        raise RuntimeError(f"generated {len(cases)} random cases, expected {count}")
    return cases


def compare_outputs(actual: tuple[torch.Tensor, torch.Tensor], expected: tuple[torch.Tensor,
                                                                               torch.Tensor]) -> dict[str, object]:
    output_results = []
    total_mismatch = 0
    max_diff = 0.0
    max_mare = 0.0
    for index, (out, ref) in enumerate(zip(actual, expected)):
        diff = (out.float() - ref.float()).abs()
        allowed = BF16_THRESHOLD + BF16_THRESHOLD * ref.float().abs()
        mismatch = diff > allowed
        mismatch_count = int(mismatch.sum().item())
        total = int(diff.numel())
        mare = float((diff / (ref.float().abs() + 1.0e-6)).max().item()) if total else 0.0
        item_max = float(diff.max().item()) if total else 0.0
        total_mismatch += mismatch_count
        max_diff = max(max_diff, item_max)
        max_mare = max(max_mare, mare)
        output_results.append({
            "index": index,
            "passed": mismatch_count == 0,
            "mismatch_count": mismatch_count,
            "total_count": total,
            "max_diff": item_max,
            "mare": mare,
        })
    return {
        "passed": total_mismatch == 0,
        "threshold": BF16_THRESHOLD,
        "mismatch_count": total_mismatch,
        "max_diff": max_diff,
        "mare": max_mare,
        "output_results": output_results,
    }


def run_case(case: Case, index: int, total: int, device: str, eval_seed: int) -> dict[str, object]:
    seed = _seed_from_case_id(case.case_id, eval_seed)
    inputs = _make_inputs(case, device, seed)
    attrs = case.attrs
    with torch.inference_mode():
        actual = mrope(*inputs, **attrs)
        expected = mrope_reference(*inputs, **attrs)
        if device.startswith("npu"):
            torch.npu.synchronize()
    compare = compare_outputs(actual, expected)
    status = "PASS" if compare["passed"] else "FAIL"
    shape = case.input_shape
    print(f"[{status}] {index:03d}/{total:03d} {case.kind} case_id={case.case_id} "
          f"positions={shape[0]} query={shape[1]} cache={shape[3]} "
          f"rotary_mode={attrs.get('rotary_mode')} cache_mode={attrs.get('cache_mode')} "
          f"section={attrs.get('mrope_section')} mismatch={compare['mismatch_count']} "
          f"mare={compare['mare']:.6e} max_diff={compare['max_diff']:.6e}")
    return {
        "case_id": case.case_id,
        "kind": case.kind,
        "note": case.note,
        "random_category": case.random_category,
        "input_shape": case.input_shape,
        "attrs": case.attrs,
        "seed": seed,
        "triton_accuracy": "PASS" if compare["passed"] else "FAIL",
        "compare": compare,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", action="store_true", help="run public cases from OPFORGE_EVIDENCE.json")
    parser.add_argument("--random-generalization", type=int, default=0, help="number of seeded random cases")
    parser.add_argument("--random-seed", type=int, default=20260617)
    parser.add_argument("--device", default="npu")
    parser.add_argument("--evidence", type=Path, default=Path(__file__).with_name("OPFORGE_EVIDENCE.json"))
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    cases: list[Case] = []
    if args.public:
        cases.extend(load_public_cases(args.evidence))
    if args.random_generalization:
        cases.extend(random_generalization_cases(args.random_generalization, args.random_seed))
    if not cases:
        raise SystemExit("nothing to run; pass --public and/or --random-generalization N")

    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    records = []
    total = len(cases)
    for index, case in enumerate(cases, start=1):
        record = run_case(case, index, total, args.device, args.random_seed)
        records.append(record)
        if args.jsonl:
            with args.jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    passed = sum(1 for r in records if r["triton_accuracy"] == "PASS")
    summary = {
        "status": "PASS" if passed == total else "FAIL",
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "public": sum(1 for r in records if r["kind"] == "public"),
        "random_generalization": sum(1 for r in records if r["kind"] == "random_generalization"),
        "random_seed": args.random_seed,
        "random_policy": RANDOM_GENERALIZATION_POLICY,
        "max_diff": max(float(r["compare"]["max_diff"]) for r in records),
        "max_mare": max(float(r["compare"]["mare"]) for r in records),
        "total_mismatch_count": sum(int(r["compare"]["mismatch_count"]) for r in records),
    }
    print("SUMMARY "
          f"status={summary['status']} total={total} passed={passed} failed={total - passed} "
          f"public={summary['public']} random={summary['random_generalization']} "
          f"max_diff={summary['max_diff']:.6e} max_mare={summary['max_mare']:.6e} "
          f"mismatch={summary['total_mismatch_count']}")
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
