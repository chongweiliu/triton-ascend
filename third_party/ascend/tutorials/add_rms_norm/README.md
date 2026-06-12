# AddRmsNorm Triton-Ascend

This directory contains the AddRmsNorm delivery material requested by
`docs/addRmsNorm.md` in the OpForge workspace.

## Requirement Summary

- Backend: Triton-Ascend on Ascend NPU.
- Operator: `addRmsNorm`.
- Inputs:
  - `x1`: BF16, ND, `[B, S, H]`
  - `x2`: BF16, ND, `[B, S, H]`
  - `gamma`: BF16, ND, `[B, S, H]`
  - `epsilon`: scalar, default `1e-6`
- Output:
  - `yOut`: BF16, ND, `[B, S, H]`
- Formula:

```text
z = x1 + x2
yOut = z * rsqrt(mean(z * z, axis=-1, keepdims=True) + epsilon) * gamma
```

The source requirement lists 20 mainstream inference `B/H` combinations:
`B in {1, 8, 16, 32, 64}` and `H in {3584, 4096, 5120, 8192}`. The validation
script expands those combinations across `S in {1, 8, 32, 128}` for an 80-case
public-style sweep and also includes a few non-public shape checks.

## Files

- `add_rms_norm.py`: Triton-Ascend kernel and Python wrapper.
- `validate_add_rms_norm.py`: functional validation and optional wall-clock
  smoke benchmark.
- `DESIGN.md`: design document draft.
- `SELF_VALIDATION_REPORT.md`: self-validation report draft based on OpForge
  CANN-Bench evidence.
- `OPFORGE_EVIDENCE.json`: structured copy of the current OpForge evidence.

## Run Validation

From this directory:

```bash
python3 validate_add_rms_norm.py --public --generalization
```

Optional smoke latency collection:

```bash
python3 validate_add_rms_norm.py --public --benchmark --warmup 3 --repeat 5
```

The script requires `torch`, `torch_npu`, Triton-Ascend, and an available NPU.
It compares the Triton output against a PyTorch semantic reference in BF16 L1
precision terms.

## Implementation Notes

The wrapper validates rank, dtype, contiguity, device, matching shape, and
positive `epsilon`. The dispatch is based on runtime metadata:

- `block_h = triton.next_power_of_2(H)`
- `H <= 4096`: one fused Triton kernel
- `H > 4096`: two Triton kernels with an owned FP32 `rstd` workspace
- `next_power_of_2(H) > 8192`: chunked partial-sum reduction plus chunked
  apply kernel, used for wider hidden sizes such as `H=8320`

The implementation does not branch on case id, workload filename, public input
values, observed outputs, or timing signatures. It does not call PyTorch math,
`torch_npu.npu_add_rms_norm`, CANN/vendor AddRmsNorm, CPU code, or a task
golden path in the measured function.

## Current Evidence

OpForge CANN-Bench public evidence:

- Run: `eval_20260612_155910`
- Public cases: `80/80` correct
- Accuracy failures: `0`
- BF16 threshold: `2^-7 = 0.0078125`
- Overall calibrated geomean speedup: `8.269904x`
- Minimum speedup: `0.485075x`
- Status: `PERF_REGRESSION`
- Manual generalization audit:
  `manual_generalization_audit_40_fixed_20260612_155512`, 40/40 passed

Important performance baseline caveat: 21 cases used
`torch_npu.npu_add_rms_norm` as the calibrated baseline; 59 cases used PyTorch
semantic fallback because the public NPU baseline did not validate numerically
for those cases. Therefore, do not claim all 80 cases are faster than CANN
AddRmsNorm. Use the grouped numbers in `SELF_VALIDATION_REPORT.md`.

## Known Limits

- The chunked path supports wider `H` by splitting the hidden dimension into
  8192-element chunks. Extremely large `H` values can still exceed workspace or
  chunk-count limits.
- The current OpForge public status is `PERF_REGRESSION`, not `PASSED`, because
  some small public cases remain slower than the calibrated baseline.
