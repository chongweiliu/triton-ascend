# AddRmsNorm Operator Design

## 1. Objective

Implement AddRmsNorm with Triton-Ascend for BF16 inference tensors on Ascend
NPU. The design follows the source requirement:

```text
z = x1 + x2
yOut = z * rsqrt(mean(z * z, axis=-1, keepdims=True) + epsilon) * gamma
```

Only `yOut` is produced. CANN API auxiliary outputs such as `xOut` and
`rstdOut` are not part of this deliverable.

## 2. Interface

```python
add_rms_norm(x1: Tensor, x2: Tensor, gamma: Tensor, epsilon: float = 1e-6) -> Tensor
```

Input constraints:

| Tensor | Dtype | Layout | Shape |
|---|---|---|---|
| `x1` | BF16 | contiguous ND | `[B, S, H]` |
| `x2` | BF16 | contiguous ND | `[B, S, H]` |
| `gamma` | BF16 | contiguous ND | `[B, S, H]` |

The output `yOut` is BF16 contiguous ND with the same shape.

## 3. Shape Coverage

The source requirement lists 20 mainstream inference combinations:

- `B in {1, 8, 16, 32, 64}`
- `H in {3584, 4096, 5120, 8192}`

The validation material expands those documented combinations across
`S in {1, 8, 32, 128}`. The implementation itself does not hard-code the public
case ids or public workload filenames. Its runtime acceptance guard is:

- rank-3 contiguous BF16 NPU tensors
- identical input shapes
- positive `epsilon`
- `block_h = next_power_of_2(H) <= 8192`

## 4. Kernel Design

The logical tensor is flattened into `n_rows = B * S` independent rows, each
with `H` elements. Each Triton program handles one row.

For `H <= 4096`, a single fused kernel:

1. loads `x1`, `x2`, and `gamma`
2. computes `z = x1 + x2` in FP32
3. reduces `sum(z*z)` over the hidden dimension
4. computes `rstd`
5. writes `z * rstd * gamma` as BF16

For `4096 < H` and `next_power_of_2(H) <= 8192`, the implementation uses two
Triton kernels:

1. compute one FP32 `rstd` value per row
2. reload `x1`, `x2`, and `gamma`, then write `yOut`

This split avoids the fused 8192-wide temporary footprint that can exceed
Triton-Ascend/BiShengIR on-chip limits.

For wider hidden sizes where `next_power_of_2(H) > 8192`, the implementation
uses a chunked same-backend path:

1. compute FP32 partial sums over 8192-element hidden chunks
2. reduce chunk sums into one FP32 `rstd` value per row
3. apply `rstd` to each chunk and write `yOut`

This path was added after generalization audit cases around `H=8320` exposed
the previous single-program hidden-size limit.

## 5. Precision

The kernel accumulates in FP32 and stores BF16 output. The validation script
checks the BF16 L1-style relative-error criterion:

- BF16 threshold: `2^-7 = 0.0078125`
- pass condition: `MERE < threshold` and `MARE < 10 * threshold`

OpForge CANN-Bench evidence for `eval_20260612_155910` shows:

- `80/80` public cases accuracy-passed
- accuracy failed cases: `0`
- total mismatch count: `0`
- max MARE: `0.007812499609375021`
- max diff: `0.015625`

## 6. No-Fallback Boundary

The measured AddRmsNorm function uses local Triton-Ascend JIT kernels only.
The wrapper uses Python for validation, output/workspace allocation, and kernel
launch metadata. It does not compute the operator in Python and does not call:

- task golden/reference code
- PyTorch AddRmsNorm-equivalent math in the measured function
- `torch_npu.npu_add_rms_norm`
- CANN/vendor AddRmsNorm
- CPU fallback
- peer or other-backend solution code

## 7. Current Performance Evidence

Structured evidence is copied in `OPFORGE_EVIDENCE.json`.

Current OpForge public result:

| Metric | Value |
|---|---:|
| Run id | `eval_20260612_155910` |
| Public cases | `80` |
| Correct | `80` |
| Failed | `0` |
| Overall calibrated geomean speedup | `8.269904x` |
| Minimum speedup | `0.485075x` |
| Status | `PERF_REGRESSION` |

Manual generalization audit:

| Metric | Value |
|---|---:|
| Audit run id | `manual_generalization_audit_40_fixed_20260612_155512` |
| Audit cases | 40 |
| Failed | 0 |
| Status | passed |

Baseline provenance must be stated with the result:

| Baseline source | Cases | Geomean speedup |
|---|---:|---:|
| `torch_npu.npu_add_rms_norm` task baseline | 21 | `3.356045x` |
| PyTorch semantic fallback baseline | 59 | `11.400119x` |

The public result proves correctness and shows strong calibrated aggregate
speedup, but it should not be phrased as 80/80 cases being faster than CANN
AddRmsNorm because 59 baselines were PyTorch fallback baselines.

## 8. Risks And Next Work

- Re-run validation in the final HiDevLab/Triton-Ascend environment and attach
  the raw logs/screenshots required by the review process.
- Improve small-shape performance; the current OpForge status is
  `PERF_REGRESSION` because several cases are below the performance threshold.
- Continue optimizing the small-shape path; cases below 1x are the current
  blocker for a clean `PASSED` public status.
