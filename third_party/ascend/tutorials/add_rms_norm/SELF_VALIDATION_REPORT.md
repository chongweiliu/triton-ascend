# AddRmsNorm Self-Validation Report

## 1. Environment

Evidence source: OpForge CANN-Bench evaluation on the local Ascend NPU
workspace.

| Item | Value |
|---|---|
| Run id | `eval_20260612_155910` |
| Backend | Triton-Ascend |
| Chip | Ascend910_9382 |
| CANN | 9.0.0 |
| Driver | 25.5.2 |
| Python | 3.11.14 |
| PyTorch | 2.10.0+cpu |
| torch_npu | 2.10.0 |
| Assigned physical NPU | 7 |

Primary local artifacts:

- `/mnt/model/lcw/SLAI-Ascend-OpForge/runs/CANNBench_Custom_add_rms_norm/agents/triton-ascend/results/eval_records/eval_20260612_155910/summary.json`
- `/mnt/model/lcw/SLAI-Ascend-OpForge/runs/CANNBench_Custom_add_rms_norm/agents/triton-ascend/results/eval_records/eval_20260612_155910/traces.jsonl`
- `/mnt/model/lcw/SLAI-Ascend-OpForge/runs/CANNBench_Custom_add_rms_norm/agents/triton-ascend/results/cann_bench_reports/eval_20260612_155910.md`
- `/mnt/model/lcw/SLAI-Ascend-OpForge/runs/CANNBench_Custom_add_rms_norm/agents/triton-ascend/results/eval_records/manual_generalization_audit_40_fixed_20260612_155512/generalization_audit.json`

## 2. Test Scope

The public validation set contains 80 cases:

- `B in {1, 8, 16, 32, 64}`
- `S in {1, 8, 32, 128}`
- `H in {3584, 4096, 5120, 8192}`
- dtype: BF16
- input value ranges:
  - `x1`: `[-1.0, 1.0]`
  - `x2`: `[-1.0, 1.0]`
  - `gamma`: `[0.5, 1.5]`
- `epsilon = 1e-6`

## 3. Accuracy Result

BF16 L1-style threshold in the CANN-Bench checker:

- threshold: `2^-7 = 0.0078125`
- normal-value criterion: `MERE < threshold` and `MARE < 10 * threshold`
- small-value/cancellation locations use the checker ErrorCount rules

Observed public result:

| Metric | Value |
|---|---:|
| Public cases | 80 |
| Accuracy passed | 80 |
| Accuracy failed | 0 |
| Total mismatch count | 0 |
| Max MARE | 0.007812499609375021 |
| Max diff | 0.015625 |

Conclusion: the OpForge public CANN-Bench run satisfies the current BF16
accuracy checker on all 80 public cases.

## 4. Performance Result

Overall calibrated public result:

| Metric | Value |
|---|---:|
| Correct cases | 80 |
| Failed cases | 0 |
| Overall calibrated geomean speedup | 8.269904x |
| Mean latency | 56.786125 us |
| Median latency | 15.550000 us |
| P95 latency | 225.166000 us |
| Min speedup | 0.485075x |
| Score | 102.290329 |
| Status | `PERF_REGRESSION` |

Baseline source split:

| Baseline source | Cases | Geomean speedup | Mean speedup | Min speedup | Max speedup |
|---|---:|---:|---:|---:|---:|
| `torch_npu.npu_add_rms_norm` task baseline | 21 | 3.356045x | 5.218323x | 0.485075x | 21.716639x |
| PyTorch semantic fallback baseline | 59 | 11.400119x | 17.899288x | 1.870117x | 63.325991x |

Important: only the 21 `task_npu_baseline` cases are directly calibrated
against the public `torch_npu.npu_add_rms_norm` helper. The remaining 59 cases
used PyTorch semantic fallback baseline because the public NPU baseline did not
validate numerically for those cases.

## 5. No-Fallback Review

The deliverable implementation:

- uses `@triton.jit` kernels for the measured path
- validates metadata in Python
- allocates output and FP32 `rstd` workspace as NPU tensors
- does not call `torch_npu.npu_add_rms_norm`
- does not call CANN/vendor AddRmsNorm
- does not use CPU fallback, task golden code, peer solution code, or workload
  answer data in the measured path

The dispatch guard is implementation-capability based:

- `H <= 4096`: one fused Triton kernel
- `4096 < H` and `next_power_of_2(H) <= 8192`: two-kernel local Triton path
- wider `H`: chunked same-backend partial-sum/reduce/apply path using
  8192-element hidden chunks

The code in this deliverable does not hard-code public case ids or public
workload filenames.

## 6. Generalization Audit

Manual OpForge audit result:

| Metric | Value |
|---|---:|
| Audit run id | `manual_generalization_audit_40_fixed_20260612_155512` |
| Cases | 40 |
| Failed | 0 |
| Status | passed |

## 7. Reproduction Command

Run from `third_party/ascend/tutorials/add_rms_norm`:

```bash
python3 validate_add_rms_norm.py --public --generalization
```

Optional smoke benchmark:

```bash
python3 validate_add_rms_norm.py --public --benchmark --warmup 3 --repeat 5
```

## 8. Current Limitations

- The current OpForge aggregate status is `PERF_REGRESSION`, not `PASSED`,
  because the minimum speedup is below 1.0 on some small cases.
- A final formal submission should attach fresh HiDevLab logs/screenshots and
  performance data generated from this deliverable directory.
