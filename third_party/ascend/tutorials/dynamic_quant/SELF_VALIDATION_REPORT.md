# DynamicQuant 自验证报告

## 结论

- 证据版本：`eval_20260616_233746`。
- public 正确性：40/40 PASS。
- OpForge 状态：`PERF_REGRESSION`。
- 主性能口径：`active_vs_active`。
- active/active geomean：8.154336x。
- active regression cases：4, 8, 17。
- baseline source split：task_npu_baseline=19，pytorch_fallback=21。
- delivery correctness 自跑：80/80 PASS，public=40，random generalization=40，dst_type 覆盖 int8=40、int4=40。
- delivery benchmark 自跑：40/40 PASS，debug host elapsed geomean 122.376 us。

当前包是候选交付材料，不能写成性能门完全通过。`pytorch_fallback` 是评测基线来源，不是 candidate fallback；candidate measured path 始终是 Triton-Ascend kernel。
delivery benchmark 的计时是本地 debug host elapsed，不替代 OpForge profiler active-window 主性能证据。

## 公开用例明细

| Case | Shape | dst_type | Baseline | Triton active | Baseline active | Active speedup | Active 回归 | Mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | (1, 1, 3584) | int8 | task_npu_baseline | 2.750 us | 3.000 us | 1.090909x | 否 | 0 |
| 2 | (1, 1, 4096) | int8 | task_npu_baseline | 2.750 us | 3.500 us | 1.272727x | 否 | 0 |
| 3 | (1, 1, 5120) | int8 | task_npu_baseline | 3.000 us | 3.500 us | 1.166667x | 否 | 0 |
| 4 | (1, 1, 8192) | int8 | task_npu_baseline | 4.250 us | 3.500 us | 0.823529x | 是 | 0 |
| 5 | (8, 1, 3584) | int8 | task_npu_baseline | 3.000 us | 3.250 us | 1.083333x | 否 | 0 |
| 6 | (8, 1, 4096) | int8 | task_npu_baseline | 3.000 us | 4.000 us | 1.333333x | 否 | 0 |
| 7 | (8, 1, 5120) | int8 | task_npu_baseline | 3.500 us | 4.000 us | 1.142857x | 否 | 0 |
| 8 | (8, 1, 8192) | int8 | task_npu_baseline | 4.250 us | 3.250 us | 0.764706x | 是 | 0 |
| 9 | (16, 1, 3584) | int8 | task_npu_baseline | 3.250 us | 3.750 us | 1.153846x | 否 | 0 |
| 10 | (16, 1, 4096) | int8 | task_npu_baseline | 3.250 us | 4.000 us | 1.230769x | 否 | 0 |
| 11 | (16, 1, 5120) | int8 | task_npu_baseline | 3.250 us | 4.000 us | 1.230769x | 否 | 0 |
| 12 | (16, 1, 8192) | int8 | task_npu_baseline | 4.500 us | 4.500 us | 1.000000x | 否 | 0 |
| 13 | (32, 1, 3584) | int8 | pytorch_fallback | 4.000 us | 172.250 us | 43.062500x | 否 | 0 |
| 14 | (32, 1, 4096) | int8 | task_npu_baseline | 4.000 us | 4.750 us | 1.187500x | 否 | 0 |
| 15 | (32, 1, 5120) | int8 | task_npu_baseline | 4.250 us | 4.750 us | 1.117647x | 否 | 0 |
| 16 | (32, 1, 8192) | int8 | task_npu_baseline | 5.250 us | 7.000 us | 1.333333x | 否 | 0 |
| 17 | (64, 1, 3584) | int8 | task_npu_baseline | 5.500 us | 5.250 us | 0.954545x | 是 | 0 |
| 18 | (64, 1, 4096) | int8 | task_npu_baseline | 5.750 us | 5.750 us | 1.000000x | 否 | 0 |
| 19 | (64, 1, 5120) | int8 | task_npu_baseline | 6.000 us | 6.750 us | 1.125000x | 否 | 0 |
| 20 | (64, 1, 8192) | int8 | task_npu_baseline | 8.500 us | 9.500 us | 1.117647x | 否 | 0 |
| 21 | (1, 1, 3584) | int4 | pytorch_fallback | 2.750 us | 194.750 us | 70.818182x | 否 | 0 |
| 22 | (1, 1, 4096) | int4 | pytorch_fallback | 2.750 us | 208.250 us | 75.727273x | 否 | 0 |
| 23 | (1, 1, 5120) | int4 | pytorch_fallback | 3.000 us | 192.750 us | 64.250000x | 否 | 0 |
| 24 | (1, 1, 8192) | int4 | pytorch_fallback | 4.000 us | 180.250 us | 45.062500x | 否 | 0 |
| 25 | (8, 1, 3584) | int4 | pytorch_fallback | 3.000 us | 212.250 us | 70.750000x | 否 | 0 |
| 26 | (8, 1, 4096) | int4 | pytorch_fallback | 2.750 us | 162.000 us | 58.909091x | 否 | 0 |
| 27 | (8, 1, 5120) | int4 | pytorch_fallback | 3.000 us | 212.250 us | 70.750000x | 否 | 0 |
| 28 | (8, 1, 8192) | int4 | pytorch_fallback | 4.250 us | 231.500 us | 54.470588x | 否 | 0 |
| 29 | (16, 1, 3584) | int4 | pytorch_fallback | 3.250 us | 182.250 us | 56.076923x | 否 | 0 |
| 30 | (16, 1, 4096) | int4 | pytorch_fallback | 3.250 us | 169.500 us | 52.153846x | 否 | 0 |
| 31 | (16, 1, 5120) | int4 | pytorch_fallback | 3.500 us | 191.750 us | 54.785714x | 否 | 0 |
| 32 | (16, 1, 8192) | int4 | pytorch_fallback | 4.500 us | 185.000 us | 41.111111x | 否 | 0 |
| 33 | (32, 1, 3584) | int4 | pytorch_fallback | 4.000 us | 202.750 us | 50.687500x | 否 | 0 |
| 34 | (32, 1, 4096) | int4 | pytorch_fallback | 4.000 us | 234.000 us | 58.500000x | 否 | 0 |
| 35 | (32, 1, 5120) | int4 | pytorch_fallback | 4.000 us | 194.500 us | 48.625000x | 否 | 0 |
| 36 | (32, 1, 8192) | int4 | pytorch_fallback | 5.250 us | 205.750 us | 39.190476x | 否 | 0 |
| 37 | (64, 1, 3584) | int4 | pytorch_fallback | 5.500 us | 219.000 us | 39.818182x | 否 | 0 |
| 38 | (64, 1, 4096) | int4 | pytorch_fallback | 5.750 us | 191.000 us | 33.217391x | 否 | 0 |
| 39 | (64, 1, 5120) | int4 | pytorch_fallback | 5.750 us | 219.000 us | 38.086956x | 否 | 0 |
| 40 | (64, 1, 8192) | int4 | pytorch_fallback | 8.500 us | 192.750 us | 22.676471x | 否 | 0 |

## 基线来源说明

`task_npu_baseline` 表示校准时使用任务 NPU helper 并通过语义校验；`pytorch_fallback` 表示任务校准使用 semantic PyTorch 路径作为基线。DynamicQuant 中 20 个 logical INT4 unpacked-storage case 没有稳定的公开 `torch_npu.npu_dynamic_quant` baseline；另外 case13 的 INT8 helper 校验出现 `max_diff=1`，因此也被标为 `pytorch_fallback`。

## 日志证据

- `logs/opforge_eval_20260616_233746_summary.json`
- `logs/opforge_eval_20260616_233746_traces.jsonl`
- `logs/opforge_eval_20260616_233746_main_results.csv`
- `logs/opforge_eval_20260616_233746_timing_summary.json`
- `logs/opforge_dynamic_quant_golden_baseline.json`
- `logs/dynamic_quant_validation_20260617_155614.log`
- `logs/dynamic_quant_validation_20260617_155614.summary.json`
- `logs/dynamic_quant_benchmark_20260617_155312.log`
- `logs/dynamic_quant_benchmark_20260617_155312.summary.json`
