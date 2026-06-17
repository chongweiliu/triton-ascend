# DynamicQuant Triton-Ascend

本目录存放 DynamicQuant 算子的 Triton-Ascend 交付材料，内容按当前 OpForge/CANN-Bench 证据整理。

## 需求概述

- 后端：Ascend NPU 上的 Triton-Ascend。
- 算子：`dynamicQuant` / `DynamicQuant`。
- 输入：`x` 为 BF16 连续 ND 张量，形状 `[B, S, H]`。
- 属性：`dst_type` 支持 `int8` 和 `int4`。
- 输出：`output` 与 `x` 同形状，int8 storage；`dst_type=int4` 时承载 `[-8, 7]` 的 unpacked logical int4 值；`scale` 为 float32 `[B, S]`。

## 文件说明

- `dynamic_quant.py`：Triton-Ascend kernel 及 Python 调用封装。
- `validate_dynamic_quant.py`：public-shape 和固定 seed random generalization 正确性验证脚本。
- `DESIGN.md`：算子设计说明。
- `SELF_VALIDATION_REPORT.md`：自验证报告和 40 public 性能明细。
- `DynamicQuant算子设计方案.docx`：设计方案文档。
- `DynamicQuant算子自验证报告.xlsx`：自验证表、逐 case 性能和日志证据工作表。
- `OPFORGE_EVIDENCE.json`：结构化 OpForge 证据副本。
- `logs/`：当前 OpForge evaluator-owned 证据文件副本。

## 运行验证

```bash
python3 validate_dynamic_quant.py --public --random-generalization 40 --random-seed 20260617 \
  --jsonl logs/dynamic_quant_validation_$(date +%Y%m%d_%H%M%S).jsonl
```

可选 `--benchmark` 只输出本地 debug host elapsed time，不作为正式性能证据。正式性能证据来自 OpForge `eval_20260616_233746` 的 profiler/timing artifacts。

## 当前证据

| 指标 | 数值 |
| --- | --- |
| 选定版本 | `eval_20260616_233746` |
| 版本选择 | latest full-public active-metric source |
| OpForge 状态 | `PERF_REGRESSION` |
| public 正确性 | 40/40 PASS |
| active/active geomean | 8.154336x |
| active/active min | 0.764706x |
| kernel/kernel geomean | 3.899296x |
| active regression cases | 4, 8, 17 |
| baseline source split | task_npu_baseline=19, pytorch_fallback=21 |
| timing source | torch_npu.profiler kernel_details.csv |
| delivery correctness self-run | 80/80 PASS; public=40, random=40, int8=40, int4=40 |
| delivery benchmark self-run | 40/40 PASS; debug host geomean 122.376 us |
| delivery self-run logs | `logs/dynamic_quant_validation_20260617_155614.log`, `logs/dynamic_quant_benchmark_20260617_155312.log` |

注意：当前状态仍为 `PERF_REGRESSION`，不能表述为性能门完全通过；同时 21 个 case 的基线是 semantic PyTorch fallback，不能表述为所有 case 都相对 CANN 官方 DynamicQuant 加速。
`delivery benchmark self-run` 是交付脚本的 debug host elapsed 计时，只作为本地自跑日志；正式性能口径仍以上面的 OpForge profiler active-window 证据为准。

## 实现边界

实现根据运行时 dtype、rank、contiguity、`dst_type`、`B*S` 行数和 hidden 维 capability boundary 选择路径，不根据 case id、workload 文件名、公开输入值、观测输出或计时特征分支。被测函数内不调用 PyTorch 等价计算、`torch_npu.npu_dynamic_quant`、CANN/vendor DynamicQuant、CPU 代码或任务 golden 路径。
