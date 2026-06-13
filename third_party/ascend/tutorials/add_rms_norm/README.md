# AddRmsNorm Triton-Ascend

本目录存放 AddRmsNorm 算子的 Triton-Ascend 交付材料，内容按当前私有 OpForge/CANN-Bench 口径整理。

## 需求概述

- 后端：Ascend NPU 上的 Triton-Ascend。
- 算子：`addRmsNorm`。
- 输入：`x1`、`x2`、`gamma` 均为 BF16 连续 ND 张量，形状 `[B, S, H]`。
- 输出：`yOut`，BF16 连续 ND 张量，形状 `[B, S, H]`。
- 公式：`z = x1 + x2; yOut = z * rsqrt(mean(z * z, axis=-1, keepdims=True) + epsilon) * gamma`。

公开风格验证覆盖 `B in {1, 8, 16, 32, 64}`、`S in {1, 8, 32, 128}`、`H in {3584, 4096, 5120, 8192}`，共 80 个 case。

## 文件说明

- `add_rms_norm.py`：Triton-Ascend kernel 及 Python 调用封装。
- `validate_add_rms_norm.py`：功能验证脚本，并支持 CANN-Bench 对齐的三路双时间采集。
- `DESIGN.md`：算子设计说明。
- `SELF_VALIDATION_REPORT.md`：自验证报告和 80 case 明细。
- `AddRmsNorm算子设计方案.docx`：设计方案模板填充件。
- `AddRmsNorm算子自验证报告.xlsx`：80 case 自验证表。
- `OPFORGE_EVIDENCE.json`：结构化证据副本。
- `logs/add_rms_norm_timing_matrix_20260613.jsonl`：当前规范三路 timing matrix。

## 运行验证

```bash
python3 validate_add_rms_norm.py --public --generalization
```

三路性能采集命令：

```bash
python3 validate_add_rms_norm.py --public --benchmark --warmup 3 --repeat 5 \
  --jsonl logs/add_rms_norm_timing_matrix_$(date +%Y%m%d_%H%M%S).jsonl \
  --profiler-data-dir logs/prof_data_timing_matrix
```

当前脚本不保留旧 profiler CSV 兼容路径：`kernel_details.csv` 缺 `Start Time(us)`、缺 `Step Id` 或存在空 `Step Id` 时直接报错，避免把不完整 profiler 数据继续纳入计分或报告。

## 计时口径

当前主比较使用 **active-window**：每个 Step Id 内可见 NPU kernel 的 `max(end)-min(start)`，case 时间取 measured step 的中位数。`kernel-sum` 仍保留为诊断：按 kernel 名称聚合每步 duration 后取中位并求和。

三路都报告两个时间：

| 路径 | active-window | kernel-sum | 用途 |
|---|---|---|---|
| Triton 候选 | `opforge_probe.kernel_details.active_window_us` | `opforge_probe.kernel_details.kernel_sum_us` | active-window 是主口径 |
| Torch 语义实现 | `baseline_active_window.device_active_window_us` | `baseline_active_window.device_kernel_duration_sum_us` | active-window 是主基线 |
| `torch_npu.npu_add_rms_norm` helper | `baseline_active_window.device_active_window_us` | `baseline_active_window.device_kernel_duration_sum_us` | 只在精度通过子集作为 NPU helper 基线 |

## 当前证据

OpForge 评测器记录 `eval_20260612_155910`：80/80 正确，状态 `PERF_REGRESSION`，手动泛化审计 `manual_generalization_audit_40_fixed_20260612_155512` 为 40/40 通过。该历史 summary 是 active-window 重构前生成的字段，交付报告中的当前比较以 `logs/add_rms_norm_timing_matrix_20260613.*` 为准。

| 指标 | 数值 |
| --- | --- |
| Triton 候选 active-window geomean | 47.855 us |
| Triton 候选 kernel-sum geomean | 20.180 us |
| Torch 语义 active-window geomean | 343.639 us |
| Torch 语义 kernel-sum geomean | 106.451 us |
| torch_npu helper active-window geomean | 209.043 us |
| torch_npu helper kernel-sum geomean | 209.136 us |
| vs Torch 主加速比 active/active | 7.180762x |
| vs Torch 诊断加速比 kernel/kernel | 5.275206x |
| vs Torch 诊断 baseline-active/candidate-kernel | 17.029071x |
| vs Torch 诊断 baseline-kernel/candidate-active | 2.224432x |
| vs torch_npu PASS 子集主加速比 active/active | 3.271697x |
| vs torch_npu PASS 子集诊断加速比 kernel/kernel | 9.813551x |
| 选中基线主加速比 active/active | 4.716968x |
| 选中基线诊断加速比 kernel/kernel | 8.498617x |

`torch_npu` helper 通过 61/80 个 case；未通过的 19 个 case 不作为 NPU helper 精度通过子集统计，选中基线规则为 `torch_npu` 通过则选 `torch_npu`，否则选 Torch 语义实现。

## 实现边界

实现根据运行时 dtype/rank/shape/contiguity 和 hidden 维能力边界选择路径，不根据 case id、workload 文件名、公开输入值、观测输出或计时特征分支。被测函数内不调用 PyTorch 等价计算、`torch_npu.npu_add_rms_norm`、CANN/vendor AddRmsNorm、CPU 代码或任务 golden 路径。
