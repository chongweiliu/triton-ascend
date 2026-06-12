# AddRmsNorm 自验证报告

## 1. 环境信息

证据来源为本地 Ascend NPU 工作区中的 OpForge CANN-Bench 评测结果。

| 项目 | 数值 |
|---|---|
| 运行编号 | `eval_20260612_155910` |
| 后端 | Triton-Ascend |
| 芯片 | Ascend910_9382 |
| CANN | 9.0.0 |
| Driver | 25.5.2 |
| Python | 3.11.14 |
| PyTorch | 2.10.0+cpu |
| torch_npu | 2.10.0 |
| 分配物理 NPU | 7 |

主要本地证据文件：

- `/mnt/model/lcw/SLAI-Ascend-OpForge/runs/CANNBench_Custom_add_rms_norm/agents/triton-ascend/results/eval_records/eval_20260612_155910/summary.json`
- `/mnt/model/lcw/SLAI-Ascend-OpForge/runs/CANNBench_Custom_add_rms_norm/agents/triton-ascend/results/eval_records/eval_20260612_155910/traces.jsonl`
- `/mnt/model/lcw/SLAI-Ascend-OpForge/runs/CANNBench_Custom_add_rms_norm/agents/triton-ascend/results/cann_bench_reports/eval_20260612_155910.md`
- `/mnt/model/lcw/SLAI-Ascend-OpForge/runs/CANNBench_Custom_add_rms_norm/agents/triton-ascend/results/eval_records/manual_generalization_audit_40_fixed_20260612_155512/generalization_audit.json`

## 2. 测试范围

公开验证集包含 80 个用例：

- `B in {1, 8, 16, 32, 64}`
- `S in {1, 8, 32, 128}`
- `H in {3584, 4096, 5120, 8192}`
- 数据类型：BF16
- 输入取值范围：
  - `x1`：`[-1.0, 1.0]`
  - `x2`：`[-1.0, 1.0]`
  - `gamma`：`[0.5, 1.5]`
- `epsilon = 1e-6`

## 3. 精度结果

CANN-Bench checker 中的 BF16 L1 风格阈值：

- 阈值：`2^-7 = 0.0078125`
- 普通数值位置判定：`MERE < threshold` 且 `MARE < 10 * threshold`
- 小值或抵消位置使用 checker 的 ErrorCount 规则

观测到的公开评测结果：

| 指标 | 数值 |
|---|---:|
| 公开用例数 | 80 |
| 精度通过用例数 | 80 |
| 精度失败用例数 | 0 |
| 总 mismatch 数 | 0 |
| 最大 MARE | 0.007812499609375021 |
| 最大 diff | 0.015625 |

结论：当前 OpForge 公开 CANN-Bench 运行在 80 个公开用例上均满足 BF16 精度检查。

## 4. 性能结果

整体公开校准结果：

| 指标 | 数值 |
|---|---:|
| 正确用例数 | 80 |
| 失败用例数 | 0 |
| 整体校准 geomean speedup | 8.269904x |
| 平均耗时 | 56.786125 us |
| 中位耗时 | 15.550000 us |
| P95 耗时 | 225.166000 us |
| 最小 speedup | 0.485075x |
| Score | 102.290329 |
| 状态 | `PERF_REGRESSION` |

按基线来源拆分：

| 基线来源 | 用例数 | Geomean speedup | 平均 speedup | 最小 speedup | 最大 speedup |
|---|---:|---:|---:|---:|---:|
| `torch_npu.npu_add_rms_norm` 任务 NPU 基线 | 21 | 3.356045x | 5.218323x | 0.485075x | 21.716639x |
| PyTorch 语义 fallback 基线 | 59 | 11.400119x | 17.899288x | 1.870117x | 63.325991x |

需要注意：只有 21 个 `task_npu_baseline` 用例是直接对齐公开
`torch_npu.npu_add_rms_norm` helper 的校准结果。其余 59 个用例使用 PyTorch 语义
fallback 基线，因为公开 NPU baseline 在这些用例上未能通过数值校验。

## 5. 无 fallback 检查

本交付实现：

- 被测路径使用 `@triton.jit` kernel
- Python 侧只进行元数据校验
- 输出张量和 FP32 `rstd` 工作区均分配在 NPU 上
- 不调用 `torch_npu.npu_add_rms_norm`
- 不调用 CANN/vendor AddRmsNorm
- 不使用 CPU fallback、任务 golden 代码、peer 解法代码或 workload 答案数据

分发 guard 基于实现能力边界：

- `H <= 4096`：单个融合 Triton kernel
- `4096 < H` 且 `next_power_of_2(H) <= 8192`：两个 Triton kernel 的本地路径
- 更宽 `H`：使用 8192 hidden 元素分块的同后端 partial-sum/reduce/apply 路径

本交付代码不硬编码公开 case id，也不硬编码公开 workload 文件名。

## 6. 泛化审计

手动 OpForge 审计结果：

| 指标 | 数值 |
|---|---:|
| 审计运行编号 | `manual_generalization_audit_40_fixed_20260612_155512` |
| 用例数 | 40 |
| 失败用例数 | 0 |
| 状态 | passed |

## 7. 复现命令

在 `third_party/ascend/tutorials/add_rms_norm` 目录执行：

```bash
python3 validate_add_rms_norm.py --public --generalization
```

可选的冒烟性能命令：

```bash
python3 validate_add_rms_norm.py --public --benchmark --warmup 3 --repeat 5
```

## 8. 当前限制

- 当前 OpForge 聚合状态为 `PERF_REGRESSION`，不是 `PASSED`，原因是部分小尺寸用例
  最小 speedup 低于 1.0。
- 正式提交时建议补充从交付目录直接运行得到的 HiDevLab 日志、截图和性能数据截图。
