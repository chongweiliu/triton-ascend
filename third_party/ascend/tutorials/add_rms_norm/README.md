# AddRmsNorm Triton-Ascend

本目录存放 AddRmsNorm 算子的 Triton-Ascend 交付材料，内容按照 OpForge 工作区
`docs/addRmsNorm.md` 的要求整理。

## 需求概述

- 后端：Ascend NPU 上的 Triton-Ascend。
- 算子：`addRmsNorm`。
- 输入：
  - `x1`：BF16，ND 布局，形状 `[B, S, H]`
  - `x2`：BF16，ND 布局，形状 `[B, S, H]`
  - `gamma`：BF16，ND 布局，形状 `[B, S, H]`
  - `epsilon`：标量，默认值 `1e-6`
- 输出：
  - `yOut`：BF16，ND 布局，形状 `[B, S, H]`
- 计算公式：

```text
z = x1 + x2
yOut = z * rsqrt(mean(z * z, axis=-1, keepdims=True) + epsilon) * gamma
```

原始需求列出了 20 组主流推理场景下的 `B/H` 组合：`B in {1, 8, 16, 32, 64}`，
`H in {3584, 4096, 5120, 8192}`。验证脚本在这些组合基础上扩展
`S in {1, 8, 32, 128}`，形成 80 个公开风格测试用例，并额外加入若干非公开形状
用于泛化检查。

## 文件说明

- `add_rms_norm.py`：Triton-Ascend kernel 及 Python 调用封装。
- `validate_add_rms_norm.py`：功能验证脚本，并支持可选的端到端耗时冒烟测试。
- `DESIGN.md`：算子设计文档。
- `SELF_VALIDATION_REPORT.md`：基于 OpForge CANN-Bench 证据整理的自验证报告。
- `OPFORGE_EVIDENCE.json`：当前 OpForge 评测证据的结构化副本。

## 运行验证

在当前目录执行：

```bash
python3 validate_add_rms_norm.py --public --generalization
```

可选的冒烟性能采集命令：

```bash
python3 validate_add_rms_norm.py --public --benchmark --warmup 3 --repeat 5
```

验证脚本需要可用的 `torch`、`torch_npu`、Triton-Ascend 以及 Ascend NPU 环境。
脚本会将 Triton-Ascend 输出与 PyTorch 语义参考结果进行 BF16 L1 精度口径对比。

## 实现说明

Python 封装负责校验 rank、dtype、连续性、device、形状一致性和正数 `epsilon`。
实际分发基于运行时元数据和 Triton-Ascend kernel 能力边界：

- `block_h = triton.next_power_of_2(H)`
- `H <= 4096`：单个融合 Triton kernel
- `H > 4096`：两个 Triton kernel，并使用本实现分配的 FP32 `rstd` 工作区
- `next_power_of_2(H) > 8192`：分块 partial-sum 归约加分块 apply kernel，用于
  `H=8320` 等更宽 hidden size 场景

实现不会根据 case id、workload 文件名、公开输入值、观测输出或计时特征进行分支。
被测函数内不调用 PyTorch 等价计算、`torch_npu.npu_add_rms_norm`、
CANN/vendor AddRmsNorm、CPU 代码或任务 golden 路径。

## 当前证据

OpForge CANN-Bench 公开评测证据：

- 运行编号：`eval_20260612_155910`
- 公开用例：`80/80` 正确
- 精度失败用例：`0`
- BF16 阈值：`2^-7 = 0.0078125`
- 整体校准 geomean speedup：`8.269904x`
- 最小 speedup：`0.485075x`
- 状态：`PERF_REGRESSION`
- 手动泛化审计：
  `manual_generalization_audit_40_fixed_20260612_155512`，40/40 通过

性能基线需要单独说明：80 个公开用例中，21 个用例使用
`torch_npu.npu_add_rms_norm` 作为校准基线；其余 59 个用例使用 PyTorch 语义
fallback 基线，因为公开 NPU baseline 在这些用例上未能通过数值校验。因此不能表述为
“80 个用例全部快于 CANN AddRmsNorm”，应使用 `SELF_VALIDATION_REPORT.md` 中的
分组性能数据。

## 已知限制

- 分块路径通过将 hidden 维拆成 8192 元素块来支持更宽的 `H`。极大的 `H` 仍可能受
  工作区大小或 chunk 数量上限约束。
- 当前 OpForge 公开状态为 `PERF_REGRESSION`，不是 `PASSED`，原因是部分小尺寸用例
  仍慢于校准基线。
