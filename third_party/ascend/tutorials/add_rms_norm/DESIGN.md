# AddRmsNorm 算子设计文档

## 1. 设计目标

本实现面向 Ascend NPU 上的 BF16 推理张量，使用 Triton-Ascend 实现 AddRmsNorm
算子。算子语义遵循需求定义：

```text
z = x1 + x2
yOut = z * rsqrt(mean(z * z, axis=-1, keepdims=True) + epsilon) * gamma
```

本交付件只输出 `yOut`。CANN API 中可能存在的 `xOut`、`rstdOut` 等辅助输出不在
本次交付范围内。

## 2. 接口定义

```python
add_rms_norm(x1: Tensor, x2: Tensor, gamma: Tensor, epsilon: float = 1e-6) -> Tensor
```

输入约束：

| Tensor | 数据类型 | 布局 | 形状 |
|---|---|---|---|
| `x1` | BF16 | 连续 ND | `[B, S, H]` |
| `x2` | BF16 | 连续 ND | `[B, S, H]` |
| `gamma` | BF16 | 连续 ND | `[B, S, H]` |

输出 `yOut` 为 BF16 连续 ND 张量，形状与输入一致。

## 3. 形状覆盖

原始需求列出了 20 组主流推理组合：

- `B in {1, 8, 16, 32, 64}`
- `H in {3584, 4096, 5120, 8192}`

验证材料在上述组合基础上扩展 `S in {1, 8, 32, 128}`。实现本身不硬编码公开
case id，也不读取公开 workload 文件名。运行时接受条件为：

- rank-3 连续 BF16 NPU 张量
- 三个输入形状完全一致
- `epsilon` 为正数
- hidden 维基于 Triton-Ascend 能力边界选择不同实现路径

## 4. Kernel 设计

逻辑张量被展平成 `n_rows = B * S` 个独立行，每行包含 `H` 个元素。每个 Triton
program 处理一行或一行中的一个 hidden 分块。

当 `H <= 4096` 时，使用单个融合 kernel：

1. 加载 `x1`、`x2` 和 `gamma`
2. 在 FP32 中计算 `z = x1 + x2`
3. 沿 hidden 维归约 `sum(z*z)`
4. 计算 `rstd`
5. 将 `z * rstd * gamma` 以 BF16 写回

当 `4096 < H` 且 `next_power_of_2(H) <= 8192` 时，使用两个 Triton kernel：

1. 每行计算一个 FP32 `rstd`
2. 重新加载 `x1`、`x2` 和 `gamma`，写出 `yOut`

该拆分避免了 8192 宽融合 kernel 中间临时量过大，降低 Triton-Ascend/BiShengIR
片上资源压力。

当 `next_power_of_2(H) > 8192` 时，使用同后端分块路径：

1. 以 8192 个 hidden 元素为一块计算 FP32 partial sum
2. 将每行的 chunk sum 归约为一个 FP32 `rstd`
3. 对每个 chunk 应用 `rstd` 并写出 `yOut`

该路径是在泛化审计暴露 `H=8320` 附近的单 program hidden-size 限制后加入的。

## 5. 精度设计

kernel 使用 FP32 累加，并以 BF16 存储输出。验证脚本采用 BF16 L1 风格的相对误差
口径：

- BF16 阈值：`2^-7 = 0.0078125`
- 通过条件：`MERE < threshold` 且 `MARE < 10 * threshold`

OpForge CANN-Bench `eval_20260612_155910` 证据显示：

- `80/80` 公开用例精度通过
- 精度失败用例数：`0`
- 总 mismatch 数：`0`
- 最大 MARE：`0.007812499609375021`
- 最大 diff：`0.015625`

## 6. 无 fallback 边界

被测 AddRmsNorm 函数只使用本地 Triton-Ascend JIT kernel。Python 封装仅负责元数据
校验、输出和工作区分配、kernel launch 参数组织，不在 Python 中计算算子结果，也不调用：

- 任务 golden/reference 代码
- 被测函数内的 PyTorch AddRmsNorm 等价计算
- `torch_npu.npu_add_rms_norm`
- CANN/vendor AddRmsNorm
- CPU fallback
- peer 或其他后端解法代码

## 7. 当前性能证据

结构化证据已复制到 `OPFORGE_EVIDENCE.json`。

当前 OpForge 公开评测结果：

| 指标 | 数值 |
|---|---:|
| 运行编号 | `eval_20260612_155910` |
| 公开用例数 | `80` |
| 正确用例数 | `80` |
| 失败用例数 | `0` |
| 整体校准 geomean speedup | `8.269904x` |
| 最小 speedup | `0.485075x` |
| 状态 | `PERF_REGRESSION` |

手动泛化审计：

| 指标 | 数值 |
|---|---:|
| 审计运行编号 | `manual_generalization_audit_40_fixed_20260612_155512` |
| 审计用例数 | 40 |
| 失败用例数 | 0 |
| 状态 | passed |

性能结果必须同时说明基线来源：

| 基线来源 | 用例数 | Geomean speedup |
|---|---:|---:|
| `torch_npu.npu_add_rms_norm` 任务 NPU 基线 | 21 | `3.356045x` |
| PyTorch 语义 fallback 基线 | 59 | `11.400119x` |

公开结果证明了 80 个用例的正确性，并展示了较高的整体校准加速比；但不能表述为
“80/80 用例均快于 CANN AddRmsNorm”，因为其中 59 个用例的基线来源是 PyTorch
语义 fallback。

交付目录内的 `validate_add_rms_norm.py --public --benchmark --warmup 3 --repeat 5`
还会重测 Triton 候选、PyTorch 语义实现和 `torch_npu.npu_add_rms_norm` helper
三路数据。该脚本按路径复刻 CANN-Bench 计时方式：候选 Triton 使用
`KernelDetailsStrategy` 的 `kernel_details.total_kernel_us`，Torch 语义实现和
`torch_npu` helper 使用 custom baseline 的 `BaselineActiveWindowStrategy`，
即 `baseline_active_window.device_active_window_us`。本次交付自验证 JSONL
显示：Triton 候选 80/80 通过，Torch 语义实现 80/80 通过，`torch_npu`
helper 61/80 通过；Speedup vs Torch 语义实现 geomean 为 `17.232065x`，
Speedup vs torch_npu helper 在 helper 精度通过的 61 个用例上 geomean 为
`9.963093x`。

## 8. 风险与后续工作

- 在最终 HiDevLab/Triton-Ascend 环境中重新运行验证，并按评审流程补充原始日志和截图。
- 继续优化小尺寸性能。当前 OpForge 状态为 `PERF_REGRESSION`，原因是部分小尺寸用例
  低于性能阈值。
- 当前 clean `PASSED` 状态的主要阻塞点是低于 1x 的小尺寸用例。
