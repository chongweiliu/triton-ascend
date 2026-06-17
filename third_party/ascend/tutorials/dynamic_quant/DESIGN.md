# DynamicQuant 算子设计说明

## 目标与范围

本实现面向 Ascend NPU 上的 Triton-Ascend，覆盖当前任务文档中的 BF16 `dynamicQuant` 单输入路径：输入 `x` 为连续 `[B, S, H]`，沿最后一维按 token 计算 `abs_max`、写出 FP32 `scale`，并根据 `dst_type=int8/int4` 写出 int8 storage 的量化结果。logical INT4 以 unpacked signed int4 值域 `[-8, 7]` 存放在 int8 tensor 中。

## 数学语义

对每个 `[B, S]` 行：

```text
qmax = 127 if dst_type == int8 else 7
qmin = -128 if dst_type == int8 else -8
scale = max(abs(x), axis=-1) / qmax
output = clamp(round(x / scale), qmin, qmax)
```

kernel 中使用 `safe_max=max(abs_max, 1e-12)`，存储 `scale=safe_max/q_abs`，量化阶段使用等价的 reciprocal multiplier `q_abs/safe_max` 避免逐元素除法。对任务生成器覆盖的 finite BF16 输入，`safe_max >= max(abs(x))` 将量化值限制在目标值域内，因此当前路径不额外发出 clamp 指令。

## 调度与分桶

- `hidden <= 8192`：每个 runtime `[B, S]` 行一个 Triton program，`BLOCK_H=hidden`，无 tail lane；`rows > 48` 时使用 48-program row-stride 版本以降低大 batch 行数的 launch/program 压力。
- `hidden > 8192`：单行 program 内使用 `BLOCK_H=4096` 两阶段 chunk loop，先 reduce abs max，再按 chunk 写 output。`hidden % 4096 == 0` 走无 mask loop，否则走 tail-safe mask loop。
- exact、row-stride 和 divisible loop 路径使用 Triton-Ascend 3.2.1 的 `al.multibuffer(x, 2)` 编译提示。

## 边界与无 fallback

入口检查 rank-3、BF16、contiguous、NPU tensor、正维度以及 `dst_type in {int8,int4}`。unsupported 输入会 fail loudly。被测 `dynamic_quant` 不调用 PyTorch、torch_npu、CANN/vendor DynamicQuant、CPU 路径、任务 golden 或其他 backend。

## 证据摘要

选定版本为 `eval_20260616_233746`。该版本 public 40/40 正确，`active_vs_active.geomean=8.154336x`，但状态仍为 `PERF_REGRESSION`，active regression cases 为 4, 8, 17。基线来源拆分为 task_npu_baseline=19、pytorch_fallback=21。
