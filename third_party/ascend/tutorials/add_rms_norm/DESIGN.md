# AddRmsNorm 算子设计文档

## 1. 设计目标

本实现面向 Ascend NPU 上的 BF16 推理张量，使用 Triton-Ascend 实现 AddRmsNorm。核心语义为：

```text
z = x1 + x2
yOut = z * rsqrt(mean(z * z, axis=-1, keepdims=True) + epsilon) * gamma
```

本交付件只输出 `yOut`。CANN API 中可能存在的 `xOut`、`rstdOut` 等辅助输出不属于本次交付范围。

## 2. 接口定义

```python
add_rms_norm(x1: Tensor, x2: Tensor, gamma: Tensor, epsilon: float = 1e-6) -> Tensor
```

输入约束：三个输入均为 NPU 上的连续 BF16 三维张量，形状相同，`epsilon` 为正数。输出为 BF16 连续 ND 张量。

## 3. 形状覆盖与分发

公开风格验证覆盖 80 个 case：`B in {1,8,16,32,64}`、`S in {1,8,32,128}`、`H in {3584,4096,5120,8192}`。交付正确性验证还包含固定 seed 随机生成的 40 个非公开 shape case。实现不硬编码公开 case id、随机 case id，也不读取 workload 文件名。

- `H <= 4096`：单个融合 Triton kernel。
- `4096 < H` 且 `next_power_of_2(H) <= 8192`：两个 Triton kernel，先算 FP32 `rstd`，再 apply。
- `next_power_of_2(H) > 8192`：8192 hidden 元素分块 partial-sum/reduce/apply 路径，用于泛化形状。

## 4. Kernel 设计

逻辑张量展平成 `B*S` 行，每行沿 hidden 维做 RMS 归一化。kernel 使用 FP32 累加，输出写回 BF16。双 kernel 和分块路径的中间 `rstd` / partial sum 工作区由本实现分配在 NPU 上。

## 5. 精度设计

验证脚本复刻 BF16 检查口径：阈值 `2^-7 = 0.0078125`，普通位置要求 `MERE < threshold` 且 `MARE < 10 * threshold`，小值或抵消位置使用 checker 的 ErrorCount 规则。`eval_20260612_155910` 显示 80/80 公开 case 精度通过，mismatch 总数为 0，最大 MARE 为 `0.007812499609375021`，最大 diff 为 `0.015625`。

当前交付目录的 `logs/add_rms_norm_random_generalization_20260613.jsonl` 记录了 80 个 public case 加 40 个 fixed-seed random generalization case，Triton 候选为 120/120 通过，其中 random generalization 为 40/40 通过。旧的 10 个手写 near-public/boundary case 已从脚本和报告中删除。

## 6. 无 fallback 边界

被测 AddRmsNorm 函数只使用本地 Triton-Ascend JIT kernel。Python 封装只做元数据校验、输出/工作区分配和 launch 参数组织，不在 Python 中计算被测结果，也不调用任务 golden/reference、PyTorch 等价实现、`torch_npu.npu_add_rms_norm`、CANN/vendor AddRmsNorm、CPU fallback、peer 或其他后端代码。

## 7. 当前性能证据

当前交付性能以 `logs/add_rms_norm_timing_matrix_20260613.jsonl` 为准。主口径为 active-window，kernel-sum 只作为诊断。

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

历史 OpForge 评测器 summary `eval_20260612_155910` 的状态为 `PERF_REGRESSION`，80/80 正确，旧字段中的 geomean speedup 为 `8.269904x`。该字段来自 active-window 重构前的候选 kernel-sum 统计；当前报告同时给出 active-window 和 kernel-sum 两套比较。

## 8. 风险与后续工作

- 小尺寸双 kernel case 的 active-window 受 kernel 间隙影响明显，例如 case 3 的 kernel-sum 为 `6.700 us`，active-window 为 `121.500 us`。
- `torch_npu` helper 只在 61/80 个 case 上通过同一 BF16 精度检查，不能作为 80 case 全覆盖基线。
- random generalization 只作为正确性泛化证据，不替代 80 public timing matrix 的性能统计。
- 当前状态仍为 `PERF_REGRESSION`，后续优化重点是小尺寸 active-window 和双 kernel 路径的 launch/gap。
