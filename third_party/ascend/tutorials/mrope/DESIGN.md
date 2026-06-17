# MRoPE 算子设计说明

## 1. 功能与输入输出

MRoPE 算子对 BF16 `query`、`key` 的每个 head 前 `rotary_dim` 通道施加 RoPE/MRoPE 旋转，输出同 shape、同 dtype 的 `query_out`、`key_out`。RoPE 使用 rank-1 `positions`；MRoPE 使用 3 行或 4 行 `positions`，并按 `mrope_section` 在 half 旋转维度上选择对应位置行。

## 2. Triton-Ascend 实现

交付源码位于 `mrope.py`。核心实现包含一个 generic Triton-Ascend fallback 和四类优化 pair kernel：RoPE half/default、MRoPE half/default、MRoPE half/interleave rotary64、MRoPE interleaved/default。pair kernel 在一个 launch 中同时处理 query 和 key，减少 launch 数和重复 position/cache 解码。

## 3. 调度策略

full-rotary MRoPE half/default 路径使用 `[64 heads, 64 columns]` tile，按 token/head block 复用同一条 cos/sin 向量；其他 pair 路径使用较小 head block。未命中优化路径但合同允许的元数据会走同 backend generic Triton kernel，不会转向框架 fallback。

## 4. 正确性边界

- `query/key/cos_sin_cache` 必须为 BF16，`positions` 必须为 int64。
- `head_size`、`rotary_dim` 为 32 的倍数，且 `rotary_dim <= head_size`。
- MRoPE rows 只接受 3/4，且必须与 `mrope_section` 长度匹配。
- MRoPE `sum(mrope_section) == rotary_dim / 2`。
- `[16,16,16,16]` 只允许 `cache_mode=default`。
- 当前实现要求输入 contiguous；这是 Triton-Ascend 实现 guard，不是文档语义收窄。
- device kernel 不静默 clamp 非法 position；非法 position 会使输出 NaN，从而 fail loudly。

## 5. 性能证据

当前交付采用边界修复后的 OpForge run `eval_20260617_151008`：20/20 PASS，官方主指标 `active_vs_active.geomean=11.657672722257`，min `1.146341463415`，score `100.018424560106`，无 active regression。历史 best `eval_20260616_213635` 略高，但发生在边界 fail-loud 修复之前，不作为当前源码交付主证据。

## 6. Baseline split

baseline calibration 记录为 `task_npu_baseline=8`、`pytorch_fallback=12`。其中 `pytorch_fallback` 是因为 `torch_npu.npu_mrope` 对部分 RoPE/interleaved/cache_interleave 配置报错或与语义 golden 不一致；这些 case 不能宣称为相对有效 CANN `npu_mrope` baseline 的加速。

## 7. 自验证

本目录实际运行 `validate_mrope.py --public --random-generalization 40 --random-seed 20260617`，日志为 `logs/mrope_public_random_20260617.log`，结果 60/60 PASS，mismatch=0。
