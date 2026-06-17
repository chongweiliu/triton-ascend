# KvRmsNormRopeCache 算子设计文档

## 1. 设计目标

本实现面向 Ascend NPU 上的 BF16 decode cache update 场景，使用 Triton-Ascend 实现 KvRmsNormRopeCache。核心语义为按 `[Dk, Dv]` 拆分 `kv`，前半做 RoPE，后半做 RMSNorm，并按 `index` 写入 cache。

## 2. 接口定义

```python
kv_rms_norm_rope_cache(kv, gamma, cos, sin, index, kCacheRef, ckvCacheRef, epsilon=1e-5, cache_mode="Norm") -> (kCacheRefOut, ckvCacheRefOut)
```

输入均为连续 NPU tensor：`kv/gamma/cos/sin/kCacheRef/ckvCacheRef` 为 BF16，`index` 为 int64。`Dk` 必须为正偶数，`Dv` 必须为正数，`kv[-1] = Dk + Dv`。本实现按公式自洽地使用 `kCacheRef[..., Dk]` 和 `ckvCacheRef[..., Dv]`。

## 3. 形状覆盖与分发

公开验证覆盖文档中的 20 个模型行：`Batch in {1,8,16,32,64}`、`HeadNum in {28,32,40,64}`、`HeadDim=128`，即 public split `Dk=Dv=64`。随机泛化验证在相同文档模型行上变化 `Skv`、`Scache`、`Bcache>=Bkv` 和合法 `Dk/Dv` split。

- `Dk=Dv=64`：使用 `_update_inplace_64_kernel`，保持公开路径 fast path。
- 其他合法 split：使用 `_update_inplace_dynamic_split_kernel`，`BLOCK_KH=next_power_of_2(Dk/2)`，`BLOCK_V=next_power_of_2(Dv)`。
- `max(Dk,Dv)>1024`：backend capacity guard，fail loudly，不 fallback。

## 4. Kernel 设计

kernel grid 为 `(heads, B)`，每个 program 负责一个 `(head, batch)` pair，并在 program 内按 `Skv` 顺序遍历 token。合法 index 写入 cache；`index=-1` 跳过；duplicate index 按源 token 顺序自然保持 last-write-wins。

RoPE 使用 low/high half 连续 load。动态 split 路径与 64/64 fast path 保持同构数据流，避免 all-lane `rope/rot_d` 泛化表达式重复读取 RoPE 行。RMSNorm value 向量单独处理，以 `Dv` lane 做 FP32 sum、rsqrt 和 `gamma` 乘法。

## 5. 精度与性能证据

- delivery 正确性验证 `60/60` 通过，其中 public `20` 个，random dynamic split `40` 个。
- OpForge latest run `eval_20260617_162945`：`20/20` 通过，active/active geomean `77.170667x`，mean latency `77.675 us`。
- 历史 best run `eval_20260616_211136` active geomean `84.559295x`；本交付选择 latest 动态 split 版本，因为它满足当前更宽的 `Dk+Dv` 支持边界。

## 6. 无 fallback 边界

被测函数只使用本地 Triton-Ascend JIT kernel。Python 封装只做元数据校验和 launch，不在 Python 中计算被测结果，也不调用任务 golden/reference、PyTorch 等价实现、`torch_npu` whole op、CANN/vendor whole op、CPU fallback、peer 或其他后端代码。

## 7. 风险说明

当前OpForge public baseline为 `pytorch_fallback`，不能从该证据扩写为对可用 CANN whole-operator 的直接 1.2x 结论。delivery benchmark 为 wall-sync sanity，正式性能引用 OpForge active-window。
