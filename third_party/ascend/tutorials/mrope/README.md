# MRoPE Triton-Ascend

本目录存放 MRoPE 算子的 Triton-Ascend 交付材料，按 `/mnt/model/lcw/SLAI-Ascend-OpForge/docs/mrope.md` 和当前 OpForge/CANN-Bench 证据整理。

## 需求概述

- 后端：Ascend NPU 上的 Triton-Ascend。
- 算子：`mrope`，输出 `query_out`、`key_out`。
- 输入：`positions:int64`，`query/key/cos_sin_cache:bfloat16`。
- Shape：RoPE positions `[num_tokens]`，MRoPE positions `[3,num_tokens]` 或 `[4,num_tokens]`；`query/key=[num_tokens, num_heads * head_size]`；`cos_sin_cache=[max_seq_len, rotary_dim]`。
- 模式：`rotary_mode=half/interleaved`，`cache_mode=default/interleave`。

## 文件说明

- `mrope.py`：Triton-Ascend kernel 及 Python 调用封装。
- `validate_mrope.py`：public shape 和固定 seed random generalization 自验证脚本。
- `DESIGN.md`：算子设计说明。
- `SELF_VALIDATION_REPORT.md`：自验证报告、public 性能明细和随机泛化明细。
- `Mrope算子设计方案.docx`：按模板生成的设计方案。
- `Mrope算子自验证报告.xlsx`：按模板生成的自验证报告，含逐 case 速度、随机泛化、日志证据工作表。
- `OPFORGE_EVIDENCE.json`：结构化证据副本。
- `logs/mrope_public_random_20260617.log`：本目录实际运行的 public+random 自验证日志。

## 运行验证

```bash
python3 validate_mrope.py --public --random-generalization 40 --random-seed 20260617 \
  --jsonl logs/mrope_public_random_$(date +%Y%m%d_%H%M%S).jsonl \
  --summary-json logs/mrope_public_random_$(date +%Y%m%d_%H%M%S).summary.json
```

## 当前证据

| 指标 | 数值 |
| --- | --- |
| OpForge run id | `eval_20260617_151008` |
| OpForge 状态 | `PASSED` |
| public 正确性 | 20/20 PASS |
| active/active geomean | 11.657673x |
| active/active min | 1.146341x |
| kernel/kernel geomean | 4.997362x |
| baseline source split | task_npu_baseline=8, pytorch_fallback=12 |
| OpForge generalization audit | 40/40 PASS |
| delivery validation | 60/60 PASS; public=20, random=40 |

注意：当前 20 个 case 中只有 8 个为有效 `task_npu_baseline`，12 个为 `pytorch_fallback`。不能将全部 20 个 case 表述为相对有效 CANN `npu_mrope` baseline 的加速。

## 实现边界

实现根据运行时 dtype/rank/shape/contiguity、`head_size`、`rotary_dim`、positions rank/rows、`mrope_section`、`rotary_mode`、`cache_mode` 选择 Triton-Ascend 路径；不按 case id、workload 文件名、公开输入值、观测输出或计时特征分支。被测路径不调用 PyTorch 等价计算、`torch_npu.npu_mrope`、CANN/vendor whole-op、CPU fallback、任务 golden 或 peer 解法。
