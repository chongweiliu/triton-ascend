# KvRmsNormRopeCache Triton-Ascend

本目录存放 KvRmsNormRopeCache 算子的 Triton-Ascend 交付材料，代码版本选用当前动态 `Dk/Dv` 实现：公开 `64/64` split 走专用 fast path，其他合法 split 走 generic dynamic path。

## 文件说明

- `kv_rms_norm_rope_cache.py`：Triton-Ascend kernel 及 Python 调用封装。
- `validate_kv_rms_norm_rope_cache.py`：public 和 fixed-seed random dynamic split 验证脚本。
- `DESIGN.md`：算子设计说明。
- `SELF_VALIDATION_REPORT.md`：自验证报告、OpForge public 性能明细和随机泛化明细。
- `KvRmsNormRopeCache算子设计方案.docx`：设计方案模板填充件。
- `KvRmsNormRopeCache算子自验证报告.xlsx`：自验证表、逐 case 性能、随机泛化和日志证据。
- `OPFORGE_EVIDENCE.json`：结构化 OpForge 证据副本。
- `logs/kv_rms_norm_rope_cache_validation_20260617.*`：20 public + 40 random dynamic split 正确性验证日志。
- `logs/kv_rms_norm_rope_cache_benchmark_20260617.*`：20 public delivery wall-sync timing sanity 日志。

## 运行验证

```bash
source /mnt/model/lcw/SLAI-Ascend-OpForge/scripts/source_cann9.sh
export ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_VISIBLE_DEVICES=1
export ASCEND_DEVICE_ID=0
export NPU_ID=0
python3 validate_kv_rms_norm_rope_cache.py --public --random-generalization 40 --random-seed 20260617
python3 validate_kv_rms_norm_rope_cache.py --public --benchmark --warmup 3 --repeat 5
```

## 当前证据

| 指标 | 数值 |
| --- | --- |
| delivery 版本 | 当前动态 `Dk/Dv` 实现 |
| latest OpForge run id | `eval_20260617_162945` |
| latest OpForge 状态 | `PASSED` |
| public 正确性 | `20/20 PASS` |
| active/active geomean | `77.170667x` |
| min active speedup | `37.517355x` |
| candidate active mean | `77.675 us` |
| baseline source split | `pytorch_fallback=20` |
| dynamic audit | `40/40 PASS, seed 20260617` |
| delivery correctness | `60/60 PASS; public 20, random 40` |
| delivery benchmark sanity | `20/20 PASS; wall-sync geomean 207.255 us` |
| best historical public run | `eval_20260616_211136`, active geomean `84.559295x` |

正式性能主证据使用 OpForge active-window。delivery benchmark 是本目录脚本的 wall-sync sanity，不替代 OpForge 计分。当前 public baseline 在 OpForge 中为 `pytorch_fallback`，因此不能把这些数字表述为对可用 CANN whole-operator 的直接 1.2x 结论。

## 实现边界

实现只按 runtime metadata 分发：dtype、rank、contiguity、shape、`Dk/Dv`、`epsilon`、`cache_mode` 和 backend capacity。被测函数内不调用任务 golden/reference、PyTorch 等价实现、`torch_npu` high-level whole op、CANN/vendor whole op、CPU fallback、peer 或其他后端代码。
