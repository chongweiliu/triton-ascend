# MRoPE 算子自验证报告

## 1. 报告说明

- 单一数值证据源：`logs/mrope_validation.jsonl`
- 本报告由当前目录 `generate_delivery.py` 生成，只读取本目录 logs/templates/references。
- L1 是商业精度等级，不是 L1 norm；本表对齐商业标准中的 MARE/MERE/RMSE 指标口径。
- 本报告是本目录自验证，不等同于完整商业 L1 认证；完整商业认证还要求标准规定的用例规模和执行轮次。
- RMSE/MERE/MARE 由本目录 checker 输出。
- 速度门槛按 `torch_npu runnable all` active/active 几何平均 >= 1.2x；无 torch_npu 可计时 case 标记为 N/A，不用 selected baseline 代替。
- 截图证据未外置，日志内容嵌入 XLSX `日志证据` 工作表。
- 不保留、不读取外部历史对照文件。

## 2. 性能总体对比

| 指标 | 数值 |
| --- | --- |
| evidence source | logs/mrope_validation.jsonl |
| total cases | 60 |
| public cases | 20 |
| random/generalization cases | 40 |
| candidate pass | 60/60 |
| main speed sample | torch_npu runnable all (31 cases) |
| main speed candidate active geomean | 11.271 us |
| main speed torch_npu active geomean | 13.765 us |
| main speed active/active geomean speedup | 1.211017x |
| main speed gate | PASS >= 1.2x |
| overall selected baseline split | torch=28, torch_npu=32 |
| public selected baseline split | torch=8, torch_npu=12 |
| overall torch_npu runnable all | 31 |
| overall torch_npu accuracy pass/fail | 31/0 |
| torch_npu runnable-all active speedup geomean | 1.211017x |
| torch_npu runnable-all speed gate | PASS >= 1.2x |
| public torch_npu runnable all | 12 |
| public torch_npu accuracy pass/fail | 12/0 |
| aux public candidate active geomean | 16.067 us |
| aux public selected baseline active geomean | 13.369 us |
| aux public selected active/active geomean speedup | 1.220977x |
| max candidate RMSE | 0 |
| commercial standard | references/commercial_standard.md @ c260c8ab7a9be4823ac8f8a07c60442de9bf141e |

## 3. 性能口径汇总

| Scope | Cases | Candidate active geomean | Baseline active geomean | Active/active geomean speedup | Precision pass | Note |
| --- | --- | --- | --- | --- | --- | --- |
| main torch_npu timed sample | 31 | 11.271 us | 13.765 us | 1.211017x | 31/31 | 主速度验收口径；candidate 和 torch_npu 均只在这同一批有 torch_npu active 计时的 case 上取几何平均；gate >= 1.2x |
| torch_npu accuracy-pass | 31 | 11.271 us | 13.765 us | 1.211017x | 31/31 | 全量 torch_npu 有效计时且本地 checker PASS 子集 |
| torch_npu accuracy-fail | 0 | N/A | N/A | N/A | 0/0 | 全量 torch_npu 有效计时但本地 checker FAIL 子集 |
| aux public selected baseline | 20 | 16.067 us | 13.369 us | 1.220977x | 20/20 | 补充语义标杆口径；torch_npu 仅在本地 checker 通过时选中，否则选 Torch |
| aux public torch semantic baseline | 20 | 16.067 us | N/A | N/A | 20/20 | 补充 Torch 语义参考口径，始终作为精度语义基准 |

## 4. Baseline 校验明细

| Case | Selected implementation | Selection rule | Torch pass | torch_npu runnable | torch_npu pass | torch_npu MERE | torch_npu MARE | torch_npu RMSE | torch_npu max diff | Reason | Seed/attrs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom/mrope_1 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 1084307072, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [0, 0, 0], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 1, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584}} |
| custom/mrope_2 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 1477511722, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [16, 24, 24], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 1, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096}} |
| custom/mrope_3 | torch | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | NO | FAIL | 0 | 0 | 0 | 0 | RuntimeError: rotary_mode only support half or interleave [ERROR] 2026-06-19-10:27:36 (PID:3639147, Device:0, RankID:-1) ERR01003 OPS invalid value | {"seed": 597351877, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [24, 20, 20], "rotary_mode": "interleaved"}, "dst_type": null, "case_detail": {"Batch": 1, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120}} |
| custom/mrope_4 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 462559110, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [16, 16, 16, 16], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 1, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192}} |
| custom/mrope_5 | torch | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | NO | FAIL | 0 | 0 | 0 | 0 | RuntimeError: torch_npu.npu_mrope does not expose cache_mode; cache_mode=interleave is not covered | {"seed": 1479209576, "attrs": {"cache_mode": "interleave", "head_size": 128, "mrope_section": [8, 12, 12], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 8, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584}} |
| custom/mrope_6 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 2071566058, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [0, 0, 0], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 8, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096}} |
| custom/mrope_7 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 187621401, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [16, 24, 24], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 8, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120}} |
| custom/mrope_8 | torch | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | NO | FAIL | 0 | 0 | 0 | 0 | RuntimeError: rotary_mode only support half or interleave [ERROR] 2026-06-19-10:29:39 (PID:3639147, Device:0, RankID:-1) ERR01003 OPS invalid value | {"seed": 42610454, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [24, 20, 20], "rotary_mode": "interleaved"}, "dst_type": null, "case_detail": {"Batch": 8, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192}} |
| custom/mrope_9 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 21837909, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [16, 16, 16, 16], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 16, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584}} |
| custom/mrope_10 | torch | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | NO | FAIL | 0 | 0 | 0 | 0 | RuntimeError: torch_npu.npu_mrope does not expose cache_mode; cache_mode=interleave is not covered | {"seed": 1564414684, "attrs": {"cache_mode": "interleave", "head_size": 128, "mrope_section": [8, 12, 12], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 16, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096}} |
| custom/mrope_11 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 1837934397, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [0, 0, 0], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 16, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120}} |
| custom/mrope_12 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 1772772694, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [16, 24, 24], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 16, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192}} |
| custom/mrope_13 | torch | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | NO | FAIL | 0 | 0 | 0 | 0 | RuntimeError: rotary_mode only support half or interleave [ERROR] 2026-06-19-10:31:37 (PID:3639147, Device:0, RankID:-1) ERR01003 OPS invalid value | {"seed": 5610152, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [24, 20, 20], "rotary_mode": "interleaved"}, "dst_type": null, "case_detail": {"Batch": 32, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584}} |
| custom/mrope_14 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 1347296229, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [16, 16, 16, 16], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 32, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096}} |
| custom/mrope_15 | torch | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | NO | FAIL | 0 | 0 | 0 | 0 | RuntimeError: torch_npu.npu_mrope does not expose cache_mode; cache_mode=interleave is not covered | {"seed": 177205156, "attrs": {"cache_mode": "interleave", "head_size": 128, "mrope_section": [8, 12, 12], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 32, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120}} |
| custom/mrope_16 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 333733719, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [0, 0, 0], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 32, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192}} |
| custom/mrope_17 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 1428411558, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [16, 24, 24], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 64, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584}} |
| custom/mrope_18 | torch | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | NO | FAIL | 0 | 0 | 0 | 0 | RuntimeError: rotary_mode only support half or interleave [ERROR] 2026-06-19-10:33:39 (PID:3639147, Device:0, RankID:-1) ERR01003 OPS invalid value | {"seed": 2109847545, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [24, 20, 20], "rotary_mode": "interleaved"}, "dst_type": null, "case_detail": {"Batch": 64, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096}} |
| custom/mrope_19 | torch_npu | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | YES | PASS | 0 | 0 | 0 | 0 |  | {"seed": 795094805, "attrs": {"cache_mode": "default", "head_size": 128, "mrope_section": [16, 16, 16, 16], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 64, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120}} |
| custom/mrope_20 | torch | Use torch_npu only when it runs and passes this directory's precision checker; otherwise use Torch semantic baseline. | PASS | NO | FAIL | 0 | 0 | 0 | 0 | RuntimeError: torch_npu.npu_mrope does not expose cache_mode; cache_mode=interleave is not covered | {"seed": 1599279944, "attrs": {"cache_mode": "interleave", "head_size": 128, "mrope_section": [8, 12, 12], "rotary_mode": "half"}, "dst_type": null, "case_detail": {"Batch": 64, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192}} |

## 5. Public 逐Case速度

| Case | Kind | Shape | DType | Selected baseline | Triton active | Torch active | torch_npu active | Selected active speedup | Torch active speedup | torch_npu active speedup | Triton precision | Torch precision | torch_npu precision | MERE | MARE | RMSE | Max diff | torch_npu error/note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom/mrope_1 | public | [[1], [1, 3584], [1, 3584], [2048, 128]] | bfloat16 | torch_npu | 3.750 us | N/A | 7.500 us | 2.000000x | N/A | 2.000000x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_2 | public | [[3, 1], [1, 4096], [1, 4096], [2048, 128]] | bfloat16 | torch_npu | 7.750 us | N/A | 9.000 us | 1.161290x | N/A | 1.161290x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_3 | public | [[3, 1], [1, 5120], [1, 5120], [2048, 128]] | bfloat16 | torch | 8.500 us | N/A | N/A | N/A | N/A | N/A | PASS | PASS | FAIL | 0 | 0 | 0 | 0 | RuntimeError: rotary_mode only support half or interleave [ERROR] 2026-06-19-10:27:36 (PID:3639147, Device:0, RankID:-1) ERR01003 OPS invalid value |
| custom/mrope_4 | public | [[4, 1], [1, 8192], [1, 8192], [2048, 128]] | bfloat16 | torch_npu | 8.000 us | N/A | 14.750 us | 1.843750x | N/A | 1.843750x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_5 | public | [[3, 8], [8, 3584], [8, 3584], [2048, 64]] | bfloat16 | torch | 14.000 us | N/A | N/A | N/A | N/A | N/A | PASS | PASS | FAIL | 0 | 0 | 0 | 0 | RuntimeError: torch_npu.npu_mrope does not expose cache_mode; cache_mode=interleave is not covered |
| custom/mrope_6 | public | [[8], [8, 4096], [8, 4096], [2048, 128]] | bfloat16 | torch_npu | 6.750 us | N/A | 10.000 us | 1.481481x | N/A | 1.481481x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_7 | public | [[3, 8], [8, 5120], [8, 5120], [2048, 128]] | bfloat16 | torch_npu | 9.500 us | N/A | 12.000 us | 1.263158x | N/A | 1.263158x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_8 | public | [[3, 8], [8, 8192], [8, 8192], [2048, 128]] | bfloat16 | torch | 24.750 us | N/A | N/A | N/A | N/A | N/A | PASS | PASS | FAIL | 0 | 0 | 0 | 0 | RuntimeError: rotary_mode only support half or interleave [ERROR] 2026-06-19-10:29:39 (PID:3639147, Device:0, RankID:-1) ERR01003 OPS invalid value |
| custom/mrope_9 | public | [[4, 16], [16, 3584], [16, 3584], [2048, 128]] | bfloat16 | torch_npu | 9.750 us | N/A | 12.000 us | 1.230769x | N/A | 1.230769x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_10 | public | [[3, 16], [16, 4096], [16, 4096], [2048, 64]] | bfloat16 | torch | 19.750 us | N/A | N/A | N/A | N/A | N/A | PASS | PASS | FAIL | 0 | 0 | 0 | 0 | RuntimeError: torch_npu.npu_mrope does not expose cache_mode; cache_mode=interleave is not covered |
| custom/mrope_11 | public | [[16], [16, 5120], [16, 5120], [2048, 128]] | bfloat16 | torch_npu | 13.000 us | N/A | 11.250 us | 0.865385x | N/A | 0.865385x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_12 | public | [[3, 16], [16, 8192], [16, 8192], [2048, 128]] | bfloat16 | torch_npu | 10.250 us | N/A | 16.250 us | 1.585366x | N/A | 1.585366x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_13 | public | [[3, 32], [32, 3584], [32, 3584], [2048, 128]] | bfloat16 | torch | 35.250 us | N/A | N/A | N/A | N/A | N/A | PASS | PASS | FAIL | 0 | 0 | 0 | 0 | RuntimeError: rotary_mode only support half or interleave [ERROR] 2026-06-19-10:31:37 (PID:3639147, Device:0, RankID:-1) ERR01003 OPS invalid value |
| custom/mrope_14 | public | [[4, 32], [32, 4096], [32, 4096], [2048, 128]] | bfloat16 | torch_npu | 10.750 us | N/A | 16.250 us | 1.511628x | N/A | 1.511628x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_15 | public | [[3, 32], [32, 5120], [32, 5120], [2048, 64]] | bfloat16 | torch | 32.000 us | N/A | N/A | N/A | N/A | N/A | PASS | PASS | FAIL | 0 | 0 | 0 | 0 | RuntimeError: torch_npu.npu_mrope does not expose cache_mode; cache_mode=interleave is not covered |
| custom/mrope_16 | public | [[32], [32, 8192], [32, 8192], [2048, 128]] | bfloat16 | torch_npu | 38.000 us | N/A | 17.250 us | 0.453947x | N/A | 0.453947x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_17 | public | [[3, 64], [64, 3584], [64, 3584], [2048, 128]] | bfloat16 | torch_npu | 19.000 us | N/A | 19.500 us | 1.026316x | N/A | 1.026316x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_18 | public | [[3, 64], [64, 4096], [64, 4096], [2048, 128]] | bfloat16 | torch | 74.500 us | N/A | N/A | N/A | N/A | N/A | PASS | PASS | FAIL | 0 | 0 | 0 | 0 | RuntimeError: rotary_mode only support half or interleave [ERROR] 2026-06-19-10:33:39 (PID:3639147, Device:0, RankID:-1) ERR01003 OPS invalid value |
| custom/mrope_19 | public | [[4, 64], [64, 5120], [64, 5120], [2048, 128]] | bfloat16 | torch_npu | 19.750 us | N/A | 22.750 us | 1.151899x | N/A | 1.151899x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/mrope_20 | public | [[3, 64], [64, 8192], [64, 8192], [2048, 64]] | bfloat16 | torch | 90.500 us | N/A | N/A | N/A | N/A | N/A | PASS | PASS | FAIL | 0 | 0 | 0 | 0 | RuntimeError: torch_npu.npu_mrope does not expose cache_mode; cache_mode=interleave is not covered |

## 6. 商业L1精度对比

| Case | Output | DType | Shape | Reference | Criterion | Candidate AE | Candidate MARE | Candidate MERE | Candidate RMSE | Baseline AE | Baseline MARE | Baseline MERE | Baseline RMSE | L1 metric status | Checker status | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom/mrope_1 | query_out | torch.bfloat16 | [1, 3584] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_1 | key_out | torch.bfloat16 | [1, 3584] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_2 | query_out | torch.bfloat16 | [1, 4096] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_2 | key_out | torch.bfloat16 | [1, 4096] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_3 | query_out | torch.bfloat16 | [1, 5120] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_3 | key_out | torch.bfloat16 | [1, 5120] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_4 | query_out | torch.bfloat16 | [1, 8192] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_4 | key_out | torch.bfloat16 | [1, 8192] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_5 | query_out | torch.bfloat16 | [8, 3584] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_5 | key_out | torch.bfloat16 | [8, 3584] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_6 | query_out | torch.bfloat16 | [8, 4096] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_6 | key_out | torch.bfloat16 | [8, 4096] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_7 | query_out | torch.bfloat16 | [8, 5120] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_7 | key_out | torch.bfloat16 | [8, 5120] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_8 | query_out | torch.bfloat16 | [8, 8192] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_8 | key_out | torch.bfloat16 | [8, 8192] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_9 | query_out | torch.bfloat16 | [16, 3584] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_9 | key_out | torch.bfloat16 | [16, 3584] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_10 | query_out | torch.bfloat16 | [16, 4096] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_10 | key_out | torch.bfloat16 | [16, 4096] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_11 | query_out | torch.bfloat16 | [16, 5120] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_11 | key_out | torch.bfloat16 | [16, 5120] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_12 | query_out | torch.bfloat16 | [16, 8192] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_12 | key_out | torch.bfloat16 | [16, 8192] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_13 | query_out | torch.bfloat16 | [32, 3584] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_13 | key_out | torch.bfloat16 | [32, 3584] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_14 | query_out | torch.bfloat16 | [32, 4096] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_14 | key_out | torch.bfloat16 | [32, 4096] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_15 | query_out | torch.bfloat16 | [32, 5120] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_15 | key_out | torch.bfloat16 | [32, 5120] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_16 | query_out | torch.bfloat16 | [32, 8192] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_16 | key_out | torch.bfloat16 | [32, 8192] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_17 | query_out | torch.bfloat16 | [64, 3584] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_17 | key_out | torch.bfloat16 | [64, 3584] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_18 | query_out | torch.bfloat16 | [64, 4096] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_18 | key_out | torch.bfloat16 | [64, 4096] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_19 | query_out | torch.bfloat16 | [64, 5120] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_19 | key_out | torch.bfloat16 | [64, 5120] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/mrope_20 | query_out | torch.bfloat16 | [64, 8192] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/mrope_20 | key_out | torch.bfloat16 | [64, 8192] | Torch semantic reference generated by this directory | BF16 mixed absolute/relative threshold, base=0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |

## 7. 随机泛化明细

| Case | Shape | Category/dst | Seed | Status | Mismatch | Max diff | MERE | MARE | RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom/mrope_random_001 | [[8], [8, 5120], [8, 5120], [256, 128]] | rope_half_default | 357155024 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_002 | [[4, 8], [8, 5120], [8, 5120], [256, 128]] | mrope4_half_default | 317870542 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_003 | [[3, 8], [8, 3584], [8, 3584], [1024, 128]] | mrope3_interleaved_default | 262279943 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_004 | [[3, 64], [64, 5120], [64, 5120], [512, 128]] | mrope3_interleaved_default | 1991089026 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_005 | [[3, 64], [64, 8192], [64, 8192], [256, 64]] | mrope3_half_interleave64 | 1376064627 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_006 | [[3, 1], [1, 5120], [1, 5120], [256, 128]] | mrope3_interleaved_default | 784605869 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_007 | [[4, 8], [8, 3584], [8, 3584], [1024, 128]] | mrope4_half_default | 189440073 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_008 | [[3, 1], [1, 3584], [1, 3584], [512, 64]] | mrope3_half_interleave64 | 948197677 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_009 | [[3, 64], [64, 8192], [64, 8192], [256, 128]] | mrope3_interleaved_default | 1588184522 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_010 | [[3, 8], [8, 4096], [8, 4096], [256, 64]] | mrope3_half_interleave64 | 1892553638 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_011 | [[3, 32], [32, 8192], [32, 8192], [1024, 64]] | mrope3_half_interleave64 | 1825869526 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_012 | [[4, 8], [8, 3584], [8, 3584], [2048, 128]] | mrope4_half_default | 347532505 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_013 | [[4, 64], [64, 4096], [64, 4096], [512, 128]] | mrope4_half_default | 1700093221 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_014 | [[3, 1], [1, 3584], [1, 3584], [512, 128]] | mrope3_half_default | 1537592907 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_015 | [[16], [16, 8192], [16, 8192], [1024, 128]] | rope_half_default | 1948272746 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_016 | [[1], [1, 8192], [1, 8192], [1024, 128]] | rope_half_default | 749951944 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_017 | [[3, 1], [1, 4096], [1, 4096], [2048, 64]] | mrope3_half_interleave64 | 1314172540 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_018 | [[3, 1], [1, 4096], [1, 4096], [256, 64]] | mrope3_half_interleave64 | 2088918334 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_019 | [[3, 32], [32, 8192], [32, 8192], [256, 64]] | mrope3_half_interleave64 | 1662217404 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_020 | [[3, 64], [64, 3584], [64, 3584], [1024, 128]] | mrope3_interleaved_default | 1237494549 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_021 | [[64], [64, 4096], [64, 4096], [1024, 128]] | rope_half_default | 1117345193 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_022 | [[4, 16], [16, 8192], [16, 8192], [512, 128]] | mrope4_half_default | 1405715249 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_023 | [[3, 64], [64, 5120], [64, 5120], [256, 128]] | mrope3_interleaved_default | 506179563 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_024 | [[3, 64], [64, 4096], [64, 4096], [1024, 128]] | mrope3_half_default | 1122512817 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_025 | [[3, 1], [1, 8192], [1, 8192], [512, 128]] | mrope3_half_default | 630834576 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_026 | [[3, 16], [16, 4096], [16, 4096], [256, 64]] | mrope3_half_interleave64 | 882373969 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_027 | [[3, 8], [8, 3584], [8, 3584], [2048, 128]] | mrope3_half_default | 1070317775 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_028 | [[4, 1], [1, 5120], [1, 5120], [2048, 128]] | mrope4_half_default | 1299025417 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_029 | [[4, 8], [8, 4096], [8, 4096], [1024, 128]] | mrope4_half_default | 2098293120 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_030 | [[3, 8], [8, 5120], [8, 5120], [256, 64]] | mrope3_half_interleave64 | 1607265919 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_031 | [[3, 64], [64, 8192], [64, 8192], [512, 128]] | mrope3_interleaved_default | 1938822793 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_032 | [[3, 64], [64, 4096], [64, 4096], [256, 128]] | mrope3_half_default | 272046855 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_033 | [[4, 32], [32, 8192], [32, 8192], [1024, 128]] | mrope4_half_default | 1374793886 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_034 | [[64], [64, 5120], [64, 5120], [2048, 128]] | rope_half_default | 277126696 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_035 | [[3, 8], [8, 3584], [8, 3584], [512, 64]] | mrope3_half_interleave64 | 1472817882 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_036 | [[3, 32], [32, 5120], [32, 5120], [1024, 128]] | mrope3_interleaved_default | 1533341534 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_037 | [[3, 8], [8, 5120], [8, 5120], [2048, 128]] | mrope3_interleaved_default | 1236142190 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_038 | [[3, 32], [32, 4096], [32, 4096], [512, 128]] | mrope3_half_default | 88831201 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_039 | [[64], [64, 3584], [64, 3584], [256, 128]] | rope_half_default | 21920662 | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/mrope_random_040 | [[3, 1], [1, 4096], [1, 4096], [1024, 128]] | mrope3_interleaved_default | 1673357719 | PASS | 0 | 0 | 0 | 0 | 0 |
