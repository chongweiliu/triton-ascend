# AddRmsNorm 算子自验证报告

## 1. 报告说明

- 单一数值证据源：`logs/add_rms_norm_validation.jsonl`
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
| evidence source | logs/add_rms_norm_validation.jsonl |
| total cases | 60 |
| public cases | 20 |
| random/generalization cases | 40 |
| candidate pass | 60/60 |
| main speed sample | torch_npu runnable all (58 cases) |
| main speed candidate active geomean | 4.116 us |
| main speed torch_npu active geomean | 15.131 us |
| main speed active/active geomean speedup | 3.676306x |
| main speed gate | PASS >= 1.2x |
| overall selected baseline split | torch=11, torch_npu=49 |
| public selected baseline split | torch=4, torch_npu=16 |
| overall torch_npu runnable all | 58 |
| overall torch_npu accuracy pass/fail | 47/11 |
| torch_npu runnable-all active speedup geomean | 3.676306x |
| torch_npu runnable-all speed gate | PASS >= 1.2x |
| public torch_npu runnable all | 20 |
| public torch_npu accuracy pass/fail | 16/4 |
| aux public candidate active geomean | 4.435 us |
| aux public selected baseline active geomean | 16.572 us |
| aux public selected active/active geomean speedup | 4.009564x |
| max candidate RMSE | 1.70598e-05 |
| commercial standard | references/commercial_standard.md @ c260c8ab7a9be4823ac8f8a07c60442de9bf141e |

## 3. 性能口径汇总

| Scope | Cases | Candidate active geomean | Baseline active geomean | Active/active geomean speedup | Precision pass | Note |
| --- | --- | --- | --- | --- | --- | --- |
| main torch_npu timed sample | 58 | 4.116 us | 15.131 us | 3.676306x | 47/58 | 主速度验收口径；candidate 和 torch_npu 均只在这同一批有 torch_npu active 计时的 case 上取几何平均；gate >= 1.2x |
| torch_npu accuracy-pass | 47 | 3.968 us | 13.444 us | 3.387874x | 47/47 | 全量 torch_npu 有效计时且本地 checker PASS 子集 |
| torch_npu accuracy-fail | 11 | 4.810 us | 25.072 us | 5.212268x | 0/11 | 全量 torch_npu 有效计时但本地 checker FAIL 子集 |
| aux public selected baseline | 20 | 4.435 us | 16.572 us | 4.009564x | 20/20 | 补充语义标杆口径；torch_npu 仅在本地 checker 通过时选中，否则选 Torch |
| aux public torch semantic baseline | 20 | 4.435 us | N/A | N/A | 20/20 | 补充 Torch 语义参考口径，始终作为精度语义基准 |

## 4. Baseline 校验明细

| Case | Selected implementation | Selection rule | Torch pass | torch_npu runnable | torch_npu pass | torch_npu MERE | torch_npu MARE | torch_npu RMSE | torch_npu max diff | Reason | Seed/attrs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom/add_rms_norm_1 | torch_npu | Use torch_npu.npu_add_rms_norm only when its output passes the same local BF16 precision check; otherwise use the Torch semantic baseline. | PASS | YES | PASS | 0.00186161 | 0.010929 | 0.0034907 | 0.015625 |  | {"seed": null, "attrs": {"Batch": 1, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1}, "dst_type": null, "case_detail": {"Batch": 1, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1}} |
| custom/add_rms_norm_2 | torch_npu | Use torch_npu.npu_add_rms_norm only when its output passes the same local BF16 precision check; otherwise use the Torch semantic baseline. | PASS | YES | PASS | 0.00187768 | 0.0103627 | 0.00362131 | 0.015625 |  | {"seed": null, "attrs": {"Batch": 1, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1}, "dst_type": null, "case_detail": {"Batch": 1, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1}} |
| custom/add_rms_norm_3 | torch_npu | Use torch_npu.npu_add_rms_norm only when its output passes the same local BF16 precision check; otherwise use the Torch semantic baseline. | PASS | YES | PASS | 0.001839 | 0.0101523 | 0.00335584 | 0.015625 |  | {"seed": null, "attrs": {"Batch": 1, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1}, "dst_type": null, "case_detail": {"Batch": 1, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1}} |
| custom/add_rms_norm_4 | torch_npu | Use torch_npu.npu_add_rms_norm only when its output passes the same local BF16 precision check; otherwise use the Torch semantic baseline. | PASS | YES | PASS | 0.00175201 | 0.010989 | 0.00346438 | 0.015625 |  | {"seed": null, "attrs": {"Batch": 1, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1}, "dst_type": null, "case_detail": {"Batch": 1, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1}} |
| custom/add_rms_norm_5 | torch_npu | Use torch_npu.npu_add_rms_norm only when its output passes the same local BF16 precision check; otherwise use the Torch semantic baseline. | PASS | YES | PASS | 0.0052275 | 0.0198675 | 0.00683994 | 0.046875 |  | {"seed": null, "attrs": {"Batch": 8, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1}, "dst_type": null, "case_detail": {"Batch": 8, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1}} |
| custom/add_rms_norm_6 | torch | Use torch_npu.npu_add_rms_norm only when its output passes the same local BF16 precision check; otherwise use the Torch semantic baseline. | PASS | YES | FAIL | 0.00813022 | 0.025 | 0.0100864 | 0.0625 | MERE/MARE threshold exceeded | {"seed": null, "attrs": {"Batch": 8, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1}, "dst_type": null, "case_detail": {"Batch": 8, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1}} |
| custom/add_rms_norm_7 | torch_npu | Use torch_npu.npu_add_rms_norm only when its output passes the same local BF16 precision check; otherwise use the Torch semantic baseline. | PASS | YES | PASS | 0.00554054 | 0.0232558 | 0.00751729 | 0.046875 |  | {"seed": null, "attrs": {"Batch": 8, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1}, "dst_type": null, "case_detail": {"Batch": 8, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1}} |
| custom/add_rms_norm_8 | torch_npu | Use torch_npu.npu_add_rms_norm only when its output passes the same local BF16 precision check; otherwise use the Torch semantic baseline. | PASS | YES | PASS | 0.00590961 | 0.0236686 | 0.00828742 | 0.0625 |  | {"seed": null, "attrs": {"Batch": 8, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1}, "dst_type": null, "case_detail": {"Batch": 8, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1}} |
| custom/add_rms_norm_9 | torch_npu | Use torch_npu.npu_add_rms_norm only when its output passes the same local BF16 precision check; otherwise use the Torch semantic baseline. | PASS | YES | PASS | 0.00776937 | 0.0300752 | 0.0104457 | 0.078125 |  | {"seed": null, "attrs": {"Batch": 16, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1}, "dst_type": null, "case_detail": {"Batch": 16, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1}} |
| custom/add_rms_norm_10 | torch_npu | Use torch_npu.npu_add_rms_norm only when its output passes the same local BF16 precision check; otherwise use the Torch semantic baseline. | PASS | YES | PASS | 0.00738263 | 0.0307692 | 0.0103094 | 0.078125 |  | {"seed": null, "attrs": {"Batch": 16, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1}, "dst_type": null, "case_detail": {"Batch": 16, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1}} |
| custom/add_rms_norm_11 | torch_npu | Use torch_npu.npu_add_rms_norm only when its output passes the same local BF16 precision check; otherwise use the Torch semantic baseline. | PASS | YES | PASS | 0.00654923 | 0.0253165 | 0.00869542 | 0.0625 |  | {"seed": null, "attrs": {"Batch": 16, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1}, "dst_type": null, "case_detail": {"Batch": 16, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1}} |
| custom/add_rms_norm_12 | torch_npu | Use torch_npu.npu_add_rms_norm only when its output passes the same local BF16 precision check; otherwise use the Torch semantic baseline. | PASS | YES | PASS | 0.00365149 | 0.0185185 | 0.00541934 | 0.046875 |  | {"seed": null, "attrs": {"Batch": 16, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1}, "dst_type": null, "case_detail": {"Batch": 16, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1}} |
| custom/add_rms_norm_13 | torch | Use torch_npu.npu_add_rms_norm only when its output passes the same local BF16 precision check; otherwise use the Torch semantic baseline. | PASS | YES | FAIL | 0.00902983 | 0.0323887 | 0.0117748 | 0.09375 | MERE/MARE threshold exceeded | {"seed": null, "attrs": {"Batch": 32, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1}, "dst_type": null, "case_detail": {"Batch": 32, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1}} |
| custom/add_rms_norm_14 | torch_npu | Use torch_npu.npu_add_rms_norm only when its output passes the same local BF16 precision check; otherwise use the Torch semantic baseline. | PASS | YES | PASS | 0.00715022 | 0.0300752 | 0.0094638 | 0.078125 |  | {"seed": null, "attrs": {"Batch": 32, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1}, "dst_type": null, "case_detail": {"Batch": 32, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1}} |
| custom/add_rms_norm_15 | torch_npu | Use torch_npu.npu_add_rms_norm only when its output passes the same local BF16 precision check; otherwise use the Torch semantic baseline. | PASS | YES | PASS | 0.00767149 | 0.035461 | 0.0109786 | 0.09375 |  | {"seed": null, "attrs": {"Batch": 32, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1}, "dst_type": null, "case_detail": {"Batch": 32, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1}} |
| custom/add_rms_norm_16 | torch_npu | Use torch_npu.npu_add_rms_norm only when its output passes the same local BF16 precision check; otherwise use the Torch semantic baseline. | PASS | YES | PASS | 0.00616747 | 0.0272109 | 0.00823426 | 0.078125 |  | {"seed": null, "attrs": {"Batch": 32, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1}, "dst_type": null, "case_detail": {"Batch": 32, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1}} |
| custom/add_rms_norm_17 | torch | Use torch_npu.npu_add_rms_norm only when its output passes the same local BF16 precision check; otherwise use the Torch semantic baseline. | PASS | YES | FAIL | 0.00913102 | 0.037037 | 0.0121364 | 0.09375 | MERE/MARE threshold exceeded | {"seed": null, "attrs": {"Batch": 64, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1}, "dst_type": null, "case_detail": {"Batch": 64, "HeadDim": 128, "HeadNum": 28, "HiddenSize": 3584, "SequenceLength": 1}} |
| custom/add_rms_norm_18 | torch_npu | Use torch_npu.npu_add_rms_norm only when its output passes the same local BF16 precision check; otherwise use the Torch semantic baseline. | PASS | YES | PASS | 0.00756485 | 0.0337838 | 0.00996927 | 0.09375 |  | {"seed": null, "attrs": {"Batch": 64, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1}, "dst_type": null, "case_detail": {"Batch": 64, "HeadDim": 128, "HeadNum": 32, "HiddenSize": 4096, "SequenceLength": 1}} |
| custom/add_rms_norm_19 | torch | Use torch_npu.npu_add_rms_norm only when its output passes the same local BF16 precision check; otherwise use the Torch semantic baseline. | PASS | YES | FAIL | 0.00814807 | 0.0364963 | 0.0107895 | 0.09375 | MERE/MARE threshold exceeded | {"seed": null, "attrs": {"Batch": 64, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1}, "dst_type": null, "case_detail": {"Batch": 64, "HeadDim": 128, "HeadNum": 40, "HiddenSize": 5120, "SequenceLength": 1}} |
| custom/add_rms_norm_20 | torch_npu | Use torch_npu.npu_add_rms_norm only when its output passes the same local BF16 precision check; otherwise use the Torch semantic baseline. | PASS | YES | PASS | 0.005659 | 0.0296296 | 0.00770984 | 0.078125 |  | {"seed": null, "attrs": {"Batch": 64, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1}, "dst_type": null, "case_detail": {"Batch": 64, "HeadDim": 128, "HeadNum": 64, "HiddenSize": 8192, "SequenceLength": 1}} |

## 5. Public 逐Case速度

| Case | Kind | Shape | DType | Selected baseline | Triton active | Torch active | torch_npu active | Selected active speedup | Torch active speedup | torch_npu active speedup | Triton precision | Torch precision | torch_npu precision | MERE | MARE | RMSE | Max diff | torch_npu error/note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom/add_rms_norm_1 | public | [1, 1, 3584] | bfloat16 | torch_npu | 2.500 us | N/A | 3.000 us | 1.200000x | N/A | 1.200000x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/add_rms_norm_2 | public | [1, 1, 4096] | bfloat16 | torch_npu | 2.500 us | N/A | 2.750 us | 1.100000x | N/A | 1.100000x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/add_rms_norm_3 | public | [1, 1, 5120] | bfloat16 | torch_npu | 3.000 us | N/A | 3.250 us | 1.083333x | N/A | 1.083333x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/add_rms_norm_4 | public | [1, 1, 8192] | bfloat16 | torch_npu | 2.750 us | N/A | 3.500 us | 1.272727x | N/A | 1.272727x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/add_rms_norm_5 | public | [8, 1, 3584] | bfloat16 | torch_npu | 3.750 us | N/A | 9.500 us | 2.533333x | N/A | 2.533333x | PASS | PASS | PASS | 1.74386e-07 | 0.005 | 1.15346e-05 | 0.00195312 |  |
| custom/add_rms_norm_6 | public | [8, 1, 4096] | bfloat16 | torch | 3.500 us | N/A | 10.750 us | N/A | N/A | 3.071429x | PASS | PASS | FAIL | 0 | 0 | 0 | 0 | MERE/MARE threshold exceeded |
| custom/add_rms_norm_7 | public | [8, 1, 5120] | bfloat16 | torch_npu | 4.250 us | N/A | 12.250 us | 2.882353x | N/A | 2.882353x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/add_rms_norm_8 | public | [8, 1, 8192] | bfloat16 | torch_npu | 4.000 us | N/A | 18.500 us | 4.625000x | N/A | 4.625000x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/add_rms_norm_9 | public | [16, 1, 3584] | bfloat16 | torch_npu | 4.250 us | N/A | 16.000 us | 3.764706x | N/A | 3.764706x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/add_rms_norm_10 | public | [16, 1, 4096] | bfloat16 | torch_npu | 3.750 us | N/A | 18.250 us | 4.866667x | N/A | 4.866667x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/add_rms_norm_11 | public | [16, 1, 5120] | bfloat16 | torch_npu | 4.750 us | N/A | 20.750 us | 4.368421x | N/A | 4.368421x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/add_rms_norm_12 | public | [16, 1, 8192] | bfloat16 | torch_npu | 4.250 us | N/A | 33.500 us | 7.882353x | N/A | 7.882353x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/add_rms_norm_13 | public | [32, 1, 3584] | bfloat16 | torch | 4.750 us | N/A | 30.500 us | N/A | N/A | 6.421053x | PASS | PASS | FAIL | 0 | 0 | 0 | 0 | MERE/MARE threshold exceeded |
| custom/add_rms_norm_14 | public | [32, 1, 4096] | bfloat16 | torch_npu | 4.500 us | N/A | 34.500 us | 7.666667x | N/A | 7.666667x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/add_rms_norm_15 | public | [32, 1, 5120] | bfloat16 | torch_npu | 5.500 us | N/A | 38.750 us | 7.045455x | N/A | 7.045455x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/add_rms_norm_16 | public | [32, 1, 8192] | bfloat16 | torch_npu | 5.500 us | N/A | 65.750 us | 11.954545x | N/A | 11.954545x | PASS | PASS | PASS | 1.74986e-08 | 0.00458715 | 3.8147e-06 | 0.00195312 |  |
| custom/add_rms_norm_17 | public | [64, 1, 3584] | bfloat16 | torch | 8.000 us | N/A | 58.250 us | N/A | N/A | 7.281250x | PASS | PASS | FAIL | 0 | 0 | 0 | 0 | MERE/MARE threshold exceeded |
| custom/add_rms_norm_18 | public | [64, 1, 4096] | bfloat16 | torch_npu | 6.500 us | N/A | 67.500 us | 10.384615x | N/A | 10.384615x | PASS | PASS | PASS | 0 | 0 | 0 | 0 |  |
| custom/add_rms_norm_19 | public | [64, 1, 5120] | bfloat16 | torch | 9.000 us | N/A | 76.000 us | N/A | N/A | 8.444444x | PASS | PASS | FAIL | 0 | 0 | 0 | 0 | MERE/MARE threshold exceeded |
| custom/add_rms_norm_20 | public | [64, 1, 8192] | bfloat16 | torch_npu | 7.750 us | N/A | 133.000 us | 17.161290x | N/A | 17.161290x | PASS | PASS | PASS | 8.51494e-09 | 0.00446428 | 6.7435e-07 | 0.000488281 |  |

## 6. 商业L1精度对比

| Case | Output | DType | Shape | Reference | Criterion | Candidate AE | Candidate MARE | Candidate MERE | Candidate RMSE | Baseline AE | Baseline MARE | Baseline MERE | Baseline RMSE | L1 metric status | Checker status | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom/add_rms_norm_1 | output | bfloat16 | [1, 1, 3584] | Torch semantic reference generated by this directory | operator checker threshold=0.0078125 | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/add_rms_norm_2 | output | bfloat16 | [1, 1, 4096] | Torch semantic reference generated by this directory | operator checker threshold=0.0078125 | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/add_rms_norm_3 | output | bfloat16 | [1, 1, 5120] | Torch semantic reference generated by this directory | operator checker threshold=0.0078125 | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/add_rms_norm_4 | output | bfloat16 | [1, 1, 8192] | Torch semantic reference generated by this directory | operator checker threshold=0.0078125 | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/add_rms_norm_5 | output | bfloat16 | [8, 1, 3584] | Torch semantic reference generated by this directory | operator checker threshold=0.0078125 | 0.00195312 | 0.005 | 1.74386e-07 | 1.15346e-05 | N/A | N/A | N/A | N/A | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/add_rms_norm_6 | output | bfloat16 | [8, 1, 4096] | Torch semantic reference generated by this directory | operator checker threshold=0.0078125 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/add_rms_norm_7 | output | bfloat16 | [8, 1, 5120] | Torch semantic reference generated by this directory | operator checker threshold=0.0078125 | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/add_rms_norm_8 | output | bfloat16 | [8, 1, 8192] | Torch semantic reference generated by this directory | operator checker threshold=0.0078125 | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/add_rms_norm_9 | output | bfloat16 | [16, 1, 3584] | Torch semantic reference generated by this directory | operator checker threshold=0.0078125 | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/add_rms_norm_10 | output | bfloat16 | [16, 1, 4096] | Torch semantic reference generated by this directory | operator checker threshold=0.0078125 | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/add_rms_norm_11 | output | bfloat16 | [16, 1, 5120] | Torch semantic reference generated by this directory | operator checker threshold=0.0078125 | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/add_rms_norm_12 | output | bfloat16 | [16, 1, 8192] | Torch semantic reference generated by this directory | operator checker threshold=0.0078125 | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/add_rms_norm_13 | output | bfloat16 | [32, 1, 3584] | Torch semantic reference generated by this directory | operator checker threshold=0.0078125 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/add_rms_norm_14 | output | bfloat16 | [32, 1, 4096] | Torch semantic reference generated by this directory | operator checker threshold=0.0078125 | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/add_rms_norm_15 | output | bfloat16 | [32, 1, 5120] | Torch semantic reference generated by this directory | operator checker threshold=0.0078125 | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/add_rms_norm_16 | output | bfloat16 | [32, 1, 8192] | Torch semantic reference generated by this directory | operator checker threshold=0.0078125 | 0.00195312 | 0.00458715 | 1.74986e-08 | 3.8147e-06 | N/A | N/A | N/A | N/A | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/add_rms_norm_17 | output | bfloat16 | [64, 1, 3584] | Torch semantic reference generated by this directory | operator checker threshold=0.0078125 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/add_rms_norm_18 | output | bfloat16 | [64, 1, 4096] | Torch semantic reference generated by this directory | operator checker threshold=0.0078125 | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |
| custom/add_rms_norm_19 | output | bfloat16 | [64, 1, 5120] | Torch semantic reference generated by this directory | operator checker threshold=0.0078125 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS | PASS | baseline=torch; RMSE由本目录 checker 输出 |
| custom/add_rms_norm_20 | output | bfloat16 | [64, 1, 8192] | Torch semantic reference generated by this directory | operator checker threshold=0.0078125 | 0.000488281 | 0.00446428 | 8.51494e-09 | 6.7435e-07 | N/A | N/A | N/A | N/A | PASS | PASS | baseline=torch_npu; RMSE由本目录 checker 输出 |

## 7. 随机泛化明细

| Case | Shape | Category/dst | Seed | Status | Mismatch | Max diff | MERE | MARE | RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom/add_rms_norm_random_001 | [8, 1, 3584] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_002 | [1, 1, 5120] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_003 | [32, 1, 8192] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_004 | [32, 1, 3584] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_005 | [64, 1, 4096] | docs_standard_value_variation |  | PASS | 0 | 0.0078125 | 3.60951e-08 | 0.00507614 | 1.70598e-05 |
| custom/add_rms_norm_random_006 | [1, 1, 3584] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_007 | [32, 1, 3584] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_008 | [8, 1, 4096] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_009 | [16, 1, 3584] | docs_standard_value_variation |  | PASS | 0 | 0.00390625 | 1.25458e-07 | 0.00719424 | 1.63123e-05 |
| custom/add_rms_norm_random_010 | [1, 1, 4096] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_011 | [1, 1, 4096] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_012 | [32, 1, 4096] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_013 | [64, 1, 4096] | docs_standard_value_variation |  | PASS | 0 | 0.0078125 | 3.41444e-08 | 0.00454545 | 1.70598e-05 |
| custom/add_rms_norm_random_014 | [1, 1, 3584] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_015 | [32, 1, 5120] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_016 | [8, 1, 8192] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_017 | [16, 1, 5120] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_018 | [8, 1, 3584] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_019 | [1, 1, 3584] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_020 | [64, 1, 8192] | docs_standard_value_variation |  | PASS | 0 | 0.000976562 | 8.36556e-09 | 0.00438596 | 1.3487e-06 |
| custom/add_rms_norm_random_021 | [8, 1, 3584] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_022 | [1, 1, 8192] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_023 | [16, 1, 4096] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_024 | [16, 1, 8192] | docs_standard_value_variation |  | PASS | 0 | 0.000488281 | 7.03168e-08 | 0.00460829 | 1.39021e-06 |
| custom/add_rms_norm_random_025 | [16, 1, 8192] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_026 | [16, 1, 3584] | docs_standard_value_variation |  | PASS | 0 | 0.00390625 | 7.48438e-08 | 0.00429185 | 1.63123e-05 |
| custom/add_rms_norm_random_027 | [1, 1, 5120] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_028 | [1, 1, 5120] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_029 | [32, 1, 3584] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_030 | [8, 1, 5120] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_031 | [8, 1, 4096] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_032 | [32, 1, 4096] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_033 | [1, 1, 8192] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_034 | [8, 1, 8192] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_035 | [64, 1, 4096] | docs_standard_value_variation |  | PASS | 0 | 0.000976562 | 3.41116e-08 | 0.00492611 | 1.92219e-06 |
| custom/add_rms_norm_random_036 | [64, 1, 3584] | docs_standard_value_variation |  | PASS | 0 | 0.0078125 | 1.83179e-08 | 0.00420168 | 1.63123e-05 |
| custom/add_rms_norm_random_037 | [64, 1, 8192] | docs_standard_value_variation |  | PASS | 0 | 0.00390625 | 4.84493e-08 | 0.00653595 | 6.65013e-06 |
| custom/add_rms_norm_random_038 | [8, 1, 5120] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_039 | [16, 1, 3584] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
| custom/add_rms_norm_random_040 | [8, 1, 8192] | docs_standard_value_variation |  | PASS | 0 | 0 | 0 | 0 | 0 |
