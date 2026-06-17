# KvRmsNormRopeCache 自验证报告

## 1. 验证环境与版本

- 交付目录：`third_party/ascend/tutorials/kv_rms_norm_rope_cache`
- 代码版本：当前动态 `Dk/Dv` Triton-Ascend 实现。
- NPU：使用 `ASCEND_RT_VISIBLE_DEVICES=1`，脚本内映射为 `npu:0`。
- CANN：`/mnt/model/lcw/.local/Ascend-9.0.0/cann-9.0.0`。
- OpForge latest evidence：`eval_20260617_162945`。
- OpForge best historical evidence：`eval_20260616_211136`。

## 2. 总体验证结论

| 项目 | 结果 |
| --- | --- |
| delivery correctness | `60/60 PASS` |
| public correctness | `20/20 PASS` |
| random dynamic split correctness | `40/40 PASS` |
| delivery benchmark sanity | `20/20 PASS` |
| delivery wall-sync geomean | `207.255 us` |
| OpForge latest public | `20/20 PASS`, active geomean `77.170667x` |
| OpForge dynamic audit | `40/40 PASS`, failed `0` |
| no-fallback | 被测路径只调用本地 Triton-Ascend kernel |

## 3. 交付日志

- 正确性日志：`logs/kv_rms_norm_rope_cache_validation_20260617.log`
- 正确性 JSONL：`logs/kv_rms_norm_rope_cache_validation_20260617.jsonl`
- 正确性 summary：`logs/kv_rms_norm_rope_cache_validation_20260617.summary.json`
- benchmark 日志：`logs/kv_rms_norm_rope_cache_benchmark_20260617.log`
- benchmark JSONL：`logs/kv_rms_norm_rope_cache_benchmark_20260617.jsonl`
- benchmark summary：`logs/kv_rms_norm_rope_cache_benchmark_20260617.summary.json`

## 4. OpForge public 性能明细

| Case | Candidate active | Baseline active | active/active | Candidate kernel | Baseline kernel | kernel/kernel | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `custom/kv_rms_norm_rope_cache_1` | 4.000 us | 558.500 us | 139.625000x | 3.900 us | 71.860 us | 18.425641x | `pytorch_fallback` |
| `custom/kv_rms_norm_rope_cache_2` | 4.250 us | 590.750 us | 139.000000x | 4.260 us | 73.160 us | 17.173709x | `pytorch_fallback` |
| `custom/kv_rms_norm_rope_cache_3` | 5.000 us | 691.750 us | 138.350000x | 4.880 us | 77.580 us | 15.897541x | `pytorch_fallback` |
| `custom/kv_rms_norm_rope_cache_4` | 7.000 us | 662.500 us | 94.642857x | 6.880 us | 72.260 us | 10.502907x | `pytorch_fallback` |
| `custom/kv_rms_norm_rope_cache_5` | 19.500 us | 1963.500 us | 100.692308x | 19.460 us | 208.840 us | 10.731758x | `pytorch_fallback` |
| `custom/kv_rms_norm_rope_cache_6` | 20.250 us | 1805.000 us | 89.135802x | 20.240 us | 213.980 us | 10.572134x | `pytorch_fallback` |
| `custom/kv_rms_norm_rope_cache_7` | 27.000 us | 1787.500 us | 66.203704x | 27.060 us | 223.610 us | 8.263489x | `pytorch_fallback` |
| `custom/kv_rms_norm_rope_cache_8` | 39.500 us | 1656.250 us | 41.930380x | 39.620 us | 253.560 us | 6.399798x | `pytorch_fallback` |
| `custom/kv_rms_norm_rope_cache_9` | 34.250 us | 2692.250 us | 78.605839x | 34.320 us | 379.830 us | 11.067308x | `pytorch_fallback` |
| `custom/kv_rms_norm_rope_cache_10` | 42.500 us | 2991.000 us | 70.376471x | 42.460 us | 382.250 us | 9.002591x | `pytorch_fallback` |
| `custom/kv_rms_norm_rope_cache_11` | 48.250 us | 3907.750 us | 80.989637x | 48.360 us | 387.150 us | 8.005583x | `pytorch_fallback` |
| `custom/kv_rms_norm_rope_cache_12` | 84.250 us | 3852.250 us | 45.724036x | 84.140 us | 396.590 us | 4.713454x | `pytorch_fallback` |
| `custom/kv_rms_norm_rope_cache_13` | 67.250 us | 6828.750 us | 101.542751x | 67.180 us | 648.010 us | 9.645877x | `pytorch_fallback` |
| `custom/kv_rms_norm_rope_cache_14` | 83.750 us | 7668.750 us | 91.567164x | 83.640 us | 651.770 us | 7.792563x | `pytorch_fallback` |
| `custom/kv_rms_norm_rope_cache_15` | 95.250 us | 6997.500 us | 73.464567x | 95.320 us | 652.830 us | 6.848825x | `pytorch_fallback` |
| `custom/kv_rms_norm_rope_cache_16` | 165.500 us | 8007.250 us | 48.382175x | 165.580 us | 663.020 us | 4.004228x | `pytorch_fallback` |
| `custom/kv_rms_norm_rope_cache_17` | 145.250 us | 12605.750 us | 86.786575x | 145.300 us | 1169.110 us | 8.046180x | `pytorch_fallback` |
| `custom/kv_rms_norm_rope_cache_18` | 151.500 us | 11067.250 us | 73.051155x | 151.620 us | 1173.240 us | 7.738029x | `pytorch_fallback` |
| `custom/kv_rms_norm_rope_cache_19` | 206.750 us | 11378.000 us | 55.032648x | 206.700 us | 1180.820 us | 5.712724x | `pytorch_fallback` |
| `custom/kv_rms_norm_rope_cache_20` | 302.500 us | 11349.000 us | 37.517355x | 302.450 us | 1200.000 us | 3.967598x | `pytorch_fallback` |

## 5. Random dynamic split 正确性明细

| Case | Shape | Split | Accuracy | max_diff | MARE |
| --- | --- | --- | --- | ---: | ---: |
| random_001 | B1,N64,Skv1,Scache5,Bcache1 | Dk=48,Dv=80 | PASS | 0 | 0 |
| random_002 | B16,N28,Skv1,Scache5,Bcache17 | Dk=32,Dv=96 | PASS | 0 | 0 |
| random_003 | B32,N32,Skv2,Scache8,Bcache32 | Dk=48,Dv=80 | PASS | 0.0078125 | 1.16492e-08 |
| random_004 | B32,N28,Skv2,Scache12,Bcache33 | Dk=32,Dv=96 | PASS | 0 | 0 |
| random_005 | B16,N28,Skv3,Scache4,Bcache17 | Dk=64,Dv=64 | PASS | 0 | 0 |
| random_006 | B8,N64,Skv4,Scache5,Bcache9 | Dk=80,Dv=48 | PASS | 0.015625 | 3.28337e-08 |
| random_007 | B32,N64,Skv1,Scache5,Bcache32 | Dk=64,Dv=64 | PASS | 0.00390625 | 3.08882e-09 |
| random_008 | B64,N28,Skv1,Scache12,Bcache65 | Dk=64,Dv=128 | PASS | 0 | 0 |
| random_009 | B32,N40,Skv4,Scache5,Bcache32 | Dk=48,Dv=80 | PASS | 0.00195312 | 5.08626e-09 |
| random_010 | B8,N28,Skv4,Scache8,Bcache8 | Dk=64,Dv=64 | PASS | 0 | 0 |
| random_011 | B1,N64,Skv4,Scache4,Bcache1 | Dk=64,Dv=128 | PASS | 0.0078125 | 9.64221e-08 |
| random_012 | B16,N32,Skv4,Scache5,Bcache16 | Dk=128,Dv=64 | PASS | 0 | 0 |
| random_013 | B64,N64,Skv1,Scache8,Bcache64 | Dk=32,Dv=96 | PASS | 0.0078125 | 4.57571e-09 |
| random_014 | B64,N64,Skv1,Scache8,Bcache64 | Dk=64,Dv=64 | PASS | 0 | 0 |
| random_015 | B32,N28,Skv4,Scache8,Bcache33 | Dk=32,Dv=96 | PASS | 0.0078125 | 1.85002e-08 |
| random_016 | B64,N32,Skv4,Scache4,Bcache64 | Dk=48,Dv=80 | PASS | 0.0078125 | 2.6985e-08 |
| random_017 | B16,N28,Skv3,Scache12,Bcache16 | Dk=64,Dv=64 | PASS | 0 | 0 |
| random_018 | B32,N64,Skv4,Scache9,Bcache33 | Dk=128,Dv=64 | PASS | 0 | 0 |
| random_019 | B1,N64,Skv3,Scache5,Bcache2 | Dk=32,Dv=96 | PASS | 0 | 0 |
| random_020 | B16,N32,Skv1,Scache8,Bcache16 | Dk=80,Dv=48 | PASS | 0 | 0 |
| random_021 | B64,N40,Skv3,Scache5,Bcache64 | Dk=128,Dv=64 | PASS | 0 | 0 |
| random_022 | B64,N64,Skv2,Scache9,Bcache65 | Dk=64,Dv=128 | PASS | 0.00390625 | 7.8594e-10 |
| random_023 | B32,N28,Skv2,Scache5,Bcache32 | Dk=48,Dv=80 | PASS | 0 | 0 |
| random_024 | B1,N28,Skv4,Scache4,Bcache1 | Dk=128,Dv=64 | PASS | 0 | 0 |
| random_025 | B32,N32,Skv2,Scache4,Bcache33 | Dk=32,Dv=96 | PASS | 0 | 0 |
| random_026 | B8,N40,Skv4,Scache5,Bcache8 | Dk=128,Dv=64 | PASS | 0 | 0 |
| random_027 | B64,N28,Skv2,Scache4,Bcache64 | Dk=80,Dv=48 | PASS | 0.00390625 | 1.23022e-08 |
| random_028 | B16,N28,Skv3,Scache8,Bcache17 | Dk=48,Dv=80 | PASS | 0.00195312 | 1.16568e-08 |
| random_029 | B32,N32,Skv2,Scache12,Bcache32 | Dk=64,Dv=128 | PASS | 0 | 0 |
| random_030 | B8,N28,Skv3,Scache5,Bcache8 | Dk=64,Dv=128 | PASS | 0 | 0 |
| random_031 | B16,N32,Skv4,Scache9,Bcache17 | Dk=64,Dv=64 | PASS | 0 | 0 |
| random_032 | B64,N28,Skv2,Scache12,Bcache65 | Dk=48,Dv=80 | PASS | 0 | 0 |
| random_033 | B1,N40,Skv3,Scache12,Bcache1 | Dk=32,Dv=96 | PASS | 0 | 0 |
| random_034 | B64,N32,Skv2,Scache9,Bcache64 | Dk=32,Dv=96 | PASS | 0.0078125 | 9.1242e-09 |
| random_035 | B64,N64,Skv2,Scache9,Bcache64 | Dk=80,Dv=48 | PASS | 0.0078125 | 3.8696e-09 |
| random_036 | B8,N40,Skv3,Scache5,Bcache8 | Dk=96,Dv=32 | PASS | 0 | 0 |
| random_037 | B32,N32,Skv1,Scache12,Bcache33 | Dk=128,Dv=64 | PASS | 0 | 0 |
| random_038 | B8,N64,Skv3,Scache4,Bcache8 | Dk=80,Dv=48 | PASS | 0 | 0 |
| random_039 | B16,N64,Skv2,Scache8,Bcache17 | Dk=128,Dv=64 | PASS | 0 | 0 |
| random_040 | B32,N64,Skv3,Scache12,Bcache33 | Dk=80,Dv=48 | PASS | 0 | 0 |

## 6. Delivery benchmark sanity 明细

该表为本目录脚本 wall-sync sanity，不能替代 OpForge active-window 计分。

| Case | Shape | delivery wall-sync latency | Accuracy |
| --- | --- | ---: | --- |
| public_001 | B1,N28,Dk64,Dv64 | 170.897 us | PASS |
| public_002 | B1,N32,Dk64,Dv64 | 152.137 us | PASS |
| public_003 | B1,N40,Dk64,Dv64 | 153.476 us | PASS |
| public_004 | B1,N64,Dk64,Dv64 | 158.287 us | PASS |
| public_005 | B8,N28,Dk64,Dv64 | 192.179 us | PASS |
| public_006 | B8,N32,Dk64,Dv64 | 169.278 us | PASS |
| public_007 | B8,N40,Dk64,Dv64 | 180.288 us | PASS |
| public_008 | B8,N64,Dk64,Dv64 | 169.818 us | PASS |
| public_009 | B16,N28,Dk64,Dv64 | 190.798 us | PASS |
| public_010 | B16,N32,Dk64,Dv64 | 163.897 us | PASS |
| public_011 | B16,N40,Dk64,Dv64 | 168.438 us | PASS |
| public_012 | B16,N64,Dk64,Dv64 | 204.899 us | PASS |
| public_013 | B32,N28,Dk64,Dv64 | 196.379 us | PASS |
| public_014 | B32,N32,Dk64,Dv64 | 194.809 us | PASS |
| public_015 | B32,N40,Dk64,Dv64 | 206.119 us | PASS |
| public_016 | B32,N64,Dk64,Dv64 | 297.133 us | PASS |
| public_017 | B64,N28,Dk64,Dv64 | 249.831 us | PASS |
| public_018 | B64,N32,Dk64,Dv64 | 319.494 us | PASS |
| public_019 | B64,N40,Dk64,Dv64 | 355.525 us | PASS |
| public_020 | B64,N64,Dk64,Dv64 | 473.700 us | PASS |

## 7. 边界说明

当前 public scored set 仍为 `Dk=Dv=64`；动态 split 已通过 40-case manual audit 和本目录 40-case random validation。当前 OpForge baseline source 为 `pytorch_fallback`，因此本报告不声称已经用可用 CANN whole-operator 直接验证 1.2x。若验收环境提供可用 CANN API，应按同一 public case set 重新采集直接 CANN baseline。
