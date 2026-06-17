# MRoPE 自验证报告

## 1. 环境信息

| 项目 | 数值 |
|---|---|
| 评测器 run id | `eval_20260617_151008` |
| 后端 | Triton-Ascend |
| 芯片 | Ascend910_9382 |
| CANN | 9.0.0 |
| Driver | 25.5.2 |
| Python | 3.11.14 |
| PyTorch | None |
| torch_npu | 2.10.0 |
| OpForge timing source | `results/eval_records/eval_20260617_151008/timing_artifacts/main_results.csv` |
| delivery validation | `logs/mrope_public_random_20260617.jsonl` |

## 2. 测试范围

public 集包含 20 个 BF16 case，覆盖 RoPE、3-row MRoPE、4-row MRoPE、half/interleaved、default/interleave cache，以及 `rotary_dim=64/128`。delivery random generalization 包含 40 个固定 seed 非公开元数据组合，`seed=20260617`，policy=`seeded_mrope_metadata_v1`。

## 3. 精度结果

| 范围 | Triton 候选 | 总数 | mismatch | max_diff | max_mare |
|---|---:|---:|---:|---:|---:|
| public + random | 60 | 60 | 0 | 0.000000e+00 | 0.000000e+00 |

## 4. 性能结果

`eval_20260617_151008` 是当前边界修复源码的 latest full-public 记录，状态 `PASSED`，20/20 PASS，主指标 `active_vs_active.geomean=11.657673x`，min `1.146341x`，score `100.018425`。baseline split 为 task_npu_baseline=8、pytorch_fallback=12。

## 5. Public 逐项明细

| Case | Shape | Baseline | Triton active | Triton kernel | Baseline active | Baseline kernel | active/active | kernel/kernel | Mismatch | MARE | Max diff |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | positions=[1], query=[1, 3584], cache=[2048, 128] | pytorch_fallback | 4.000 us | 3.980 us | 845.500 us | 94.260 us | 211.375000x | 23.683417x | 0 | 0.000000e+00 | 0.000000e+00 |
| 2 | positions=[3, 1], query=[1, 4096], cache=[2048, 128] | task_npu_baseline | 8.250 us | 8.260 us | 9.500 us | 9.500 us | 1.151515x | 1.150121x | 0 | 0.000000e+00 | 0.000000e+00 |
| 3 | positions=[3, 1], query=[1, 5120], cache=[2048, 128] | pytorch_fallback | 7.250 us | 7.300 us | 1052.000 us | 256.070 us | 145.103448x | 35.078082x | 0 | 0.000000e+00 | 0.000000e+00 |
| 4 | positions=[4, 1], query=[1, 8192], cache=[2048, 128] | task_npu_baseline | 8.250 us | 8.260 us | 14.250 us | 14.360 us | 1.727273x | 1.738499x | 0 | 0.000000e+00 | 0.000000e+00 |
| 5 | positions=[3, 8], query=[8, 3584], cache=[2048, 64] | pytorch_fallback | 13.000 us | 13.000 us | 1298.250 us | 219.300 us | 99.865385x | 16.869231x | 0 | 0.000000e+00 | 0.000000e+00 |
| 6 | positions=[8], query=[8, 4096], cache=[2048, 128] | pytorch_fallback | 8.000 us | 7.880 us | 914.000 us | 119.920 us | 114.250000x | 15.218274x | 0 | 0.000000e+00 | 0.000000e+00 |
| 7 | positions=[3, 8], query=[8, 5120], cache=[2048, 128] | task_npu_baseline | 9.250 us | 9.240 us | 10.750 us | 10.720 us | 1.162162x | 1.160173x | 0 | 0.000000e+00 | 0.000000e+00 |
| 8 | positions=[3, 8], query=[8, 8192], cache=[2048, 128] | pytorch_fallback | 23.000 us | 22.900 us | 1221.750 us | 368.050 us | 53.119565x | 16.072052x | 0 | 0.000000e+00 | 0.000000e+00 |
| 9 | positions=[4, 16], query=[16, 3584], cache=[2048, 128] | task_npu_baseline | 10.250 us | 10.300 us | 11.750 us | 11.820 us | 1.146341x | 1.147573x | 0 | 0.000000e+00 | 0.000000e+00 |
| 10 | positions=[3, 16], query=[16, 4096], cache=[2048, 64] | pytorch_fallback | 18.250 us | 18.320 us | 1210.500 us | 245.360 us | 66.328767x | 13.393013x | 0 | 0.000000e+00 | 0.000000e+00 |
| 11 | positions=[16], query=[16, 5120], cache=[2048, 128] | pytorch_fallback | 14.750 us | 14.660 us | 912.250 us | 152.640 us | 61.847458x | 10.412005x | 0 | 0.000000e+00 | 0.000000e+00 |
| 12 | positions=[3, 16], query=[16, 8192], cache=[2048, 128] | task_npu_baseline | 9.750 us | 9.660 us | 16.500 us | 16.440 us | 1.692308x | 1.701863x | 0 | 0.000000e+00 | 0.000000e+00 |
| 13 | positions=[3, 32], query=[32, 3584], cache=[2048, 128] | pytorch_fallback | 34.250 us | 34.180 us | 1043.500 us | 449.250 us | 30.467153x | 13.143651x | 0 | 0.000000e+00 | 0.000000e+00 |
| 14 | positions=[4, 32], query=[32, 4096], cache=[2048, 128] | task_npu_baseline | 10.500 us | 10.520 us | 16.500 us | 16.420 us | 1.571429x | 1.560836x | 0 | 0.000000e+00 | 0.000000e+00 |
| 15 | positions=[3, 32], query=[32, 5120], cache=[2048, 64] | pytorch_fallback | 30.250 us | 30.260 us | 963.000 us | 294.840 us | 31.834711x | 9.743556x | 0 | 0.000000e+00 | 0.000000e+00 |
| 16 | positions=[32], query=[32, 8192], cache=[2048, 128] | pytorch_fallback | 39.250 us | 39.360 us | 721.500 us | 198.510 us | 18.382166x | 5.043445x | 0 | 0.000000e+00 | 0.000000e+00 |
| 17 | positions=[3, 64], query=[64, 3584], cache=[2048, 128] | task_npu_baseline | 19.000 us | 19.080 us | 22.250 us | 22.340 us | 1.171053x | 1.170860x | 0 | 0.000000e+00 | 0.000000e+00 |
| 18 | positions=[3, 64], query=[64, 4096], cache=[2048, 128] | pytorch_fallback | 70.250 us | 70.240 us | 1030.250 us | 597.910 us | 14.665480x | 8.512386x | 0 | 0.000000e+00 | 0.000000e+00 |
| 19 | positions=[4, 64], query=[64, 5120], cache=[2048, 128] | task_npu_baseline | 20.000 us | 19.920 us | 24.000 us | 23.900 us | 1.200000x | 1.199799x | 0 | 0.000000e+00 | 0.000000e+00 |
| 20 | positions=[3, 64], query=[64, 8192], cache=[2048, 64] | pytorch_fallback | 90.500 us | 90.380 us | 985.000 us | 326.690 us | 10.883978x | 3.614627x | 0 | 0.000000e+00 | 0.000000e+00 |

## 6. 随机泛化明细

| # | Shape | Category | Rotary | Cache | Triton acc | Mismatch | MARE | Max diff |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | positions=[5], query=[5, 8192], cache=[256, 128] | rope_half_default | half | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 2 | positions=[4, 5], query=[5, 8192], cache=[256, 128] | mrope4_half_default | half | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 3 | positions=[3, 3], query=[3, 2048], cache=[256, 128] | mrope3_interleaved_default | interleaved | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 4 | positions=[3, 64], query=[64, 1024], cache=[256, 128] | mrope3_interleaved_default | interleaved | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 5 | positions=[3, 2], query=[2, 5120], cache=[256, 128] | mrope3_interleaved_default | interleaved | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 6 | positions=[4, 3], query=[3, 8192], cache=[1024, 128] | mrope4_half_default | half | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 7 | positions=[3, 1], query=[1, 1024], cache=[512, 64] | mrope3_half_interleave64 | half | interleave | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 8 | positions=[3, 64], query=[64, 512], cache=[512, 128] | mrope3_interleaved_default | interleaved | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 9 | positions=[4, 3], query=[3, 1024], cache=[2048, 128] | mrope4_half_default | half | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 10 | positions=[4, 7], query=[7, 8192], cache=[2048, 128] | mrope4_half_default | half | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 11 | positions=[3, 17], query=[17, 512], cache=[2048, 128] | mrope3_half_default | half | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 12 | positions=[3, 17], query=[17, 1024], cache=[256, 128] | mrope3_half_default | half | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 13 | positions=[3, 7], query=[7, 512], cache=[1024, 128] | mrope3_half_default | half | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 14 | positions=[3, 2], query=[2, 512], cache=[1024, 128] | mrope3_half_default | half | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 15 | positions=[3, 1], query=[1, 4096], cache=[2048, 64] | mrope3_half_interleave64 | half | interleave | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 16 | positions=[3, 1], query=[1, 512], cache=[2048, 64] | mrope3_half_interleave64 | half | interleave | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 17 | positions=[4, 8], query=[8, 4096], cache=[1024, 128] | mrope4_half_default | half | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 18 | positions=[33], query=[33, 2048], cache=[2048, 128] | rope_half_default | half | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 19 | positions=[3, 3], query=[3, 3584], cache=[1024, 64] | mrope3_half_interleave64 | half | interleave | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 20 | positions=[3, 64], query=[64, 512], cache=[1024, 128] | mrope3_interleaved_default | interleaved | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 21 | positions=[3, 64], query=[64, 2048], cache=[512, 64] | mrope3_half_interleave64 | half | interleave | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 22 | positions=[3, 64], query=[64, 1024], cache=[2048, 64] | mrope3_half_interleave64 | half | interleave | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 23 | positions=[3, 2], query=[2, 1024], cache=[512, 128] | mrope3_interleaved_default | interleaved | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 24 | positions=[3, 31], query=[31, 4096], cache=[2048, 64] | mrope3_half_interleave64 | half | interleave | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 25 | positions=[31], query=[31, 2048], cache=[512, 128] | rope_half_default | half | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 26 | positions=[33], query=[33, 512], cache=[512, 128] | rope_half_default | half | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 27 | positions=[3, 7], query=[7, 2048], cache=[512, 128] | mrope3_half_default | half | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 28 | positions=[8], query=[8, 512], cache=[1024, 128] | rope_half_default | half | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 29 | positions=[3, 64], query=[64, 1024], cache=[512, 128] | mrope3_interleaved_default | interleaved | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 30 | positions=[31], query=[31, 3584], cache=[1024, 128] | rope_half_default | half | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 31 | positions=[64], query=[64, 4096], cache=[2048, 128] | rope_half_default | half | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 32 | positions=[3, 3], query=[3, 5120], cache=[512, 64] | mrope3_half_interleave64 | half | interleave | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 33 | positions=[3, 31], query=[31, 8192], cache=[1024, 128] | mrope3_interleaved_default | interleaved | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 34 | positions=[3, 7], query=[7, 512], cache=[1024, 64] | mrope3_half_interleave64 | half | interleave | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 35 | positions=[4, 3], query=[3, 5120], cache=[512, 128] | mrope4_half_default | half | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 36 | positions=[3, 31], query=[31, 3584], cache=[1024, 128] | mrope3_half_default | half | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 37 | positions=[3, 5], query=[5, 512], cache=[256, 128] | mrope3_interleaved_default | interleaved | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 38 | positions=[3, 1], query=[1, 5120], cache=[1024, 128] | mrope3_interleaved_default | interleaved | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 39 | positions=[17], query=[17, 2048], cache=[512, 128] | rope_half_default | half | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |
| 40 | positions=[4, 64], query=[64, 2048], cache=[2048, 128] | mrope4_half_default | half | default | PASS | 0 | 0.000000e+00 | 0.000000e+00 |

## 7. 无 fallback 检查

交付实现的被测路径使用 `@triton.jit` kernel。Python 侧只进行元数据校验、输出分配和 launch 参数组织，不调用 PyTorch 等价实现、`torch_npu.npu_mrope`、CANN/vendor MRoPE、CPU fallback、任务 golden 路径或 peer 解法。验证脚本中的 reference 只用于自验证比较，不在被测路径中调用。

## 8. 复现命令

```bash
python3 validate_mrope.py --public --random-generalization 40 --random-seed 20260617   --jsonl logs/mrope_public_random_20260617.jsonl   --summary-json logs/mrope_public_random_20260617.summary.json
```
