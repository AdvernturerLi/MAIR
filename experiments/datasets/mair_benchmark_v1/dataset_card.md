# mair_benchmark_v1 数据集卡片

> 无泄漏合成图像修复基准，用于 MAIR 的方法验证与消融实验。生成日期 2026-10-10。

## 1. 概览

| 项 | 值 |
|---|---|
| 数据集名 | mair_benchmark_v1 |
| 来源 HQ | MiO_test `original/GT/`（100 张，取其中 30 张） |
| 深度图（雾合成） | MiO_test `original/depth/LR_tensor/{id}/predict_depth.mat` |
| 划分方式 | 先按 HQ 原图 seeded shuffle 划分，再生成退化图 |
| validation | **10 张 HQ** × 8 组合 = **80 张 LQ** |
| test | **20 张 HQ** × 8 组合 = **160 张 LQ** |
| 合计 | 30 HQ + 240 LQ |

## 2. 划分（无泄漏）

- 以 **HQ 原图编号** 为单位划分：同一 HQ 产生的所有退化图只进同一 split。
- `split_seed = 42`，用独立 `RandomState(42)` 对 100 个 HQ 编号做 permutation，取前 10 为 validation、再 20 为 test。

**validation HQ（10）**：084, 054, 071, 046, 045, 040, 023, 081, 011, 001

**test HQ（20）**：019, 031, 074, 034, 091, 005, 077, 078, 013, 032, 056, 089, 027, 043, 070, 016, 041, 097, 010, 073

## 3. 8 类退化组合

> 命名约定：`blur` = **motion blur**，`low-light` = **dark**，`low-resolution` = `lr(0.25×)`。
> 组合内退化按「scene → imaging → compression」顺序应用（与论文退化先验一致）。

| combo | 退化序列（应用顺序） | 阶段 | val | test |
|---|---|---|---|---|
| noise+blur | motion blur → noise | imaging, imaging | 10 | 20 |
| rain+haze | rain → haze | scene, scene | 10 | 20 |
| low-light+noise | dark → noise | scene, imaging | 10 | 20 |
| jpeg+noise | noise → jpeg compression artifact | imaging, compression | 10 | 20 |
| blur+low-resolution | motion blur → low resolution | imaging, imaging | 10 | 20 |
| haze+low-resolution | haze → low resolution | scene, imaging | 10 | 20 |
| rain+blur | rain → motion blur | scene, imaging | 10 | 20 |
| rain+noise+low-resolution | rain → noise → low resolution | scene, imaging, imaging | 10 | 20 |

## 4. 退化参数范围（均显式抽取并记录）

| 退化 | 参数 | 范围 |
|---|---|---|
| rain | value | [50, 100) 整数 |
| haze | A, beta | A∈[0.7,1.0), beta∈[0.6,1.8) |
| dark | darken_type + shift/gamma/dst_max | constant shift[30,50) / gamma[0.5,0.7) / linear[100,150) |
| motion blur | severity | {0,1,2} |
| noise | noise_type + sigma/scale | Gaussian sigma[20,50] / Poisson scale[1,3] |
| low resolution | 无 | 固定 0.25×（MATLAB imresize） |
| jpeg | quality_factor | [10, 30) 整数 |

**种子**：`master_seed = 20261010`，每个样本 `random_seed = master_seed + 样本序号`（0 起），
生成前 `np.random.seed(random_seed)`。

**可复现说明**：motion blur 的 `angle`、rain 的 `length/angle/noise` 由退化函数内部
`np.random` 抽取，受 `random_seed` 控制；重跑同 seed 可复现（见 `degradation_config.yaml`）。

## 5. 文件

```
mair_benchmark_v1/
├── dataset_card.md               # 本文件
├── degradation_config.yaml       # 退化配置 + 划分 + 种子
├── validation_manifest.csv       # 80 行
├── test_manifest.csv             # 160 行
├── hash_manifest.csv             # 270 个文件（30 HQ + 240 LQ）的 sha256
├── generate_dataset.py           # 生成脚本（可复现）
├── check_dataset.py              # 校验脚本
├── validation/hq/  (10)  +  validation/lq/  (80)
└── test/hq/        (20)  +  test/lq/        (160)
```

### manifest 列

`sample_id, split, hq_id, hq_path, lq_path, hq_sha256, lq_sha256,
degradation_sequence, degradation_parameters, random_seed, width, height`

其中 `degradation_sequence` 与 `degradation_parameters` 为 JSON 字符串（平行列表）。

## 6. 校验

```bash
cd /root/autodl-tmp/MAIR
/opt/miniconda/envs/diffbir/bin/python experiments/datasets/mair_benchmark_v1/check_dataset.py
```

校验项：① validation/test HQ 哈希无交集；② LQ 文件无重复（路径+内容哈希）；③ 同一 HQ 不跨 split；
④ 文件全部存在且可读；⑤ 尺寸符合退化配置（含 low-resolution 的为 HQ 0.25×）；⑥ manifest 哈希与磁盘一致；
⑦ hash_manifest 覆盖所有文件且无重复。**当前结果：ALL CHECKS PASSED。**
