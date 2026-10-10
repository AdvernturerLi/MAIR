# MAIR 复现环境基线（冻结版本）

> **版本名：`MAIR-available-noexp-v1`**
> 含义：资源受限工具集 + 替代规划模型(DeepSeek) + 无经验(no-experience)。
> 冻结日期：2026-10-10。本文件是所有后续实验的共同基线，改动请新起版本名。

## 1. 代码与配置

| 项 | 值 |
|---|---|
| MAIR Git commit | `a16a66136ee5387f5fbb731d4a972d6b6bc46be0` |
| 规划/选工具模型 | `deepseek-v4-flash-vision-exp`（**替代** GPT-4o，论文配置未取得） |
| provider | `https://api.deepseek.com` |
| temperature | `0.0` |
| max_tokens | `8192` |
| API key | 已脱敏，不入库（支持 `OPENAI_API_KEY` 环境变量覆盖） |
| STOP_THRESHOLD | `low`（严重度严格低于该值即停） |
| 低分辨率判据 | `USE_SIZE_HEURISTIC_FOR_SR=true`，`LOW_RES_MAX_SIDE=300`，`LOW_RES_AS_BLUR=true`（blur 代理，非论文正文） |
| 经验库 | **无经验（no-experience）**，未提供经验文件时显式切换 |

## 2. 硬件与框架

| 项 | 值 |
|---|---|
| GPU | NVIDIA A100-PCIE-40GB（40 GiB），驱动 580.178.04 |
| PyTorch (diffbir/depictqa) | `1.13.1+cu117` |
| CUDA / cuDNN | 11.7 / 8500（torch 内置，系统无 nvcc） |
| 工具 conda 环境 | swinir/restormer/mprnet/xrestormer/hat/fbcnn（软链接→diffbir）；diffbir=主环境 |

## 3. DepictQA（感知/反思模型）

| 项 | 值 |
|---|---|
| 基座 | Vicuna-v1.5-7B（`weights/vicuna-7b-v1.5/`，2 shards） |
| 权重 | `weights/delta/degra_eval.pt`（评级）、`weights/delta/DQ495K_Abstractor.pt`（对比）、`weights/ViT-L-14.pt`（CLIP） |
| 代码 commit | `06583c6ff2a0e6ae8095d24ee853b792f1e3293d` |
| 服务 | `app_eval.py`(:5001) + `app_comp.py`(:5002)，Flask `debug=False`，`fp16+low_cpu_mem_usage`，串行启动，驻留约 28G |

## 4. 实际可用工具清单（check_tools.py，2026-10-10）

| 退化（subtask） | 可用工具 |
|---|---|
| super-resolution | diffbir, xrestormer, swinir_gan, swinir_psnr, hat |
| denoising | xrestormer, swinir_15, swinir_50, mprnet, restormer |
| motion deblurring | restormer, mprnet, xrestormer |
| defocus deblurring | restormer（drbnet/ifan 缺 env+权重） |
| deraining | xrestormer, restormer, mprnet |
| dehazing | xrestormer（ridcp/dehazeformer 缺 env；maxim 缺权重） |
| brightening | histogram_equalization, gamma_correction, constant_shift（纯 OpenCV） |
| jpeg compression artifact removal | fbcnn_blind, fbcnn_5, fbcnn_90, swinir_40 |

可用退化（≥1 工具）：low resolution / noise / motion blur / defocus blur / haze / rain / dark / jpeg compression artifact（**8 类**）。

> 权重文件在软链接的 AgenticIR 工具 repo 内；SwinIR/MAXIM/DiffBIR/RIDCP/DRBNet/IFAN
> 的各自权重文件名在 `executor/multitask_tools.py` 及对应 `__init__.py` 中声明（M07）。

## 5. 复现档位声明

- 当前为 **资源受限（available-tools）配置**：工具集由运行时 code+env+weights 检查决定；
  规划/选工具用 DeepSeek 替代 GPT-4o；经验为 no-experience。
- 未取得的论文等价物：GPT-4o 原始服务、论文的离线经验蒸馏产物。
- 低分辨率 `<300px` 与 blur 代理为启发式，非论文正文（M09 待核实）。

## 6. 测试与验证

- `python -m pytest tests/ -q` → **54 passed**（服务器 diffbir 环境与本地均通过）。
- 冒烟测试（tiger_degraded.png）：三阶段计划 `[jpeg removal, deraining]` 正确；
  指令"保留雨"→ 计划只剩 jpeg；轨迹完整。
- 原始输出：`check_tools.txt`、`pytest.txt`（同目录）。
