# MAIR 复现

复现 [MAIR](https://arxiv.org/abs/2503.09403)（Multi-Agent Image Restoration, IJCV），基于 AgenticIR 的工具与 LLM 基础设施。

## 与 AgenticIR 的差异

| 特性 | AgenticIR | MAIR（本仓库） |
|---|---|---|
| 调度 | 单 agent，LLM + 经验库决定顺序 | **三阶段退化先验**约束阶段顺序 + 阶段内 LLM |
| 智能体 | 单 agent（IRAgent） | **两级**：Scheduler（粗感知+规划）+ Expert（每退化一个，细感知+选工具+反思） |
| 工具选择 | 跑全部工具再反思 | **工具注册表**：GPT-4o 读注册表选候选工具 |

## 三阶段退化先验

退化按 `scene(雨/雾/低光) → imaging(噪声/模糊/低分辨率) → compression(JPEG)` 发生，修复逆序：

```
Stage 1 · compression：jpeg compression artifact removal
Stage 2 · imaging：denoising / motion deblurring / defocus deblurring / super-resolution
Stage 3 · scene：deraining / dehazing / brightening
```

## 目录结构

```
MAIR/
├── main.py            # 入口：调度器感知规划 → 专家顺序执行
├── stage_prior.py     # 三阶段退化先验 + 映射
├── scheduler.py       # Level 1 调度器
├── expert.py          # Level 2 专家
├── registry.py        # 工具注册表（33 个工具表单）
├── prompts.py         # MAIR prompt
├── mair_batch_eval.py # 批跑评测（PSNR/SSIM）
├── executor/          # 复用自 AgenticIR（工具调用层；tools/ 软链接另行部署）
├── llm/               # 复用自 AgenticIR（GPT4 / DepictQA）
├── utils/             # 复用自 AgenticIR
└── pipeline/prompts.py
```

## 运行

```bash
# 1. 准备环境（复用 AgenticIR 的 conda 环境 diffbir / depictqa）
# 2. 起 DepictQA 两个评估服务（5001 / 5002）
# 3. 软链接工具 repo：ln -sfn <AgenticIR>/executor/*/tools executor/*/tools
# 4. 跑单张图
cd MAIR
python main.py --input 输入图 --output 输出目录 [--no-retrieval]

# 批跑（对比实验）
python mair_batch_eval.py
```

## 依赖环境

- `diffbir`：torch 1.13.1+cu117，跑调度器/专家/工具
- `depictqa`：transformers 4.35.0 + peft 0.3.0 + accelerate 0.25.0，跑 DepictQA 评估
- 工具 conda 环境：swinir/restormer/mprnet/maxim/hat/fbcnn/xrestormer 等（软链接到 diffbir）
