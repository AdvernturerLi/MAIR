# MAIR 复现

> **2026-10-15 更新**：当前实现已切换到新版 IJCV Algorithm 3，默认使用作者公开经验、GPT 质量比较并允许保留输入，支持指令更新退化与剩余退化上下文。
> 最新使用方法、测试及未完成条件见 [IJCV2026_REPRODUCTION.md](IJCV2026_REPRODUCTION.md)。当前仍有工具适配/权重与服务器验证缺口，不能声明“完整复现”。
> 已在学校 A100 服务器跑通 DeepSeek 替代模型的真实单图流程；MAXIM 去雨、去模糊已在独立 JAX 环境验证，76 项离线测试通过。其他工具/权重缺口见最新说明。
>
> 下文保留的是 **2026-10-10 旧版说明**；其中默认 no-experience、最多三个候选、经验未公开等描述不适用于当前算法版本。不要用旧 v1 基线运行器运行新算法。

复现 [MAIR](https://arxiv.org/abs/2503.09403)（Multi-Agent Image Restoration, IJCV），
基于 AgenticIR 的工具与 LLM 基础设施。本仓库忠实实现论文第 3 节的
**三阶段退化先验 + 两级多智能体（Scheduler / Expert）+ 工具注册表**。

> **复现档位**：当前为「资源受限（available-tools）配置」——规划/选工具模型用
> DeepSeek 替代 GPT-4o，工具集由运行时可用性检查决定，经验模块为无经验
> （no-experience）档位。详见 [REPRODUCTION_STATUS.md](REPRODUCTION_STATUS.md)。

## 与 AgenticIR 的差异

| 特性 | AgenticIR | MAIR（本仓库） |
|---|---|---|
| 调度 | 单 agent，LLM + 经验库决定顺序 | **三阶段退化先验**约束阶段顺序 + 阶段内 LLM |
| 智能体 | 单 agent（IRAgent） | **两级**：Scheduler（粗感知+规划）+ Expert（每退化一个，细感知+选工具+反思） |
| 工具选择 | 跑全部工具再反思 | **工具注册表**：GPT-4o 读注册表选候选工具 |

## 三阶段退化先验

退化按 `scene(雨/雾/低光) → imaging(噪声/模糊/低分辨率) → compression(JPEG)` 发生，
修复逆序：

```
Stage 1 · compression：jpeg compression artifact removal
Stage 2 · imaging：denoising / motion deblurring / defocus deblurring / super-resolution
Stage 3 · scene：deraining / dehazing / brightening
```

阶段顺序由确定性约束保证（`stage_prior.py`），阶段内排序才交给 LLM（可带经验）。

## 目录结构

```
MAIR/
├── main.py            # 入口：调度器感知规划 → 专家顺序执行 + 轨迹
├── stage_prior.py     # 三阶段退化先验 + 退化↔子任务映射
├── scheduler.py       # Level 1 调度器（粗感知 + 三阶段规划 + 指令）
├── expert.py          # Level 2 专家（细感知 + 选工具 + 反思 + 阈值停止）
├── registry.py        # 工具注册表 + 运行时可用性检查
├── instruction.py     # 用户指令结构与"保留/仅处理某退化"确定性解析
├── trajectory.py      # 机器可读运行轨迹（M10）
├── check_tools.py     # 工具盘点：code / env / weights
├── prompts.py         # MAIR prompt
├── mair_batch_eval.py # 批跑评测（PSNR/SSIM）
├── executor/          # 复用自 AgenticIR（工具调用层；tools/ 软链接另行部署）
├── llm/               # 复用自 AgenticIR（GPT4 / DepictQA）
├── utils/             # 复用自 AgenticIR
└── tests/             # 离线测试（不启动 GPU）
```

## 运行

### 1. 准备环境（复用 AgenticIR 的 conda 环境）
- `diffbir`：torch 1.13.1+cu117，跑调度器/专家/工具
- `depictqa`：transformers 4.35.0 + peft 0.3.0 + accelerate 0.25.0，跑 DepictQA 评估
- 工具 conda 环境：swinir/restormer/mprnet/xrestormer/hat/fbcnn 等

### 2. 起 DepictQA 两个评估服务（5001 / 5002）

### 3. 软链接工具 repo
```bash
ln -sfn <AgenticIR>/executor/*/tools executor/*/tools
```

### 4. 配置

复制配置并填 key。**当前可用（替代模型）**：

```bash
cp config.example.yml config.yml   # DeepSeek（OpenAI 兼容接口）
```

**论文同款（paper-faithful）**：

```bash
cp config.gpt4o.example.yml config.yml   # GPT-4o（仍用本地 DepictQA）
```

### 5. 跑单张图

```bash
cd MAIR

# 资源受限配置（默认 no-experience，DeepSeek）
python main.py --input 输入图 --output 输出目录

# 论文同款配置（GPT-4o）
python main.py --input 输入图 --output 输出目录 --config config.yml

# 带用户指令（保留某退化 / 只处理某退化）
python main.py --input 输入图 --output 输出目录 --instruction "保留雾但去雨"

# 带经验（需含 distilled 与 source 字段；无文件会自动切 no-experience）
python main.py --input 输入图 --output 输出目录 --experience 经验.json
```

每次运行会在 `输出目录/trajectory.json` 保存机器可读轨迹（输入哈希、模型/配置、
粗感知、指令、阶段计划、每个专家的细感知/候选/工具输出/停止原因/错误/耗时）。

### 6. 工具盘点（服务器上跑，确认哪些工具真正可用）

```bash
python check_tools.py
```

### 7. 离线测试（不启动 GPU）

```bash
python -m pytest tests/ -q
```

## 配置项

| 键 | 含义 | 默认 |
|---|---|---|
| `STOP_THRESHOLD` | 专家停止阈值，严重度严格低于该值即停（论文示例 `low`） | `low` |
| `USE_SIZE_HEURISTIC_FOR_SR` | 是否用尺寸启发式判低分辨率（非论文正文，见 M09） | `true` |
| `LOW_RES_MAX_SIDE` | 低分辨率最长边阈值 | `300` |
| `LOW_RES_AS_BLUR` | 超分专家用 `blur` 严重度代理"低分辨率"；`false` 时退回中性 `medium`（DepictQA 无法评估 LR，避免崩溃） | `true` |

## 经验文件格式

```json
{
  "source": "offline distillation of 13000+ tool sequences (MAIR Tab.1)",
  "distilled": "Stage 2: 先 deblur 再 denoise 再 super-resolution ……"
}
```

- `distilled` 必须非空；`source` 记录经验来源（离线总结自哪些尝试记录）。
- 未提供经验文件 / 文件缺失 / `distilled` 为空时，**显式切到 no-experience 档位**，
  不会把空经验伪装成论文默认 MAIR。

## 已知待核实项（如实标注，不臆造）

- **低分辨率判定**（M09）：论文正文未给出精确判据，当前用 AgenticIR 的
  `<300px` 尺寸启发式，并以 `blur` 严重度代理"低分辨率"。在无法核对作者
  实现/补充材料前，不声称与论文完全一致。
- **经验蒸馏**：论文的经验来自离线总结的「工具序列+分数」。当前未产出论文等价的
  经验数据，故保持 no-experience 档位，不生成虚构经验。
- **替代模型**：规划/选工具默认用 DeepSeek（非 GPT-4o），DepictQA 仍用本地服务。
  无法取得论文同款时，报告标注为"替代模型复现"。
