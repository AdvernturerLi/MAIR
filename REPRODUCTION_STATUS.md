# MAIR 复现核查与修复记录

> 2026-10-10。对照《MAIR 论文复现核查与修复任务书》M01–M10 的逐项改动状态。
> 本文只陈述「代码层面已改到什么程度」；权重/服务是否真正部署以 `check_tools.py`
> 与服务器冒烟测试为准。

## 0. 复审修复记录（2026-10-10 第二轮）

| # | 问题 | 修复 |
|---|---|---|
| 1 | `--force` 可删任意非空目录 | `main.py:_require_empty_output` 增加安全校验，禁止删除项目根、输入图（及其所在目录）及它们的任意祖先目录 |
| 2 | 尺寸触发的超分绕过可用工具检查 | `scheduler.py` 超分分支同样先查 `available_tools("super-resolution")` |
| 3 | `LOW_RES_AS_BLUR:false` 导致超分专家 IndexError | `depictqa.py` 退回中性 `medium`（非空），`expert.py:_fine_perceive` 兜底防空列表 |
| 4 | "只保留雾"被反向解析成"只处理雾" | `instruction.py:parse_only` 跳过含"保留"标记的子句，区分「只处理」与「只保留」 |
| 5 | 工具盘点把缺权重的多变体误判可用 | `tool.py` 新增 `required_weight_files`，SwinIR/MAXIM/DiffBIR/RIDCP/DRBNet/IFAN 声明各自权重文件 |
| 6 | 轨迹可能记错模型/异常时不落盘 | `main.py` 从已构造客户端取真实模型；感知/规划/执行异常时也保存失败轨迹 |

## 一、逐项改动清单（M01–M10）

| ID | 状态 | 改动位置 | 说明 |
|---|---|---|---|
| M01 | ✅ 完成 | `stage_prior.py`（已有）+ 测试 | 三阶段先验本已正确；补充 `DEGRADATION_TO_SUBTASK` 映射；`tests/test_stage_prior.py` 覆盖压缩→成像→场景的逆序。 |
| M02 | ✅ 完成 | `scheduler.py:190-230`、`prompts.py` | 阶段内规划的 GPT 请求现传入原图 `img_path`，prompt 含粗感知（文本 severity）、指令与阶段经验；`tests/test_scheduler.py::test_plan_*_multimodal` 用假客户端断言请求含原图。 |
| M03 | ✅ 完成 | `instruction.py`（新）、`main.py`、`scheduler.py`、`expert.py` | 单图入口新增 `--instruction`；"保留/只处理某退化"经确定性解析落实到任务清单（不只放进 prompt）；指令全文同时进入 Scheduler 与 Expert 的 GPT 请求。 |
| M04 | ✅ 完成 | `scheduler.py:_load_experience` | 经验要求非空 `distilled` + `source`；无文件/缺失/空时显式切 no-experience 档位并记录；`tests/test_experience.py` 覆盖。 |
| M05 | ✅ 完成 | `expert.py:_select_tools`、`prompts.py` | 候选选择请求现包含图像、细粒度 severity、指令、工具表单四类信息；`test_select_tools_passes_instruction_to_prompt` 覆盖。 |
| M06 | ✅ 完成 | `expert.py` | 停止阈值改为可记录参数 `stop_threshold`（默认 `low`，严格低于即停）；失败时**所有有效失败候选**两两比较（不再只看最低严重度组）；`test_compare_all_failed_candidates` 覆盖。 |
| M07 | ✅ 完成 | `registry.py:available_tools`、`executor/tool.py:availability`、`check_tools.py` | 移除硬编码 `WORKING_TOOLS/WORKING_DEGRADATIONS`；运行时按 code+env+weights 可用性检查选工具；`check_tools.py` 输出盘点；新增工具经检查即可被选，不靠改白名单。 |
| M08 | ✅ 完成（替代档位） | `config.example.yml`、`config.gpt4o.example.yml`、`main.py` | 提供 DeepSeek（替代）与 GPT-4o（论文同款）两套配置；每次运行在轨迹记录 provider/model/参数；DepictQA 仍本地服务。论文同款权重/经验不可得时标注"替代模型复现"。 |
| M09 | ⚠️ 待核实（已配置化） | `scheduler.py`、`llm/depictqa.py`、`config.example.yml` | 低分辨率 `<300px` 启发式与 `blur` 代理均加配置开关（`USE_SIZE_HEURISTIC_FOR_SR`/`LOW_RES_MAX_SIDE`/`LOW_RES_AS_BLUR`），README 明确标注非论文正文。作者实现/补充材料未核。 |
| M10 | ✅ 完成 | `trajectory.py`（新）、`main.py`、`scheduler.py`、`expert.py` | 每张图保存 `trajectory.json`：输入哈希、模型/配置、粗感知、指令、经验来源、阶段计划、专家细感知/候选/工具输出/停止原因/错误/耗时。 |

## 二、复现档位

1. **论文配置（paper-faithful）**：`config.gpt4o.example.yml`（GPT-4o）+ 本地 DepictQA
   + 实际可用的完整工具集 + 真实经验文件。当前**经验文件未产出**、部分权重未部署，
   因此该档位暂不能完整声明。
2. **资源受限配置（available-tools）**：默认档位。DeepSeek 替代 GPT-4o，工具集由
   `check_tools.py` 的可用性检查决定，经验为 no-experience。
3. **消融配置**：`--no-retrieval`（无经验）、`--instruction ""`（无指令）等，均显式
   并写入轨迹。

## 三、复现声明（当前）

> 复现了 MAIR 的三阶段退化先验与两级多智能体流程（Scheduler 粗感知+规划、
> Expert 细感知+候选工具选择+反思），并贯通了用户指令、多模态规划请求、
> 经验校验与机器可读轨迹；当前使用替代规划模型（DeepSeek）与本地 DepictQA，
> 工具集由运行时可用性检查决定（约 7 类退化可用，离焦模糊待权重部署），
> 经验模块仍为无经验（no-experience）档位。

## 四、测试

```bash
python -m pytest tests/ -q        # 42 项离线测试（不启动 GPU）
python check_tools.py             # 服务器上盘点工具 code/env/weights
```

## 五、仍无法忠实复现的环节

- 论文的**离线经验蒸馏**（Tab.1 的 13000+ 工具序列 + 分数 → GPT-4o 总结文本）未产出。
- 低分辨率判据（M09）的作者精确实现未核对，暂用启发式。
- GPT-4o 原始权重/服务不可得，规划与选工具用 DeepSeek 替代。
