# 独立经验构建

用户要求 AgenticIR 和 MAIR 各自构建、各自保存经验。它们不采用同一评价目标，不能把同一经验换文件名当成两种机制的复现。

## 已准备的流程

`pilot.py` 将工具原始执行前缀缓存复用，并保留每个节点的日志、输出哈希和重试记录。所有数据来自与 test160 按 HQ ID、内容哈希均不重合的 validation 划分；不使用两个 baseline 的测试结果蒸馏经验。

AgenticIR：本地 DepictQA 评价完整序列输出的原始退化残留程度，medium/high/very high 计为失败，分别统计每种顺序的失败率，生成 `agenticir/fail_rate.json`，再蒸馏为独立 `agenticir/schedule_experience.json`。

MAIR：要求 PSNR、SSIM、LPIPS、DISTS、MANIQA、CLIP-IQA、MUSIQ 七项均可用。对同一输入的候选结果分别进行 z-score 标准化，LPIPS/DISTS 翻转方向后求和，生成 `mair/scored_sequences.json` 和 `mair/order_scores.json`，再蒸馏为独立 `mair/schedule_experience.json`。在同一输入候选池内拟合标准化参数是本实现明确记录的选择，原作者确切归一化样本池未公开。

MAIR 蒸馏仅形成阶段内排序建议，不改变压缩→成像→场景的跨阶段约束。论文公开经验文件保持原样，试跑经验 kind 为 independently_distilled_controlled_validation_pilot，不能标成原作者经验或五个原始真实数据集的13000多序列实验复现。

## 规模

验证集为 80 张 LQ，由 10 张 HQ、8 类组合构成，最后一组含雨＋噪声＋低分辨率三种退化。完整展开并缓存公共执行前缀后仍有 **10780 次工具调用、8720 个完整序列输出**。这里只计算修复生成，不含 DepictQA、七项 IQA 和蒸馏成本。

准备的试跑仅选择 HQ 084、054，各自的噪声＋运动模糊、雨＋雾，共 4 张 LQ、196 个工具调用节点、160 个最终候选。只覆盖这两个组合，不支持把试跑知识推广为所有退化的充分结论。

## 数据边界与当前状态

验证图像只交给本地修复模型、127.0.0.1 DepictQA 和本地 IQA。蒸馏调用为 `gpt(prompt=汇总统计文字)`，不提供 img_path；汇总不含图像内容。

**当前试跑尚未启动，用户要求先分析规模。用户明确选择暂不外传，只做服务器本地探索和统计。** 已将脚本改为默认不蒸馏、不调用外部语言模型；`--distill` 为默认关闭的独立开关，未经后续明确授权不得启用。此前两次自动审核拦截的原流程不会启动。

服务器脚本已上传，但没有运行探索或蒸馏。模型包 pyiqa 0.1.16 已安装，完整七指标的实际权重加载尚未验证，不把缺失指标填零或静默降为较少指标。

5 项测试覆盖：双退化和三退化规模计算、外部蒸馏只接收 prompt 文本、本地 DepictQA 地址限定、外部蒸馏必须显式开启且默认关闭。运行：`python -m pytest experience_learning/tests -q`。


## 论文逻辑链修订

以[PAPER_EXPERIMENT_PROTOCOL.md](PAPER_EXPERIMENT_PROTOCOL.md)为当前实验设计：补充匹配工具顺序、三任务条件规则及MAIR先验内部消融。旧小试仅作工程验证，外部蒸馏仍关闭；扩大探索尚未启动。


## 扩大探索已启动

当前运行目录为 `/root/autodl-tmp/experience_validation_pairwise7_10hq_20261016_ridcp_excluded`。10个验证HQ×7种双任务，70输入；RIDCP排除，3140次调用、2540叶子，复用小试178节点。`progress.json`为探索/评价状态，`distillation_progress.json`为DeepSeek文本蒸馏队列状态。测试集未参与经验构建。三任务条件实验尚未启动。
