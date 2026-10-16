# IJCV 2026 复现修订

算法版本：`mair-ijcv2026-algorithm3-v2`。依据本地 `论文/mair最新.pdf`，DOI 10.1007/s11263-026-02792-5，Algorithm 1–3 与 Sec.8–9。

用户已确认允许使用替代多模态模型，不要求 GPT-4o 同配置。服务器现用 `deepseek-v4-flash-vision-exp`，调用地址为 `https://api.deepseek.com`。所有结果必须标注替代模型与实际工具子集。

## 已实现

- Scheduler 按三阶段规划，保留阶段内 GPT 排序；每个非空阶段包含前序计划、图像、感知结果、用户指令、经验。
- 有用户指令时先由 GPT 更新待处理退化，再校验名称、去重与工具可用性。关键词解析不再是自然语言理解的唯一入口。
- 默认加载 `experience/paper_ijcv2026.json`，记录来源与 SHA-256。无经验消融用 `--no-retrieval`。显式经验文件缺失、格式错误、缺来源会报错，避免静默改变实验配置。
- Expert 选择收到图像、指令、当前程度、剩余退化、工具表单；优先单工具，不再人为限定最多三个。
- Algorithm 3：每个候选独立处理专家输入；低于 low 即返回；否则由 GPT 对比当前最佳与新输出。当前最佳初始为输入；former/neither 均保留当前最佳。所有执行失败明确抛错。
- 所有专家统一遵守输入停止阈值，去掉原来超分的特例。
- Sec.9 的 16 个工具表单示例中，15 个已映射为有来源的注册表单。第二个 DiffBIR 表单原文未指明 checkpoint，存于 `paper_registry.UNRESOLVED_DIFFBIR_FORM`，没有臆造变体映射。其他本地表单明确标注自行整理。
- 严格配置检查 GPT-4o、经验、Table 15 工具可用性；缺条件直接失败。旧 v1 基线运行器禁止把新算法结果追加进历史数据；历史指标计算仍可使用。

## 经验机制如何运行

1. 离线构造：作者在五组真实验证集上进行 13000 多条工具执行序列的尝试。使用 PSNR、SSIM、LPIPS、DISTS、MANIQA、CLIP-IQA、MUSIQ 的标准化综合评分评估方案（Sec.3.2、4.1）。
2. 离线总结：GPT-4o 读取执行序列及对应分数，总结通常有效的阶段内顺序（Sec.3.3）。这一步与处理测试图像的推理分开。
3. 运行时注入：Scheduler 把固定的经验文本和感知、指令一起放进规划 prompt，让 GPT 决定阶段内顺序。经验不是硬编码排序，也不是相似图片检索或在线更新的成功/失败记忆库。代码保留的 `with_retrieval` 名称是历史接口，不表示论文有向量检索。

Sec.8 已公开的经验内容是：

- 阶段 2：同时涉及去模糊、超分、去噪时，通常先去模糊，再超分，再去噪；不涉及超分时，先去噪，再去模糊。
- 阶段 3：先去雨，再去雾及低光增强。原文没有规定去雾与低光增强之间的唯一顺序。

这些是通常有效的建议，由 LLM 使用；三阶段之间的压缩→成像→场景顺序则由代码约束。

采用作者公开的经验文本足以复现该推理输入，但不等于独立重做了作者的离线经验构建实验。作者的原始尝试记录、精确工具版本及蒸馏提示未在本地提供。

## 运行

在 MAIR 根目录中运行，使用新输出目录。不得把本版结果与旧版 available-noexp-v1 混为同一配置。

```bash
python check_tools.py
python main.py --config config.gpt4o.example.yml --input INPUT.png --output output/ijcv2026_case01
```

凭据由 `OPENAI_API_KEY` 环境变量提供。严格配置要求有效 GPT-4o 服务及完整工具部署；现有 DeepSeek config.yml 可用于资源受限验证，但不能称论文同配置。不要把含密钥的 config.yml 放进迁移包。

```bash
# 显式资源受限/替代模型运行：使用现有 config.yml
python main.py --config config.yml --input INPUT.png --output output/ijcv2026_available_case01
# 无经验消融（不能与 REQUIRE_PAPER_RESOURCES:true 同时启用）
python main.py --config config.yml --no-retrieval --input INPUT.png --output output/ijcv2026_noexp_case01
```

## 尚未完成，不能声明完整复现

Table 15 中第二个 DiffBIR 训练变体、DWGAN、Retinexformer、CoTF 没有完整可验证的适配与部署；MAXIM 缺少对应权重。注册表单不等于实现了工具；`paper_resources.inventory()` 将缺少适配和缺少部署分别报告。DehazeFormer Outdoor-B/Indoor-L/Indoor-B 已完成三个独立变体的适配，并在学校服务器分别真实推理成功。

MAXIM 复查修正：去雨、去模糊权重文件存在且 NPZ 结构完整，对应三个任务曾被注释；去噪权重为截断文件，去雾权重缺失。原 `maxim` 环境没有 jax/jaxlib/flax。已恢复工具注册、增加 NPZ 完整性检查及失败时清理临时输入的逻辑。模板 `ENABLE_MAXIM` 默认 false；服务器专用环境已完成真实 GPU 验证，现有 `config.alternative.runtime.yml` 已设置为 true。由于坏/缺权重仍被过滤，实际只启用去雨和去模糊。

### MAXIM 独立环境验收（已通过）

- 路径：`/root/autodl-tmp/MAIR_ijcv2026_20261015/.venvs/maxim-jax`，没有修改原 conda 工具环境。
- JAX/JAXlib/CUDA plugin 0.4.30、Flax 0.8.4、NumPy 1.26.4、cuDNN 9.1.1.17；完整冻结清单位于服务器 `output/server_validation/maxim/requirements.freeze.txt`。
- 原默认 XLA 卷积自动调优触发 cuDNN 执行错误；最小矩阵乘和两种卷积定位后，对 MAXIM 子进程设置 `XLA_FLAGS=--xla_gpu_autotune_level=0`，GPU 基础算子与完整模型均通过。未修改网络结构或原始权重。这是运行时兼容设置，不能将此耗时与作者环境直接比较。
- `JAX_PLATFORMS=cuda` 强制 GPU，不能静默退回 CPU；`XLA_PYTHON_CLIENT_PREALLOCATE=false` 避免预占大部分显存，与 DepictQA 服务共存。
- 原始权重测试：128×85 图，去雨/去模糊输出均有效，分别约 23.2s/27.3s，包含启动及编译开销。
- 实际 MAIR Tool 适配器：去雨处理 512×340 图，成功输出同尺寸图片，耗时约 25.0s；输入临时目录清理检查通过。
- 验收记录：服务器 `output/server_validation/maxim/runtime_ready.json`。其中记录专用解释器、测试路径、已验证任务和依赖冻结文件 SHA-256。

后续运行使用新版目录及 `--config config.alternative.runtime.yml`。模板默认关闭是为了防止其他尚未安装该环境的机器误启用；不能仅修改开关就宣称 MAXIM 运行可用。

低分辨率检测与其严重度定义没有作者代码可核验；保留 AgenticIR 的尺寸启发式与 blur 代理并写入轨迹。粗感知纳入阈值 medium 同样是实现假设，可通过 PERCEPTION_THRESHOLD 配置。不能将这些未公开细节称为逐行还原。

GPU 服务器 `10.251.171.6:29119` 经用户添加公钥后已成功登录。新版独立部署到 `/root/autodl-tmp/MAIR_ijcv2026_20261015`，原 MAIR 和旧结果不变。服务器为 A100 40GB。GPT-4o 配置文件为该目录的 `config.gpt4o.runtime.yml`，需用户自行填写 key；原 MAIR 配置为 DeepSeek，未将其密钥复制或打印。

## 验证

本机及服务器均有 76 项离线测试通过，含单候选保留输入、GPT 平局、多候选独立输入、成功立即停止、四个候选、剩余退化、指令补充低严重度任务、经验校验与哈希、严格资源检查、真实图片文件的模拟端到端流程，以及模型变体、MAXIM 权重/清理/默认禁用/独立解释器策略。模拟模型与工具不代表论文指标复现。

真实 GPU 工具验证：服务器验证图 `084__rain+haze.png`（1024×680），三个 DehazeFormer 变体均生成有效同尺寸输出，耗时分别约 25.1s、5.1s、8.2s（含各工具进程启动，不作性能结论）。结果位于服务器 `output/server_validation/dehazeformer_variants/results.json`。

DepictQA 已在服务器启动，仅监听 `127.0.0.1:5001`，真实验证图 haze 请求返回 `very high`。服务日志和 PID 位于服务器 `output/server_validation/depictqa.log`、`depictqa.pid`；结果位于 `depictqa_probe.json`。新版 GPT 质量比较不需要启动原先的 DepictQA 5002 比较服务。

真实端到端验证（经用户明确授权将该图像发至 DeepSeek）：`084__rain+haze.png` 缩至 512×340，指令只处理雨和雾并保持身份特征；加载论文公开经验后规划 deraining→dehazing，先选 Restormer，再选 DehazeFormer Indoor-L。每步 DepictQA 从 very high 降至 very low 后停止，最终结果与轨迹落盘，总耗时约 57.9 秒，仅为单张冒烟测试，不是基准性能结论。额外用真实双图 API 比较输入与输出，格式有效、选择 latter。这是模型主观判定，不能替代参考指标或人工质量评价。

服务器输出：`output/server_validation/mair_rain_haze/result.png`、`trajectory.json`，本地副本位于 `experiments/paper_audit_20261015/server_validation/`。当次运行尚未恢复 MAXIM 候选，后续启用它必须使用新实验目录。

Windows 测试依赖装在工作区 `.deps/mair-test`；命令：

```powershell
$env:PYTHONPATH='G:\Image_reconstruct\.deps\mair-test'
python -m pytest tests -q --basetemp=experiments/paper_audit_20261015/pytest-tmp-NEW
```

`--basetemp` 必须指向专用测试临时目录，不能指向含重要文件的目录。
