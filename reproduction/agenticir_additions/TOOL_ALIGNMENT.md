# MAIR 工具对齐（2026-10-15）

对照 MAIR `test160_noexp_20261015/source_snapshot` 的实际可用论文工具子集，固定为 32 个工具。清单和 17 个包装/配置文件的 SHA-256 见 `tool_alignment.json`。

服务器独立目录：`/root/autodl-tmp/AgenticIR_mairtools_20261015`。服务器 `TOOL_ALIGNMENT_REPORT.json` 已验证所有工具代码、解释器和权重文件可用，包装文件与 MAIR 快照一致。MAXIM 使用同一个独立 JAX 环境；模型仓库与权重只读复用原目录。此项检查不代表全部工具再次完成端到端 GPU 测试。

AgenticIR 的规划、搜索、比较和回溯实现未改动。工具集合通过 executor 初始化时读取清单固定，不再包含 IFAN、未指定变体的 DehazeFormer 或当前不可用工具。保留三个准确指定的 DehazeFormer 变体，以及可用的 MAXIM 去雨、运动去模糊。

2026-10-15 已按用户授权启动无经验测试集 160 张：入口是 `run_manifest.py`，而非会二次缩减工具集合的旧 `quick_run.py`、`batch_eval_llm.py`。正式目录为 `experiments/AgenticIR_noexp_test160_deepseek-v4-flash-vision-exp_20261015`。

采用与 MAIR 相同的固定数据清单、原始图像尺寸、DeepSeek 模型配置及评测规则：成功生成可读取图片即保留；尺寸不同仅在内存中用 OpenCV INTER_CUBIC 对齐参考图进行 RGB PSNR/SSIM 计算，记录原尺寸与 metric_resized，原图不变。

规划、搜索、退化等级判断和回溯策略保持 AgenticIR 原逻辑；新增的运行防护包括安全解析 JSON/元组、接口超时重试、工具失败日志与下一候选恢复、失败产物移出图片树以避免 HTML 渲染连锁异常。整张最多在独立进程中尝试三次；仍失败会停止批次并保留全部诊断，不用输入图冒充成功结果。每张总超时为 7200 秒。

质量比较仍使用原 DepictQA 比较模型及检查点，按需加载，比较后释放显存；同精度 float16 直接加载至 GPU，以降低主机内存峰值。退化评估保持原有 DepictQA 服务。两者的模型和任务不同，运行服务配置与比较应用校验值记录于 `runtime_services.json`。
