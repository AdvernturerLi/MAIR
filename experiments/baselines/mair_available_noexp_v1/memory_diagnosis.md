# 显存诊断报告（MAIR-available-noexp-v1-partial50）

> 结论先行：**不是「40G 显存不足」这么简单，根因是「DepictQA 两服务随批跑碎片化累积 + 生成式 DiffBIR 峰值 ~8G 挤在同一张 40G 卡」**。属于第 3 类「多样本运行造成显存累计」。

## 1. 批处理停止位置

- 停止在 **第 50 个样本之后**（50/80），由我**手动 `pkill` 停止**，不是进程自身崩溃。
- 停止原因：连续出现 X-Restormer OOM（小工具都放不下），判断继续跑会拖垮剩余低分辨率样本，遂停下重启 DepictQA。

## 2. 停止时正在运行的工具

- 停止时正在跑 `023__low-light+noise` 之前的 `040__rain+noise+low-resolution`，其**超分/去雾子任务里 X-Restormer OOM**（`torch.cuda.OutOfMemoryError ... 766.00 MiB ... 514.38 MiB free`）。

## 3. 最后成功 / 第一个未完成

- 最后一个完成样本：`023__rain+haze`（success）
- 第一个未完成样本：`023__low-light+noise`

## 4. CUDA OOM 原始错误（两类）

- **DiffBIR**（生成式超分）：`CUDA out of memory. Tried to allocate 352.00 MiB (GPU 0; 39.49 GiB total; 7.85 GiB already allocated; 62.38 MiB free; 7.93 GiB reserved)`
- **X-Restormer**（判别式去雾）：`CUDA out of memory. Tried to allocate 766.00 MiB (GPU 0; 39.49 GiB total; 5.70 GiB already allocated; 514.38 MiB free; 7.43 GiB reserved)`

## 5. 进程退出方式

- 手动 `pkill`（SIGTERM/SIGKILL），退出码为信号值，非 Python 异常退出。运行器本身的异常处理（专家捕获工具 OOM → 回退）**一直在正常工作**，没有进程被系统 OOM-killer 杀死（dmesg 无 oom-kill 记录）。

## 6. OOM 前可获得的显存数据

| 时刻 | GPU used | 说明 |
|---|---|---|
| 重启 DepictQA 后 | ~28.6 GiB | 干净基线（两服务 2×14G） |
| 跑 ~7 张后 | ~31.4 GiB | DepictQA 累积 ~3G |
| X-Restormer OOM 时 | ~39.5 GiB | 31.4G(DepictQA) + 5.7G(X-Restormer) + 碎片，只剩 514MB |

## 7. 显存是否随样本逐渐增长

**是。** DepictQA 两服务从 28G 涨到 31.4G（约 0.4~0.5G/样本），与工具子进程无关（工具每次跑完退出释放）。这是 DepictQA 服务进程内部的碎片化累积（PyTorch caching allocator 不归还）。

## 8. 根因分类（四选一）

1. ~~单样本工具峰值不足~~：不是。单个工具（DiffBIR ~8G / X-Restormer ~6G）在干净基线（28G）下都能 fit（28+8=36<40）。
2. ~~异常后资源未释放~~：不是。工具是子进程，OOM 后子进程退出、显存释放；运行器继续下一候选。
3. **多样本运行造成显存累计**：**是主因**。DepictQA 服务进程碎片化累积（28→31.4G），压缩了工具可用空间，最终连 X-Restormer 都放不下。
4. ~~进程被系统杀死~~：不是。无 oom-kill。

## 9. DiffBIR OOM 后是否执行 empty_cache

- 没有显式调用 `torch.cuda.empty_cache()`。但因为工具在**子进程**里，OOM 后子进程退出，其显存被 OS 回收，等价于清空。**真正没被回收的是 DepictQA 服务进程内的碎片**，empty_cache 对其无效（需重启服务）。

## 10. 残留模型/tensor/子进程

- 残留主占用：**DepictQA 两服务**（eval 5001 + comp 5002）驻留 28G+ 且累积，是唯一长期占用者。
- 无残留工具子进程（每次跑完退出）。
- 缓解措施（记忆里已有）：`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256` + 定期重启评估器。

## 11. 建议

- 短期：续跑时**每 ~8 张重启 DepictQA**（或 GPU>30G 就重启）。
- 中期：把 DiffBIR 从 SR 候选排除（像 MAXIM 一样），消除 ~8G 峰值工具。
- 长期：双卡分流（DepictQA 卡0 + 工具卡1），或量化 DepictQA（此前 8-bit 已因 torch1.13.1+peft0.3.0 无支持而放弃）。
