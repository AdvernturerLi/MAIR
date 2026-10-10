# 续跑计划（MAIR-available-noexp-v1，剩余 30 张）

## 目标

在**不改变实验条件**的前提下，跑完剩余 30 张 validation 样本（见 `unfinished_samples.txt`），
与已完成的 50 张合并成完整 80 张基线。

## 必须保持不变的配置

- MAIR 代码 commit：`0fa952a`（三阶段先验 / Scheduler / Expert / 工具参数一律不动）
- `experience_enabled = false`（no-experience）
- 停止阈值 `STOP_THRESHOLD = low`
- 工具优先级 / 注册表 / prompt：不变
- 数据集 `validation_manifest.csv`、指标协议（PSNR/SSIM/LPIPS/DISTS + 增益）：不变

## 续跑方式（只跑剩余 30 张，不覆盖前 50 张）

```bash
# 1) 启动 DepictQA（串行，带碎片化缓解）
cd /root/autodl-tmp/AgenticIR/DepictQA
source /opt/miniconda/etc/profile.d/conda.sh && conda activate depictqa
export PYTHONPATH=$PWD/src:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
setsid nohup python src/app_eval.py --cfg experiments/agenticir/config_eval.yaml > /root/depictqa_eval.log 2>&1 < /dev/null &
# 等 5001 起来后
setsid nohup python src/app_comp.py --cfg experiments/agenticir/config_comp.yaml > /root/depictqa_comp.log 2>&1 < /dev/null &

# 2) 续跑（--skip-completed 跳过前 50 张）
cd /root/autodl-tmp/MAIR
/opt/miniconda/envs/diffbir/bin/python experiments/baselines/mair_available_noexp_v1/run_baseline.py --skip-completed

# 3) 算指标
/opt/miniconda/envs/diffbir/bin/python experiments/baselines/mair_available_noexp_v1/run_baseline.py --metrics-only
```

## 资源释放规则（需单独 commit 再上）

- 每 ~8 张（或 GPU>30G）重启 DepictQA，`--skip-completed` 保证断点续跑。
- 每个 `sample_id` 最终只有一个正式结果（runner 重跑会先清空该 sample 的 runs 目录，
  并保留首次失败的 `error.log.first` / `trajectory.first.json`）。
- 若需要修资源释放/批处理基础设施：单独 commit，说明是否影响决策与恢复结果，加最小测试；
  **不修改三阶段先验、Scheduler/Expert 决策逻辑、工具参数**。

## 完成标准

- `unfinished_samples.txt` 中 30 个 sample 全部出现在 `per_sample_results.csv` 且状态非空。
- 全 80 张跑完后，重新生成完整版汇总（非 partial）。
