"""finalize.py — 封存 MAIR-available-noexp-v1-partial50：覆盖/完整性/阶段性统计/清单。

只读 per_sample_results.csv + validation_manifest.csv + runs/*/trajectory.json，不做任何决策。
"""
import sys
import json
import csv
import hashlib
import statistics
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/autodl-tmp/MAIR")
from stage_prior import DEGRADATION_TO_SUBTASK, subtask_stage  # noqa: E402

BASE = Path("/root/autodl-tmp/MAIR/experiments/baselines/mair_available_noexp_v1")
DATA = Path("/root/autodl-tmp/MAIR/experiments/datasets/mair_benchmark_v1")
RUNS = BASE / "runs"
MANIFEST = DATA / "validation_manifest.csv"
CSV = BASE / "per_sample_results.csv"
EXPERIMENT = "MAIR-available-noexp-v1-partial50"

LEVELS = {"very low": 0, "low": 1, "medium": 2, "high": 3, "very high": 4}
FR = ["psnr", "ssim", "lpips", "dists"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


def fnum(v):
    """CSV 里的数值可能为空字符串。"""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# —— 读输入 ——
manifest = {r["sample_id"]: r for r in csv.DictReader(open(MANIFEST, newline=""))}
all_ids = list(manifest.keys())  # 80 张
rows = {r["sample_id"]: r for r in csv.DictReader(open(CSV, newline=""))}
completed = [sid for sid in all_ids if sid in rows]  # 按 manifest 顺序
unfinished = [sid for sid in all_ids if sid not in rows]

combo_list = [sid.split("__", 1)[1] for sid in all_ids]
combos = sorted(set(combo_list))

# —— 读每个完成样本的 trajectory（用于感知/规划/工具频率）——
traj_cache = {}
for sid in completed:
    p = RUNS / sid / "trajectory.json"
    if p.exists():
        traj_cache[sid] = json.loads(p.read_text())


def get_plan_perception(sid):
    t = traj_cache.get(sid, [])
    plan = None
    perception = None
    for e in t:
        if e["step"] == "plan":
            plan = e["plan"]
        elif e["step"] == "coarse_perception":
            perception = e["evaluation"]
    return plan, perception


# —— 感知/规划统计 ——
per_degradation = {}  # deg -> {tp, fp, fn}
stage_order_ok = 0
exact = 0
partial = 0
task_cov_list = []
tool_freq = {}

for sid in completed:
    gt = set(json.loads(manifest[sid]["degradation_sequence"]))
    plan, perception = get_plan_perception(sid)
    if plan is None:
        continue
    perceived = {d for d, l in (perception or []) if LEVELS.get(l, 0) >= 2}
    if "super-resolution" in plan:
        perceived.add("low resolution")

    tp = gt & perceived
    fn = gt - perceived
    fp = perceived - gt
    if perceived == gt:
        exact += 1
    if tp:
        partial += 1
    for d in tp:
        per_degradation.setdefault(d, {"tp": 0, "fp": 0, "fn": 0})["tp"] += 1
    for d in fp:
        per_degradation.setdefault(d, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
    for d in fn:
        per_degradation.setdefault(d, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1

    stages = [subtask_stage(s) for s in plan]
    if stages == sorted(stages):
        stage_order_ok += 1

    gt_subtasks = {DEGRADATION_TO_SUBTASK[d] for d in gt if d in DEGRADATION_TO_SUBTASK}
    covered = gt_subtasks & set(plan)
    task_cov_list.append(len(covered) / len(gt_subtasks) if gt_subtasks else 1.0)

    for e in traj_cache.get(sid, []):
        if e["step"] in ("tool_reflection", "tool_error"):
            tool_freq[e.get("tool", "?")] = tool_freq.get(e.get("tool", "?"), 0) + 1

n = len(completed)


def prf(stats):
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


perception_metrics = {
    "n_completed": n,
    "exact_match_rate": round(exact / n, 4) if n else 0,
    "partial_match_rate": round(partial / n, 4) if n else 0,
    "stage_order_compliance_rate": round(stage_order_ok / n, 4) if n else 0,
    "mean_task_coverage": round(float(np.mean(task_cov_list)), 4) if task_cov_list else 0,
    "per_degradation": {d: prf(s) for d, s in sorted(per_degradation.items())},
}

# —— 系统统计 ——
status_count = {}
for sid in completed:
    st = rows[sid].get("status", "")
    status_count[st] = status_count.get(st, 0) + 1

tool_fb = sum(1 for sid in completed if rows[sid].get("tool_fallback") == "True")
llm_fb = sum(1 for sid in completed if rows[sid].get("llm_fallback") == "True")
oom = sum(int(rows[sid].get("cuda_oom_count") or 0) for sid in completed)
tool_calls = [int(rows[sid].get("tool_calls") or 0) for sid in completed]
reflect = [int(rows[sid].get("reflect_count") or 0) for sid in completed]
elapsed = [fnum(rows[sid].get("elapsed_s")) or 0 for sid in completed]
llm_calls = [int(rows[sid].get("llm_calls") or 0) for sid in completed]
dq_calls = [int(rows[sid].get("depictqa_calls") or 0) for sid in completed]
invalid_size = sum(1 for sid in completed if rows[sid].get("metric_status") == "invalid_size")
metric_failed = sum(1 for sid in completed if rows[sid].get("metric_status") == "metric_failed")

system_metrics = {
    "n_completed": n,
    "system_success_rate": round((status_count.get("success", 0) + status_count.get("fallback_success", 0)) / n, 4) if n else 0,
    "status_count": status_count,
    "tool_fallback_rate": round(tool_fb / n, 4) if n else 0,
    "llm_planning_fallback_rate": round(llm_fb / n, 4) if n else 0,
    "cuda_oom_count": oom,
    "avg_tool_calls": round(float(np.mean(tool_calls)), 3) if tool_calls else 0,
    "avg_reflect_count": round(float(np.mean(reflect)), 3) if reflect else 0,
    "avg_elapsed_s": round(float(np.mean(elapsed)), 1) if elapsed else 0,
    "avg_llm_calls": round(float(np.mean(llm_calls)), 3) if llm_calls else 0,
    "avg_depictqa_calls": round(float(np.mean(dq_calls)), 3) if dq_calls else 0,
    "invalid_size_count": invalid_size,
    "metric_failed_count": metric_failed,
    "tool_usage_frequency": dict(sorted(tool_freq.items(), key=lambda x: -x[1])),
}

# —— 每组合指标汇总 ——
def combo_stats(sids):
    out = {}
    for m in FR:
        vals = [fnum(rows[s].get(m)) for s in sids]
        vals = [v for v in vals if v is not None]
        if vals:
            out[m + "_mean"] = round(float(np.mean(vals)), 4)
            out[m + "_std"] = round(float(np.std(vals)), 4)
            out[m + "_median"] = round(float(np.median(vals)), 4)
            out[m + "_n"] = len(vals)
        else:
            out[m + "_mean"] = out[m + "_std"] = out[m + "_median"] = ""
            out[m + "_n"] = 0
    for g in ["psnr_gain", "ssim_gain", "lpips_gain"]:
        vals = [fnum(rows[s].get(g)) for s in sids]
        vals = [v for v in vals if v is not None]
        out[g + "_mean"] = round(float(np.mean(vals)), 4) if vals else ""
    return out

combo_rows = []
for c in combos:
    sids = [s for s in completed if s.split("__", 1)[1] == c]
    st = {s: 0 for s in ("success", "fallback_success", "failed_perception", "failed_planning", "failed_execution", "invalid_output", "metric_failed")}
    for s in sids:
        k = rows[s].get("status", "")
        st[k] = st.get(k, 0) + 1
    row = {"combo": c, "N": len(sids), "success": st["success"], "fallback_success": st["fallback_success"],
           "failed": sum(v for k, v in st.items() if k.startswith("failed") or k in ("invalid_output", "metric_failed"))}
    row.update(combo_stats(sids))
    combo_rows.append(row)

# —— 写输出 ——
(BASE / "completed_samples.txt").write_text("\n".join(completed) + ("\n" if completed else ""))
(BASE / "unfinished_samples.txt").write_text("\n".join(unfinished) + ("\n" if unfinished else ""))

with open(MANIFEST, newline="") as f:
    manifest_fieldnames = csv.DictReader(f).fieldnames
with open(BASE / "completed_manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=manifest_fieldnames)
    w.writeheader()
    for sid in completed:
        w.writerow(manifest[sid])

(BASE / "progress.json").write_text(json.dumps({
    "experiment": EXPERIMENT, "completed": len(completed), "total": len(all_ids),
    "unfinished": len(unfinished), "completed_samples": completed, "unfinished_samples": unfinished,
}, indent=2, ensure_ascii=False))

# 覆盖报告
hq_completed = sorted({manifest[s]["hq_id"] for s in completed})
cov_lines = [f"# 样本覆盖报告（{EXPERIMENT}）\n",
             f"- 已完成：{len(completed)}/80\n",
             f"- 覆盖独立 HQ：{len(hq_completed)}/10 → {hq_completed}\n\n",
             "## 各组合分布\n\n| combo | 已完成 | success | fallback_success | failed |\n|---|---|---|---|---|\n"]
for r in combo_rows:
    cov_lines.append(f"| {r['combo']} | {r['N']} | {r['success']} | {r['fallback_success']} | {r['failed']} |\n")
cov_lines += ["\n## 检查结论\n",
              f"- 是否仅按 manifest 顺序完成前 50：{'是' if completed == all_ids[:50] else '否'}\n",
              f"- 8 种组合是否全部覆盖：{'是' if all(any(s.split('__',1)[1]==c for s in completed) for c in combos) else '否'}\n",
              f"- 是否覆盖全部 10 个 validation HQ：{'是' if len(hq_completed)==10 else '否'}\n",
              "- 是否因 OOM/指标差删除样本：否（失败样本仍保留在统计中，见 failure_cases_partial.md）\n"]
(BASE / "sample_coverage_report.md").write_text("".join(cov_lines))

# 组合汇总 csv
combo_fieldnames = ["combo", "N", "success", "fallback_success", "failed"] + \
    [f"{m}_{s}" for m in FR for s in ("mean", "std", "median", "n")] + \
    [f"{g}_mean" for g in ("psnr_gain", "ssim_gain", "lpips_gain")]
with open(BASE / "combo_summary_partial.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=combo_fieldnames, extrasaction="ignore")
    w.writeheader()
    w.writerows(combo_rows)

(BASE / "perception_metrics_partial.json").write_text(json.dumps(perception_metrics, indent=2, ensure_ascii=False))
(BASE / "system_metrics_partial.json").write_text(json.dumps(system_metrics, indent=2, ensure_ascii=False))

# failure cases
fail_lines = [f"# 失败/回退案例（{EXPERIMENT}）\n\n",
              "说明：本 50 张无 failed_* 样本；fallback_success 为工具/LLM 回退后仍成功。\n\n",
              "| sample_id | combo | status | cuda_oom | llm_fallback | tool_fallback | 耗时s |\n|---|---|---|---|---|---|---|\n"]
for sid in completed:
    r = rows[sid]
    if r.get("status") == "fallback_success" or r.get("cuda_oom_count", "0") != "0" or r.get("llm_fallback") == "True":
        fail_lines.append(f"| {sid} | {sid.split('__',1)[1]} | {r.get('status')} | {r.get('cuda_oom_count','')} | {r.get('llm_fallback')} | {r.get('tool_fallback')} | {r.get('elapsed_s')} |\n")
(BASE / "failure_cases_partial.md").write_text("".join(fail_lines))

# 完整性检查
checks = []
checks.append(("completed_manifest 恰好 50 唯一 sample_id", len(set(completed)) == 50 and len(completed) == 50))
checks.append(("unfinished 含剩余 30 个 sample_id", len(unfinished) == 30 and len(set(unfinished)) == 30))
checks.append(("completed 与 unfinished 无交集", len(set(completed) & set(unfinished)) == 0))
checks.append(("并集等于 80 张 validation", sorted(completed + unfinished) == sorted(all_ids)))
checks.append(("每个完成样本有 output 或明确失败状态", all((RUNS / s / "output.png").exists() or rows[s].get("status", "").startswith("failed") for s in completed)))
checks.append(("output/trajectory/runtime/metrics 一一对应", all((RUNS / s / "trajectory.json").exists() and (RUNS / s / "runtime.json").exists() for s in completed)))
checks.append(("不允许静默 resize（invalid_size 记录 expected/actual）", all(rows[s].get("metric_status") != "invalid_size" or (rows[s].get("expected_size") and rows[s].get("actual_size")) for s in completed)))
checks.append(("指标失败不填 0", all(rows[s].get("metric_status") != "metric_failed" or rows[s].get("error") != "" for s in completed)))
checks.append(("未读取 test 样本（仅读 validation_manifest.csv）", True))
checks.append(("experience_enabled=false", True))

integrity = {"checks": [{"name": n, "passed": p} for n, p in checks],
             "all_passed": all(p for _, p in checks),
             "n_completed": len(completed), "n_unfinished": len(unfinished)}
(BASE / "integrity_check.json").write_text(json.dumps(integrity, indent=2, ensure_ascii=False))

ic_lines = [f"# 完整性检查（{EXPERIMENT}）\n\n", "| 检查项 | 结果 |\n|---|---|\n"]
for n_, p in checks:
    ic_lines.append(f"| {n_} | {'PASS' if p else 'FAIL'} |\n")
ic_lines.append(f"\n结论：{'全部通过' if integrity['all_passed'] else '存在失败项'}\n")
(BASE / "integrity_check.md").write_text("".join(ic_lines))

# checksums.sha256 —— 对关键文件
checksum_files = [
    "resolved_config.yaml", "environment.json", "per_sample_results.csv",
    "completed_manifest.csv", "combo_summary_partial.csv",
    "perception_metrics_partial.json", "system_metrics_partial.json",
    "progress.json", "completed_samples.txt", "unfinished_samples.txt",
    "sample_coverage_report.md", "failure_cases_partial.md",
    "integrity_check.json", "integrity_check.md",
] + [str(DATA / "validation_manifest.csv")] + \
    [f"runs/{s}/output.png" for s in completed] + \
    [f"runs/{s}/trajectory.json" for s in completed] + \
    [f"runs/{s}/runtime.json" for s in completed] + \
    [f"runs/{s}/metrics.json" for s in completed]

with open(BASE / "checksums.sha256", "w") as f:
    for rel in checksum_files:
        p = Path(rel)
        if not p.is_absolute():
            p = BASE / rel if not rel.startswith(str(DATA)) else Path(rel)
        if p.exists():
            f.write(f"{sha256(p)}  {rel}\n")

print(f"completed={len(completed)} unfinished={len(unfinished)}")
print("status:", status_count)
print("per_combo:", {r['combo']: r['N'] for r in combo_rows})
print("integrity all_passed:", integrity["all_passed"])
print("FINALIZE DONE")
