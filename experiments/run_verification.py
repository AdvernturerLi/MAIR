"""功能验证集批跑：12 张图 → 轨迹 + 汇总 summary.json。"""
import sys
import json
import shutil
import time
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/MAIR")  # 让脚本从任意目录运行时都能 import scheduler

from scheduler import Scheduler
from expert import Expert
from trajectory import Trajectory

ROOT = Path("/root/autodl-tmp/MAIR/experiments/functional_verification")
LQ_DIR = ROOT / "lq"
RUNS_DIR = ROOT / "runs"
manifest = json.loads((ROOT / "manifest.json").read_text())


def extract_experts(entries):
    """从轨迹提取每个 subtask 的候选/停止/比较/错误。"""
    by_subtask = {}
    for e in entries:
        s = e.get("subtask")
        if not s:
            continue
        st = by_subtask.setdefault(s, {})
        if e["step"] == "candidate_tools":
            st["candidates"] = e["candidates"]
        elif e["step"] == "stop":
            st["stop"] = {"tool": e["tool"], "level": e["level"], "reason": e["reason"]}
        elif e["step"] == "compare_failed_candidates":
            st["compare"] = {"candidates": e["candidates"], "best": e["best"]}
        elif e["step"] == "all_tools_failed":
            st["all_failed"] = True
        elif e["step"] == "skip_below_threshold":
            st["skip_below"] = e["level"]
        elif e["step"] == "tool_error":
            st.setdefault("tool_errors", []).append({"tool": e["tool"], "error": e["error"]})
    return by_subtask


summary = {}
for sample_id, meta in manifest.items():
    lq = LQ_DIR / f"{sample_id}.png"
    out_dir = RUNS_DIR / sample_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    traj = Trajectory(out_dir, input_path=lq)
    t0 = time.time()
    rec = {"sample_id": sample_id, "gt": meta["degradations"]}
    try:
        scheduler = Scheduler(config_path=Path("config.yml"), with_retrieval=False,
                              silent=True, trajectory=traj)
        evaluation = scheduler.perceive(lq)
        plan = scheduler.plan(lq, evaluation)
        rec["perception"] = evaluation
        rec["plan"] = plan

        clean = out_dir / "0-img"
        clean.mkdir()
        cur = clean / "input.png"
        shutil.copy(lq, cur)

        ok = True
        for subtask in plan:
            degradation = scheduler.subtask_degra_dict[subtask]
            expert = Expert(subtask, degradation, config_path=Path("config.yml"),
                            silent=True, trajectory=traj)
            out = expert.restore(cur, out_dir)
            if out is None:
                ok = False
                break
            cur = out
        rec["ok"] = ok
        if ok:
            shutil.copy(cur, out_dir / "result.png")
            traj.record("final", result=str(out_dir / "result.png"))
        rec["experts"] = extract_experts(traj.entries)
        traj.save()
    except Exception as e:
        traj.record("error", error=f"{type(e).__name__}: {e}")
        traj.save()
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["ok"] = False
    rec["elapsed_s"] = round(time.time() - t0, 1)
    summary[sample_id] = rec
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[{sample_id}] ok={rec.get('ok')} plan={rec.get('plan')} {rec.get('elapsed_s')}s", flush=True)

print("ALL DONE", flush=True)
