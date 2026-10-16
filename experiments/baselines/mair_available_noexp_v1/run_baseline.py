"""MAIR 无经验基线运行器（MAIR-available-noexp-v1）。

用法：
  python run_baseline.py                       # 跑 MAIR（保存 output/trajectory/runtime + 状态）
  python run_baseline.py --metrics-only        # 单独算指标（PSNR/SSIM/LPIPS/DISTS + 增益）
  python run_baseline.py --resume --skip-completed --retry-failed
  python run_baseline.py --limit 8             # 只跑前 N 个（预运行用）
"""
import sys
import os
import json
import csv
import time
import argparse
import hashlib
import shutil
import traceback
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/MAIR")

import yaml
import cv2

from scheduler import Scheduler
from expert import Expert
from trajectory import Trajectory
import registry
from stage_prior import DEGRADATION_TO_SUBTASK
import prompts
from llm import GPT4, DepictQA

BASE = Path("/root/autodl-tmp/MAIR/experiments/baselines/mair_available_noexp_v1")
DATA = Path("/root/autodl-tmp/MAIR/experiments/datasets/mair_benchmark_v1")
RUNS = BASE / "runs"
MANIFEST = DATA / "validation_manifest.csv"
RESULT_CSV = BASE / "per_sample_results.csv"

CSV_FIELDS = [
    "sample_id", "split", "hq_id", "combo", "status",
    "elapsed_s", "llm_calls", "prompt_tokens", "completion_tokens",
    "depictqa_calls", "tool_calls", "reflect_count",
    "tool_fallback", "llm_fallback", "cuda_oom_count",
    "metric_status", "psnr", "ssim", "lpips", "dists",
    "input_psnr", "input_ssim", "input_lpips", "input_dists", "input_upscaled",
    "psnr_gain", "ssim_gain", "lpips_gain",
    "expected_size", "actual_size", "error",
]


class CountingGPT4:
    def __init__(self, gpt4):
        self._g = gpt4
        self.calls = 0

    def __call__(self, *a, **k):
        self.calls += 1
        return self._g(*a, **k)

    @property
    def prompt_tokens(self):
        return self._g.prompt_tokens

    @property
    def completion_tokens(self):
        return self._g.completion_tokens


class CountingDepictQA:
    def __init__(self, dq):
        self._d = dq
        self.calls = 0

    def __call__(self, *a, **k):
        self.calls += 1
        return self._d(*a, **k)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_config(cgpt4, config_path):
    """运行时最终生效配置（脱敏，不含 API key）。"""
    cfg = {}
    if Path(config_path).exists():
        cfg = yaml.safe_load(Path(config_path).read_text()) or {}
    return {
        "experiment": "MAIR-available-noexp-v1",
        "commit": "0fa952a",
        "experience_enabled": False,
        "llm": {
            "provider": getattr(cgpt4._g, "base_url", None),
            "model": getattr(cgpt4._g, "model", None),
            "temperature": cfg.get("TEMPERATURE"),
            "max_tokens": cfg.get("MAX_TOKENS"),
            "max_format_retries": 5,
        },
        "depictqa": {
            "base_model": "vicuna-7b-v1.5",
            "weights": ["degra_eval.pt", "DQ495K_Abstractor.pt", "ViT-L-14.pt"],
            "eval_url": "http://127.0.0.1:5001/evaluate_degradation",
            "compare_url": "http://127.0.0.1:5002/compare_quality",
            "timeout_s": 120,
            "low_res_as_blur": bool(cfg.get("LOW_RES_AS_BLUR", True)),
        },
        "expert": {
            "stop_threshold": cfg.get("STOP_THRESHOLD", "low"),
            "below_threshold_means": "severity strictly below stop_threshold",
        },
        "low_resolution_heuristic": {
            "use_size_heuristic": bool(cfg.get("USE_SIZE_HEURISTIC_FOR_SR", True)),
            "max_side": int(cfg.get("LOW_RES_MAX_SIDE", 300)),
        },
        "tools": {
            subtask: registry.available_tools(subtask)
            for subtask in registry.REGISTRY
        },
        "diffbir_fallback": "expert 捕获工具异常后继续下一候选；全失败则两两比较选最佳",
        "secrets": "API key 已脱敏，不写入任何产物",
    }


def build_environment():
    import torch
    import platform
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": os.popen("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").read().strip(),
        "available_metrics": ["psnr", "ssim", "lpips", "dists"],
        "unavailable_metrics": ["niqe", "maniqa", "clip_iqa", "musiq"],
        "unavailable_reason": "服务器未安装 pyiqa；piq 0.8 无这些指标",
    }


def extract_from_trajectory(traj):
    tool_calls = 0
    reflect_count = 0
    tool_fallback = False
    llm_fallback = False
    cuda_oom = 0
    for e in traj.entries:
        if e["step"] == "tool_reflection":
            tool_calls += 1
            reflect_count += 1
        elif e["step"] == "tool_error":
            tool_calls += 1
            tool_fallback = True
            if "OutOfMemory" in str(e.get("error", "")):
                cuda_oom += 1
        elif e["step"] == "experience" and "阶段内排序失败" in str(e.get("message", "")):
            llm_fallback = True
    return {
        "tool_calls": tool_calls, "reflect_count": reflect_count,
        "tool_fallback": tool_fallback, "llm_fallback": llm_fallback,
        "cuda_oom_count": cuda_oom,
    }


def run_sample(row, cgpt4, cdq, config_path):
    sample_id = row["sample_id"]
    lq_path = DATA / row["lq_path"]
    hq_path = DATA / row["hq_path"]
    out_dir = RUNS / sample_id
    # 重跑时：保留首次失败记录（error.log/trajectory.json → *.first），再清空输出目录，
    # 避免工具 precheck 报"输出目录应为空"
    first_error = first_traj = None
    if out_dir.exists():
        if (out_dir / "error.log").exists():
            first_error = (out_dir / "error.log").read_text()
        if (out_dir / "trajectory.json").exists():
            first_traj = (out_dir / "trajectory.json").read_text()
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    if first_error is not None:
        (out_dir / "error.log.first").write_text(first_error)
    if first_traj is not None:
        (out_dir / "trajectory.first.json").write_text(first_traj)

    rec = {f: "" for f in CSV_FIELDS}
    rec["sample_id"] = sample_id
    rec["split"] = row["split"]
    rec["hq_id"] = row["hq_id"]
    rec["combo"] = sample_id.split("__", 1)[1]

    traj = Trajectory(out_dir, input_path=lq_path)
    t0 = time.time()
    start_pt = cgpt4.prompt_tokens
    start_ct = cgpt4.completion_tokens
    phase = "perceive"
    try:
        scheduler = Scheduler(config_path=config_path, with_retrieval=False,
                              silent=True, gpt4=cgpt4, depictqa=cdq, trajectory=traj)
        evaluation = scheduler.perceive(lq_path)
        phase = "plan"
        plan = scheduler.plan(lq_path, evaluation)

        clean = out_dir / "0-img"
        clean.mkdir(exist_ok=True)
        cur = clean / "input.png"
        shutil.copy(lq_path, cur)

        phase = "execute"
        ok = True
        for subtask in plan:
            degradation = scheduler.subtask_degra_dict[subtask]
            expert = Expert(subtask, degradation, config_path=config_path,
                            silent=True, gpt4=cgpt4, depictqa=cdq, trajectory=traj)
            out = expert.restore(cur, out_dir)
            if out is None:
                ok = False
                break
            cur = out

        if not ok:
            rec["status"] = "failed_execution"
        else:
            shutil.copy(cur, out_dir / "output.png")
            traj.record("final", result=str(out_dir / "output.png"))

            # 输出尺寸校验（禁止静默 resize）
            out_img = cv2.imread(str(out_dir / "output.png"))
            hq_img = cv2.imread(str(hq_path))
            rec["expected_size"] = json.dumps(list(hq_img.shape[:2]))
            rec["actual_size"] = json.dumps(list(out_img.shape[:2]))
            if out_img.shape[:2] != hq_img.shape[:2]:
                rec["status"] = "invalid_output"
            else:
                rec["status"] = "success"
    except Exception as e:
        rec["status"] = {"perceive": "failed_perception", "plan": "failed_planning",
                         "execute": "failed_execution"}[phase]
        rec["error"] = f"{type(e).__name__}: {e}"
        traj.record("error", error=f"{type(e).__name__}: {e}")
        (out_dir / "error.log").write_text(traceback.format_exc())

    traj.save()

    ext = extract_from_trajectory(traj)
    rec["elapsed_s"] = round(time.time() - t0, 1)
    rec["llm_calls"] = cgpt4.calls
    rec["prompt_tokens"] = cgpt4.prompt_tokens - start_pt
    rec["completion_tokens"] = cgpt4.completion_tokens - start_ct
    rec["depictqa_calls"] = cdq.calls
    rec["tool_calls"] = ext["tool_calls"]
    rec["reflect_count"] = ext["reflect_count"]
    rec["tool_fallback"] = ext["tool_fallback"]
    rec["llm_fallback"] = ext["llm_fallback"]
    rec["cuda_oom_count"] = ext["cuda_oom_count"]

    # 工具回退 → fallback_success（若流程本身成功）
    if rec["status"] == "success" and (ext["tool_fallback"] or ext["llm_fallback"]):
        rec["status"] = "fallback_success"

    (out_dir / "runtime.json").write_text(json.dumps({
        "sample_id": sample_id, "elapsed_s": rec["elapsed_s"],
        "llm_calls": rec["llm_calls"], "prompt_tokens": rec["prompt_tokens"],
        "completion_tokens": rec["completion_tokens"],
        "depictqa_calls": rec["depictqa_calls"], "tool_calls": rec["tool_calls"],
        "reflect_count": rec["reflect_count"],
        "tool_fallback": rec["tool_fallback"], "llm_fallback": rec["llm_fallback"],
        "cuda_oom_count": rec["cuda_oom_count"],
        "input_path": row["lq_path"], "input_hash": row["lq_sha256"],
        "output_path": str(out_dir / "output.png") if (out_dir / "output.png").exists() else None,
    }, indent=2, ensure_ascii=False))

    return rec


def load_results():
    if RESULT_CSV.exists():
        with open(RESULT_CSV, newline="") as f:
            return {r["sample_id"]: r for r in csv.DictReader(f)}
    return {}


def write_results(results):
    with open(RESULT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in results.values():
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-completed", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--metrics-only", action="store_true")
    ap.add_argument("--config", type=Path, default=Path("config.yml"))
    args = ap.parse_args()

    if not args.metrics_only:
        try:
            from paper_resources import ALGORITHM_VERSION
        except ImportError:
            pass  # Historical checkout predates the IJCV algorithm revision.
        else:
            raise SystemExit(
                f"This v1 runner cannot write {ALGORITHM_VERSION} results into the old baseline. "
                "Use a new experiment directory and the revised entry point. "
                "--metrics-only remains available for existing images."
            )

    rows = list(csv.DictReader(open(MANIFEST, newline="")))
    if args.limit:
        rows = rows[:args.limit]

    results = load_results()

    if args.metrics_only:
        _metrics_only(rows, results)
        return

    # —— 生成 resolved_config + environment（跑之前）——
    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    gpt4 = GPT4(config_path=args.config, system_message=prompts.system_message, silent=True)
    cgpt4 = CountingGPT4(gpt4)
    cdq = CountingDepictQA(DepictQA(silent=True, low_res_as_blur=bool(cfg.get("LOW_RES_AS_BLUR", True))))
    (BASE / "resolved_config.yaml").write_text(yaml.safe_dump(resolve_config(cgpt4, args.config), allow_unicode=True))
    (BASE / "environment.json").write_text(json.dumps(build_environment(), indent=2, ensure_ascii=False))

    failed_statuses = {"failed_perception", "failed_planning", "failed_execution",
                       "invalid_output", "metric_failed"}
    for row in rows:
        sid = row["sample_id"]
        prev = results.get(sid, {})
        prev_status = prev.get("status", "")
        done = prev_status in ("success", "fallback_success")
        failed = prev_status in failed_statuses
        if args.retry_failed:
            if not failed:
                continue  # 只重跑失败样本
        elif args.skip_completed or args.resume:
            if done:
                continue  # 跳过已完成
        # 重跑失败样本时保留第一次失败记录（error.log → error.log.first）
        if failed:
            err = RUNS / sid / "error.log"
            if err.exists():
                err.rename(RUNS / sid / "error.log.first")
        # 重置调用计数（每样本独立统计）
        cgpt4.calls = 0
        cdq.calls = 0
        rec = run_sample(row, cgpt4, cdq, args.config)
        results[sid] = rec
        write_results(results)
        print(f"[{sid}] status={rec['status']} tool_calls={rec['tool_calls']} "
              f"llm={rec['llm_calls']} dq={rec['depictqa_calls']} {rec['elapsed_s']}s", flush=True)
    print("RUN DONE")


def _metrics_only(rows, results):
    from metrics import Metrics, compute_gains
    m = Metrics(device="cuda")
    for row in rows:
        sid = row["sample_id"]
        rec = results.get(sid)
        if not rec or rec.get("status") in ("failed_perception", "failed_planning", "failed_execution"):
            continue
        out_png = RUNS / sid / "output.png"
        hq_png = DATA / row["hq_path"]
        lq_png = DATA / row["lq_path"]
        rec["metric_status"] = "not_computed"
        if not out_png.exists():
            continue
        try:
            out_metrics, out_status, detail = m.compute(out_png, hq_png)
            in_metrics, in_status, _ = m.compute(lq_png, hq_png, resize_to_ref=True)
            rec["metric_status"] = out_status
            if out_status == "ok":
                for k, v in out_metrics.items():
                    rec[k] = round(v, 5)
                if in_status == "ok":
                    for k, v in in_metrics.items():
                        if k == "upscaled":
                            rec["input_upscaled"] = bool(v)
                        else:
                            rec[f"input_{k}"] = round(v, 5)
                gains = compute_gains(in_metrics, out_metrics)
                for k, v in gains.items():
                    rec[k] = round(v, 5)
            elif out_status == "invalid_size":
                rec["expected_size"] = json.dumps(detail["expected_size"])
                rec["actual_size"] = json.dumps(detail["actual_size"])
                rec["status"] = "invalid_output"
            else:
                rec["error"] = detail.get("error", "")
                rec["status"] = "metric_failed"
            (RUNS / sid / "metrics.json").write_text(json.dumps({
                "metric_status": rec["metric_status"],
                "restored": {k: rec.get(k) for k in ["psnr", "ssim", "lpips", "dists"] if rec.get(k) != ""},
                "input": {k: rec.get(f"input_{k}") for k in ["psnr", "ssim", "lpips", "dists"] if rec.get(f"input_{k}") != ""},
                "input_upscaled": rec.get("input_upscaled", ""),
                "gains": {k: rec.get(k) for k in ["psnr_gain", "ssim_gain", "lpips_gain"] if rec.get(k) != ""},
                "expected_size": rec.get("expected_size"),
                "actual_size": rec.get("actual_size"),
                "error": rec.get("error", ""),
            }, indent=2, ensure_ascii=False))
        except Exception as e:
            rec["metric_status"] = "metric_failed"
            rec["error"] = f"{type(e).__name__}: {e}"
        results[sid] = rec
        print(f"[{sid}] metric_status={rec['metric_status']} psnr={rec.get('psnr')}", flush=True)
    write_results(results)
    print("METRICS DONE")


if __name__ == "__main__":
    main()
