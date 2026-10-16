"""MAIR IJCV algorithm revision; published experience enabled by default."""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_func

from scheduler import Scheduler
from expert import Expert
from trajectory import Trajectory
from paper_resources import ALGORITHM_VERSION, validate_paper_resources

COMBOS = {
    "rain+haze": ["001", "002"],
    "noise+jpeg compression artifact": ["001", "002"],
    "motion blur+low resolution": ["001", "002"],
    "rain+low resolution": ["001", "002"],
    "haze+noise": ["001", "002"],
}
DATASET_ROOT = Path("/root/autodl-tmp/AgenticIR/dataset")
HQ_DIR = DATASET_ROOT / "HQ"
LQ_ROOT = DATASET_ROOT / "LQ/d2"
OUT_ROOT = Path("output/mair_ijcv2026_batch").resolve()
RESULTS_PATH = Path("output/mair_ijcv2026_batch_results.json")


def compute_psnr_ssim(img_path, ref_path):
    img = cv2.imread(str(img_path))
    ref = cv2.imread(str(ref_path))
    if img is None or ref is None:
        return None, None
    if img.shape[:2] != ref.shape[:2]:
        img = cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)
    mse = np.mean((img.astype(np.float64) - ref.astype(np.float64)) ** 2)
    psnr = float(10 * np.log10(255.0**2 / mse)) if mse > 0 else float("inf")
    ssim = float(ssim_func(img, ref, channel_axis=2, data_range=255))
    return psnr, ssim


def run_mair(lq_path: Path, out_dir: Path):
    """跑 MAIR 三阶段先验，返回 (result_path, plan, evaluation)。"""
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    trajectory = Trajectory(out_dir, input_path=lq_path)
    scheduler = Scheduler(config_path=Path("config.yml"), with_retrieval=True,
                          trajectory=trajectory)
    if scheduler._cfg.get("REQUIRE_PAPER_RESOURCES", False):
        validate_paper_resources(scheduler)
    trajectory.record("config", algorithm_version=ALGORITHM_VERSION,
                      experience_sha256=scheduler.experience_hash,
                      model=getattr(scheduler.gpt4, "model", None))
    evaluation = scheduler.perceive(lq_path)
    plan = scheduler.plan(lq_path, evaluation)

    clean_input_dir = out_dir / "0-img"
    clean_input_dir.mkdir(parents=True, exist_ok=True)
    cur_img = clean_input_dir / "input.png"
    shutil.copy(lq_path, cur_img)

    for index, subtask in enumerate(plan):
        degradation = scheduler.subtask_degra_dict[subtask]
        expert = Expert(subtask, degradation, config_path=Path("config.yml"),
                        trajectory=trajectory)
        out = expert.restore(cur_img, out_dir, remaining_degradations=[
            scheduler.subtask_degra_dict[s] for s in plan[index:]])
        if out is None:
            trajectory.record("abort", subtask=subtask, reason="expert returned None")
            trajectory.save()
            return None, plan, evaluation
        cur_img = out

    final = out_dir / "result.png"
    shutil.copy(cur_img, final)
    trajectory.record("final", result=str(final))
    trajectory.save()
    return final, plan, evaluation


def main():
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    for combo, img_ids in COMBOS.items():
        for img_id in img_ids:
            lq_path = LQ_ROOT / combo / f"{img_id}.png"
            hq_path = HQ_DIR / f"{img_id}.png"
            key = f"{combo}/{img_id}"
            if key in results and results[key].get("psnr") is not None:
                continue  # 已成功完成，跳过
            out_dir = OUT_ROOT / combo / img_id
            try:
                res, plan, evaluation = run_mair(lq_path, out_dir)
                psnr, ssim = compute_psnr_ssim(res, hq_path)
                results[key] = {"psnr": psnr, "ssim": ssim, "plan": plan}
                print(f"[{key}] PSNR={psnr:.3f} SSIM={ssim:.4f} | plan={plan}", flush=True)
            except Exception as e:
                results[key] = {"psnr": None, "ssim": None, "plan": f"FAILED: {type(e).__name__}: {e}"}
                print(f"[{key}] FAILED: {type(e).__name__}: {e}", flush=True)
            RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
