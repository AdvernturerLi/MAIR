"""生成 mair_benchmark_v1 无泄漏实验数据集。

按 HQ 原图先做 validation/test 划分（同一 HQ 的所有退化图只进同一 split），
再对每个 HQ 生成 8 类退化组合。所有退化参数显式抽取并记录，配 per-sample
随机种子保证可复现。
"""
import sys
import json
import csv
import hashlib
from pathlib import Path

import numpy as np
import cv2
import yaml

sys.path.insert(0, "/root/autodl-tmp/AgenticIR/dataset")
from add_single_degradation import (  # noqa: E402
    lr, darken, add_noise, add_jpeg_comp_artifacts,
    add_haze, add_motion_blur, add_defocus_blur, add_rain,
)

BASE = Path("/root/autodl-tmp/MAIR/experiments/datasets/mair_benchmark_v1")
GT_DIR = Path("/root/autodl-tmp/MiO_test/original/GT")
DEPTH_DIR = Path("/root/autodl-tmp/MiO_test/original/depth/LR_tensor")

SPLIT_SEED = 42
MASTER_SEED = 20261010
N_VAL = 10
N_TEST = 20

# 8 组合（用户命名 → 规范退化顺序，遵循 scene→imaging→compression）
# "blur" = motion blur；"low-light" = dark（darken）；"low-resolution" = lr(0.25×)
COMBINATIONS = {
    "noise+blur": ["motion blur", "noise"],
    "rain+haze": ["rain", "haze"],
    "low-light+noise": ["dark", "noise"],
    "jpeg+noise": ["noise", "jpeg compression artifact"],
    "blur+low-resolution": ["motion blur", "low resolution"],
    "haze+low-resolution": ["haze", "low resolution"],
    "rain+blur": ["rain", "motion blur"],
    "rain+noise+low-resolution": ["rain", "noise", "low resolution"],
}

FUNC = {
    "rain": add_rain, "haze": add_haze, "dark": darken,
    "motion blur": add_motion_blur, "defocus blur": add_defocus_blur,
    "noise": add_noise, "low resolution": lr,
    "jpeg compression artifact": add_jpeg_comp_artifacts,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def draw_params(degradation: str):
    """按各退化函数默认分布显式抽取参数，返回 (调用 kwargs, 记录 dict)。"""
    if degradation == "rain":
        value = int(np.random.randint(50, 100))
        return {"value": value}, {"value": value}
    if degradation == "haze":
        A = float(np.random.uniform(0.7, 1.0))
        beta = float(np.random.uniform(0.6, 1.8))
        return {"A": A, "beta": beta}, {"A": round(A, 4), "beta": round(beta, 4)}
    if degradation == "dark":
        t = str(np.random.choice(["constant shift", "gamma correction", "linear mapping"]))
        if t == "constant shift":
            arg = int(np.random.randint(30, 50))
            return {"darken_type": t, "arg": arg}, {"darken_type": t, "shift": arg}
        if t == "gamma correction":
            arg = float(np.random.uniform(0.5, 0.7))
            return {"darken_type": t, "arg": arg}, {"darken_type": t, "gamma": round(arg, 4)}
        arg = int(np.random.randint(100, 150))
        return {"darken_type": t, "arg": arg}, {"darken_type": t, "dst_max": arg}
    if degradation in ("motion blur", "defocus blur"):
        severity = int(np.random.randint(3))
        return {"severity": severity}, {"severity": severity}
    if degradation == "noise":
        t = str(np.random.choice(["Gaussian", "Poisson"]))
        if t == "Gaussian":
            sigma = float(np.random.uniform(20, 50))
            return {"noise_type": t, "arg": sigma}, {"noise_type": t, "sigma": round(sigma, 2)}
        scale = float(np.random.uniform(1, 3))
        return {"noise_type": t, "arg": scale}, {"noise_type": t, "scale": round(scale, 2)}
    if degradation == "low resolution":
        return {}, {}
    if degradation == "jpeg compression artifact":
        qf = int(np.random.randint(10, 30))
        return {"quality_factor": qf}, {"quality_factor": qf}
    raise ValueError(degradation)


def apply_degradation(img, degradation: str, hq_id: str, kwargs: dict):
    if degradation == "haze":
        return add_haze(img, idx=hq_id, depth_dir=DEPTH_DIR, **kwargs)
    return FUNC[degradation](img, **kwargs)


# —— HQ 划分（seeded shuffle，保证无泄漏）——
hq_ids = sorted([p.stem for p in GT_DIR.glob("*.png") if p.stem.isdigit()], key=int)
assert len(hq_ids) >= N_VAL + N_TEST, f"HQ 数量不足：{len(hq_ids)}"
rng = np.random.RandomState(SPLIT_SEED)
perm = rng.permutation(len(hq_ids)).tolist()
val_ids = [hq_ids[i] for i in perm[:N_VAL]]
test_ids = [hq_ids[i] for i in perm[N_VAL:N_VAL + N_TEST]]

# 校验 depth map（haze 合成需要）
for hid in val_ids + test_ids:
    if not (DEPTH_DIR / hid / "predict_depth.mat").exists():
        raise SystemExit(f"缺 depth map: {hid}")

rows = []
hash_rows = []
sample_idx = 0

for split, ids in [("validation", val_ids), ("test", test_ids)]:
    (BASE / split / "hq").mkdir(parents=True, exist_ok=True)
    (BASE / split / "lq").mkdir(parents=True, exist_ok=True)
    for hid in ids:
        hq_dst = BASE / split / "hq" / f"{hid}.png"
        hq_dst.write_bytes((GT_DIR / f"{hid}.png").read_bytes())
        hq_hash = sha256(hq_dst)
        hash_rows.append({"file_path": str(hq_dst.relative_to(BASE)),
                          "sha256": hq_hash, "split": split})
        hq_img = cv2.imread(str(hq_dst))
        assert hq_img is not None, f"无法读取 HQ {hid}"

        for combo_id, seq in COMBINATIONS.items():
            sample_id = f"{hid}__{combo_id}"
            seed = MASTER_SEED + sample_idx
            sample_idx += 1
            np.random.seed(seed)

            img = hq_img.copy()
            seq_params = []
            for deg in seq:
                kwargs, rec = draw_params(deg)
                img = apply_degradation(img, deg, hid, kwargs)
                seq_params.append(rec)

            lq_dst = BASE / split / "lq" / f"{sample_id}.png"
            cv2.imwrite(str(lq_dst), img)
            lq_hash = sha256(lq_dst)
            h, w = img.shape[:2]
            rows.append({
                "sample_id": sample_id, "split": split, "hq_id": hid,
                "hq_path": str(hq_dst.relative_to(BASE)),
                "lq_path": str(lq_dst.relative_to(BASE)),
                "hq_sha256": hq_hash, "lq_sha256": lq_hash,
                "degradation_sequence": json.dumps(seq, ensure_ascii=False),
                "degradation_parameters": json.dumps(seq_params, ensure_ascii=False),
                "random_seed": seed, "width": w, "height": h,
            })
            hash_rows.append({"file_path": str(lq_dst.relative_to(BASE)),
                              "sha256": lq_hash, "split": split})

FIELDS = ["sample_id", "split", "hq_id", "hq_path", "lq_path", "hq_sha256",
          "lq_sha256", "degradation_sequence", "degradation_parameters",
          "random_seed", "width", "height"]


def write_csv(path, rows_):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows_)


write_csv(BASE / "validation_manifest.csv", [r for r in rows if r["split"] == "validation"])
write_csv(BASE / "test_manifest.csv", [r for r in rows if r["split"] == "test"])

with open(BASE / "hash_manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["file_path", "sha256", "split"])
    w.writeheader()
    w.writerows(hash_rows)

config = {
    "dataset": "mair_benchmark_v1",
    "source": {"hq_dir": str(GT_DIR), "depth_dir": str(DEPTH_DIR)},
    "split_seed": SPLIT_SEED,
    "master_seed": MASTER_SEED,
    "n_validation_hq": N_VAL,
    "n_test_hq": N_TEST,
    "validation_hq_ids": val_ids,
    "test_hq_ids": test_ids,
    "naming_note": {"blur": "motion blur", "low-light": "dark", "low-resolution": "lr(0.25x)"},
    "combinations": COMBINATIONS,
    "seed_determined_internal_params": "motion blur 的 angle、rain 的 length/angle/noise 由函数内部 np.random 抽取，受 random_seed 控制（重跑同 seed 可复现）",
}
(BASE / "degradation_config.yaml").write_text(
    yaml.safe_dump(config, allow_unicode=True, sort_keys=False))

print(f"validation: {len(val_ids)} HQ, {len([r for r in rows if r['split']=='validation'])} LQ")
print(f"test: {len(test_ids)} HQ, {len([r for r in rows if r['split']=='test'])} LQ")
print(f"total LQ: {len(rows)}, total files: {len(hash_rows)}")
print("DONE")
