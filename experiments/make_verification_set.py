"""生成功能验证集：12 张已知退化的 LQ 图（单一/跨阶段/同阶段/三退化）。"""
import sys
import json
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, "/root/autodl-tmp/AgenticIR/dataset")
from add_single_degradation import (
    lr, add_noise, add_jpeg_comp_artifacts, add_haze,
    add_motion_blur, add_rain,
)

np.random.seed(42)  # 复现用固定种子

GT_DIR = Path("/root/autodl-tmp/MiO_test/original/GT")
DEPTH_DIR = Path("/root/autodl-tmp/MiO_test/original/depth/LR_tensor")
OUT_DIR = Path("/root/autodl-tmp/MAIR/experiments/functional_verification")
LQ_DIR = OUT_DIR / "lq"
LQ_DIR.mkdir(parents=True, exist_ok=True)

HQ_ID = "001"

# 退化应用顺序 = 真实世界退化发生顺序（scene→imaging→compression）
FUNC = {
    "low resolution": lambda im: lr(im),
    "noise": lambda im: add_noise(im),
    "jpeg compression artifact": lambda im: add_jpeg_comp_artifacts(im),
    "rain": lambda im: add_rain(im),
    "haze": lambda im: add_haze(im, idx=HQ_ID, depth_dir=DEPTH_DIR),
    "motion blur": lambda im: add_motion_blur(im),
}

CASES = {
    "s1_jpeg": ["jpeg compression artifact"],
    "s2_noise": ["noise"],
    "s3_rain": ["rain"],
    "s4_lowres": ["low resolution"],
    "c1_noise_jpeg": ["noise", "jpeg compression artifact"],
    "c2_rain_jpeg": ["rain", "jpeg compression artifact"],
    "c3_haze_lowres": ["haze", "low resolution"],
    "c4_rain_noise": ["rain", "noise"],
    "t1_noise_motionblur": ["motion blur", "noise"],
    "t2_rain_haze": ["haze", "rain"],
    "x1_rain_noise_jpeg": ["rain", "noise", "jpeg compression artifact"],
    "x2_motionblur_noise_lowres": ["motion blur", "noise", "low resolution"],
}

img = cv2.imread(str(GT_DIR / f"{HQ_ID}.png"))
assert img is not None, f"GT {HQ_ID} 不存在"
print(f"HQ {HQ_ID} shape: {img.shape}")

manifest = {}
for sample_id, degradations in CASES.items():
    cur = img.copy()
    for d in degradations:
        cur = FUNC[d](cur)
    out_path = LQ_DIR / f"{sample_id}.png"
    cv2.imwrite(str(out_path), cur)
    manifest[sample_id] = {"hq": f"{HQ_ID}.png", "degradations": degradations,
                           "size": [int(x) for x in cur.shape[:2]]}
    print(f"{sample_id:24s} {degradations} -> {cur.shape[:2]}")

(OUT_DIR / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False))
print("DONE")
