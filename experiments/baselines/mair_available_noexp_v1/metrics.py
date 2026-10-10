"""指标计算：PSNR/SSIM/LPIPS/DISTS（+ 相对输入的增益）。

规则（对齐任务书）：
- RGB、固定像素范围 [0,255]；
- 输出必须与 GT 同尺寸，**禁止静默 resize**——尺寸不一致时返回 metric_status=invalid_size，
  并记录 expected_size / actual_size；
- 指标错误不填 0，记录 failed 与原因。
"""
from pathlib import Path

import numpy as np
import cv2
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import piq

FR_METRICS = ["psnr", "ssim", "lpips", "dists"]
# 不可用指标（服务器未装 pyiqa，piq 0.8 无这些），如实标注
UNAVAILABLE_NR = ["niqe", "maniqa", "clip_iqa", "musiq"]


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"无法读取图像: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class Metrics:
    """加载一次 LPIPS + DISTS 模型，逐图计算。"""

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.lpips_fn = lpips.LPIPS(net="alex").to(self.device)
        self.dists_fn = piq.DISTS().to(self.device)

    def _to_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        return t.to(self.device)

    def compute(self, img_path: Path, ref_path: Path, resize_to_ref: bool = False):
        """返回 (metrics: dict | None, status: str, detail: dict)。

        resize_to_ref=True 时，若 img 与 ref 尺寸不同，显式双三次上采样到 ref 尺寸，
        用于低分辨率组合的"输入参考"指标（会在 metrics 里带 upscaled 标记）。
        """
        try:
            img = _read_rgb(img_path)
            ref = _read_rgb(ref_path)
        except Exception as e:
            return None, "metric_failed", {"error": f"{type(e).__name__}: {e}"}

        upscaled = False
        if img.shape != ref.shape:
            if resize_to_ref:
                img = cv2.resize(img, (ref.shape[1], ref.shape[0]),
                                 interpolation=cv2.INTER_CUBIC)
                upscaled = True
            else:
                # 禁止静默 resize：尺寸不一致时标记 invalid_size
                return None, "invalid_size", {
                    "expected_size": list(ref.shape), "actual_size": list(img.shape)
                }

        try:
            psnr = float(peak_signal_noise_ratio(ref, img, data_range=255))
            ssim = float(structural_similarity(ref, img, channel_axis=2, data_range=255))
            ti = self._to_tensor(img)
            tr = self._to_tensor(ref)
            lpips_v = float(self.lpips_fn(ti, tr).item())
            dists_v = float(self.dists_fn(ti, tr).item())
        except Exception as e:
            return None, "metric_failed", {"error": f"{type(e).__name__}: {e}"}

        metrics = {"psnr": psnr, "ssim": ssim, "lpips": lpips_v, "dists": dists_v}
        if upscaled:
            metrics["upscaled"] = True
        return metrics, "ok", {}


def compute_gains(input_metrics, restored_metrics):
    """PSNR/SSIM 增益 = 恢复 - 输入；LPIPS 增益 = 输入 - 恢复（越低越好）。"""
    gains = {}
    if input_metrics and restored_metrics:
        gains["psnr_gain"] = restored_metrics["psnr"] - input_metrics["psnr"]
        gains["ssim_gain"] = restored_metrics["ssim"] - input_metrics["ssim"]
        gains["lpips_gain"] = input_metrics["lpips"] - restored_metrics["lpips"]
    return gains
