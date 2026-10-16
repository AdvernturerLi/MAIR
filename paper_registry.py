"""Sec.9 registry content, translated/condensed without inventing deployment.

The two DiffBIR forms on p.20 are not labelled by checkpoint. Only the first
is mapped to the existing general DiffBIR adapter; the second stays unresolved.
"""
SOURCE = "IJCV 2026 134:205, Sec.9 pp.19-22, doi:10.1007/s11263-026-02792-5"


def form(name, functionality, suitable, speed, note=""):
    return dict(name=name, desc=functionality, suitable=suitable, speed=speed,
                note=note, source=SOURCE)


FORMS = {
    "denoising": [
        form("swinir_15", "SwinIR denoising", "Additive white Gaussian noise, sigma=15", "Fast; low computational demand"),
        form("swinir_50", "SwinIR denoising", "Additive white Gaussian noise, sigma=50", "Fast; low computational demand"),
        form("restormer", "Restormer denoising", "Blind denoising; performs well on real-world noise", "Fast; low computational demand"),
        form("xrestormer", "X-Restormer denoising", "Blind denoising, especially synthesized noise patterns", "Fast; low computational demand"),
    ],
    "super-resolution": [
        form("xrestormer", "X-Restormer super-resolution", "Real-time tasks prioritizing speed and consistency; not recommended for high perceptual quality, photography enhancement or detailed restoration", "Very fast; low latency", "Lower visual quality; less detail-intensive tasks"),
        form("diffbir", "DiffBIR super-resolution", "High visual quality and consistency; complex real-world images, severe degradation and noise; detailed restoration. Unsuitable for real-time or low-resource tasks", "Slow; substantial GPU memory", "First DiffBIR form in p.20; excellent visual quality and consistency"),
        form("swinir_psnr", "SwinIR super-resolution", "PSNR and precise reconstruction prioritized, including scientific/medical imaging; not recommended for high perceptual realism", "Very fast; very low computational demand", "May lack rich details and naturalness; avoid when finer, perceptually pleasing detail is required"),
    ],
    "jpeg compression artifact removal": [
        form("fbcnn_90", "FBCNN DeJPEG QF=90", "Very low compression degradation; subtle artifacts in near-lossless JPEG", "Fast; moderate computational demand"),
        form("fbcnn_5", "FBCNN DeJPEG QF=5", "Noticeable JPEG artifacts, described as moderate degradation in the published form (QF=5)", "Fast; moderate computational demand"),
        form("fbcnn_blind", "FBCNN blind DeJPEG", "Unknown or varying compression levels; no specified quality factor", "Fast; moderate computational demand"),
        form("swinir_40", "SwinIR DeJPEG", "Medium to high JPEG degradation; not suited for near-lossless cases or extreme artifacts beyond the training range", "Fast; moderate computational demand"),
    ],
    "brightening": [
        form("gamma_correction", "Gamma Correction low-light enhancement", "Contrast enhancement and wide dynamic range; less effective for very low light", "Very fast; minimal computational demand"),
        form("constant_shift", "Constant Shift low-light enhancement", "Uniform lighting; quick brightness, exposure correction and preprocessing; unsuitable for complex lighting or high contrast", "Very fast; minimal computational demand"),
        form("retinexformer", "Retinexformer low-light enhancement", "Low-light scenes with high darkness levels", "Moderate speed and computational demand", "May cause overexposure and degrade visual quality"),
        form("cotf", "CoTF low-light enhancement", "Real-time exposure correction and dynamic adjustments under complex lighting", "Moderate speed and computational demand", "Often results in poor visual quality"),
    ],
}

UNRESOLVED_DIFFBIR_FORM = form(
    "unresolved_diffbir_second_form", "DiffBIR super-resolution",
    "High visual quality in complex real-world low-resolution, noise and compression cases; prioritize if subsequent denoising or deblurring is planned",
    "Extremely slow; substantial GPU memory",
    "May introduce unnecessary details; avoid precise texture-preservation tasks. The published form does not identify the checkpoint; no adapter mapping is asserted.",
)
