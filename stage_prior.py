"""MAIR 三阶段退化先验。

真实世界退化按三个阶段先后发生：
  D = D_compression ∘ D_imaging ∘ D_scene
即：场景退化（雨/雾/低光）最早、成像退化（噪声/模糊/低分辨率）其次、
存储压缩（JPEG）最后。

修复应逆序：
  R = R_scene ∘ R_imaging ∘ R_compression
即：先修 JPEG → 再修噪声/模糊/超分 → 最后修雨/雾/低光。
"""
from utils.custom_types import Degradation, Subtask

# 子任务 → 阶段（值越小越先修）
SUBTASK_STAGE: dict[Subtask, int] = {
    # Stage 1 · compression（压缩，最后发生，最先修）
    "jpeg compression artifact removal": 1,
    # Stage 2 · imaging（成像：噪声/模糊/低分辨率）
    "denoising": 2,
    "motion deblurring": 2,
    "defocus deblurring": 2,
    "super-resolution": 2,
    # Stage 3 · scene（场景：雨/雾/低光）
    "deraining": 3,
    "dehazing": 3,
    "brightening": 3,
}

# 退化 → 阶段（感知结果直接归类）
DEGRADATION_STAGE: dict[Degradation, int] = {
    "jpeg compression artifact": 1,
    "noise": 2,
    "motion blur": 2,
    "defocus blur": 2,
    "low resolution": 2,
    "rain": 3,
    "haze": 3,
    "dark": 3,
}

STAGE_NAMES: dict[int, str] = {1: "compression", 2: "imaging", 3: "scene"}


def subtask_stage(subtask: Subtask) -> int:
    """子任务所属阶段，未知默认归为 imaging(2)。"""
    return SUBTASK_STAGE.get(subtask, 2)


def degradation_stage(degradation: Degradation) -> int:
    """退化所属阶段，未知默认归为 imaging(2)。"""
    return DEGRADATION_STAGE.get(degradation, 2)
