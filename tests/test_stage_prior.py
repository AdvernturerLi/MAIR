from stage_prior import (
    subtask_stage,
    degradation_stage,
    DEGRADATION_TO_SUBTASK,
    STAGE_NAMES,
)


def test_compression_stage_first():
    assert subtask_stage("jpeg compression artifact removal") == 1
    assert STAGE_NAMES[1] == "compression"


def test_imaging_stage_middle():
    for s in ["denoising", "motion deblurring", "defocus deblurring", "super-resolution"]:
        assert subtask_stage(s) == 2


def test_scene_stage_last():
    for s in ["deraining", "dehazing", "brightening"]:
        assert subtask_stage(s) == 3


def test_restoration_reverses_degradation():
    # 退化发生顺序 scene→imaging→compression；修复顺序应相反
    stages = [degradation_stage(d) for d in ["jpeg compression artifact", "noise", "rain"]]
    assert stages == [1, 2, 3]


def test_unknown_defaults_to_imaging():
    assert subtask_stage("unknown task") == 2
    assert degradation_stage("unknown degradation") == 2


def test_degradation_to_subtask_mapping():
    assert DEGRADATION_TO_SUBTASK["rain"] == "deraining"
    assert DEGRADATION_TO_SUBTASK["dark"] == "brightening"
    assert DEGRADATION_TO_SUBTASK["low resolution"] == "super-resolution"
    assert DEGRADATION_TO_SUBTASK["jpeg compression artifact"] == "jpeg compression artifact removal"
