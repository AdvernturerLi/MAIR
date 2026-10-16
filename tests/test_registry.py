from pathlib import Path
import zipfile
import pytest

import executor.tool as tool_mod
import registry
from executor.tool import Tool


def test_weights_present_specific_files(tmp_path):
    # M07：声明了具体权重文件时，只校验这些文件（区分同 work_dir 的多变体工具）
    t = Tool("swinir_gan", "super_resolution")
    t.work_dir = tmp_path
    (tmp_path / "model_zoo" / "swinir").mkdir(parents=True)
    t.required_weight_files = [Path("model_zoo/swinir/gan.pth")]
    assert t._weights_present() is False                      # gan.pth 不存在
    (tmp_path / "model_zoo" / "swinir" / "gan.pth").write_bytes(b"")
    assert t._weights_present() is True
    # 另一个变体需要不同的文件，即使同目录存在 gan.pth 也判为缺失
    t2 = Tool("swinir_psnr", "super_resolution")
    t2.work_dir = tmp_path
    t2.required_weight_files = [Path("model_zoo/swinir/psnr.pth")]
    assert t2._weights_present() is False


def test_restormer_required_weight_files():
    from executor.multitask_tools import Restormer
    assert Restormer("defocus_deblurring").required_weight_files[0].as_posix() == \
        "Defocus_Deblurring/pretrained_models/single_image_defocus_deblurring.pth"
    assert Restormer("deraining").required_weight_files[0].as_posix() == \
        "Deraining/pretrained_models/deraining.pth"
    assert Restormer("denoising").required_weight_files[0].as_posix() == \
        "Denoising/pretrained_models/real_denoising.pth"
    assert Restormer("motion_deblurring").required_weight_files[0].as_posix() == \
        "Motion_Deblurring/pretrained_models/motion_deblurring.pth"


def test_available_tools_filters_by_availability(monkeypatch):
    # M07：可用工具由运行时可用性检查决定，而非硬编码白名单
    def fake_available(self):
        return self.tool_name in {"xrestormer", "restormer"}

    monkeypatch.setattr(tool_mod.Tool, "available", fake_available)
    avail = registry.available_tools("deraining")
    assert set(avail) == {"xrestormer", "restormer"}


def test_available_tools_empty_when_none_available(monkeypatch):
    monkeypatch.setattr(tool_mod.Tool, "available", lambda self: False)
    assert registry.available_tools("deraining") == []


def test_registry_has_all_declared_subtasks():
    from stage_prior import DEGRADATION_TO_SUBTASK
    for subtask in DEGRADATION_TO_SUBTASK.values():
        assert subtask in registry.REGISTRY, f"注册表缺 subtask: {subtask}"
        assert len(registry.REGISTRY[subtask]) >= 1


def test_dehazeformer_variants_have_distinct_checkpoints():
    from executor.dehazing import DehazeFormer
    for variant, model, folder in [
        ("outdoor_b", "dehazeformer-b", "outdoor"),
        ("indoor_b", "dehazeformer-b", "indoor"),
        ("indoor_l", "dehazeformer-l", "indoor"),
    ]:
        tool = DehazeFormer(variant)
        assert tool.tool_name == f"dehazeformer_{variant}"
        assert tool.required_weight_files == [Path(f"saved_models/{folder}/{model}.pth")]
        tool.input_dir, tool.output_dir = Path("input"), Path("output")
        opts = tool._get_cmd_opts()
        assert opts[opts.index("--model") + 1] == model
        assert opts[opts.index("--save_dir") + 1] == f"saved_models/{folder}"


def test_maxim_rejects_truncated_npz(tmp_path):
    from executor.multitask_tools import MAXIM
    tool = MAXIM("denoising")
    tool.work_dir = tmp_path
    path = tmp_path / tool.required_weight_files[0]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"partial download")
    assert not tool._weights_present()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("example.npy", b"structure only for test")
    assert tool._weights_present()


def test_maxim_failure_cleans_temporary_input(monkeypatch, tmp_path):
    from executor.multitask_tools import MAXIM
    tool = MAXIM("deraining")
    tool.input_dir = tmp_path / "input"
    tool.input_dir.mkdir()
    (tool.input_dir / "input.png").write_bytes(b"image")

    def fail(self):
        self._preprocess()
        raise RuntimeError("subprocess failure")

    monkeypatch.setattr(Tool, "_invoke", fail)
    with pytest.raises(RuntimeError, match="subprocess failure"):
        tool._invoke()
    assert sorted(p.name for p in tool.input_dir.iterdir()) == ["input.png"]


def test_maxim_uses_separate_interpreter_and_gpu_policy(monkeypatch, tmp_path):
    from executor.multitask_tools import MAXIM
    python = tmp_path / "isolated environment" / "python"
    monkeypatch.setenv("MAIR_MAXIM_PYTHON", str(python))
    tool = MAXIM("deraining")
    tool.input_dir, tool.output_dir = tmp_path / "in", tmp_path / "out"
    assert tool._python_path() == python
    command = tool._get_cmd()
    assert command.startswith("XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda ")
    assert "XLA_FLAGS=--xla_gpu_autotune_level=0" in command
    assert str(python) in command
