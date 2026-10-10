from pathlib import Path

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
