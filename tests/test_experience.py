import json
import pytest

from scheduler import Scheduler


class FakeDepictQA:
    def __call__(self, *a, **k):
        return "[]"


class FakeGPT4:
    def __call__(self, *a, **k):
        return '{"thought":"t","order":[]}'


def _make(tmp_path, experience_path, with_retrieval=True):
    return Scheduler(
        config_path=tmp_path / "nope.yml",
        experience_path=experience_path,
        with_retrieval=with_retrieval,
        depictqa=FakeDepictQA(),
        gpt4=FakeGPT4(),
    )


def test_default_loads_published_experience(tmp_path, monkeypatch):
    monkeypatch.setattr("scheduler.available_tools", lambda s: [])
    s = _make(tmp_path, experience_path=None)
    assert s.with_retrieval is True
    assert "followed by super-resolution and then denoising" in s.schedule_experience
    assert "Section 8" in s.experience_source
    assert len(s.experience_hash) == 64


def test_missing_experience_file_switches(tmp_path, monkeypatch):
    monkeypatch.setattr("scheduler.available_tools", lambda s: [])
    with pytest.raises(ValueError):
        _make(tmp_path, experience_path=tmp_path / "missing.json")


def test_empty_distilled_switches(tmp_path, monkeypatch):
    monkeypatch.setattr("scheduler.available_tools", lambda s: [])
    exp = tmp_path / "exp.json"
    exp.write_text(json.dumps({"distilled": ""}))
    with pytest.raises(ValueError):
        _make(tmp_path, experience_path=exp)


def test_valid_experience_loaded_with_source(tmp_path, monkeypatch):
    monkeypatch.setattr("scheduler.available_tools", lambda s: [])
    exp = tmp_path / "exp.json"
    exp.write_text(json.dumps({"source": "offline distillation", "distilled": "deblur first"}))
    s = _make(tmp_path, experience_path=exp)
    assert s.with_retrieval is True
    assert s.schedule_experience == "deblur first"
    assert s.experience_source == "offline distillation"


def test_explicit_no_retrieval_stays_off(tmp_path, monkeypatch):
    monkeypatch.setattr("scheduler.available_tools", lambda s: [])
    exp = tmp_path / "exp.json"
    exp.write_text(json.dumps({"source": "x", "distilled": "y"}))
    s = _make(tmp_path, experience_path=exp, with_retrieval=False)
    assert s.with_retrieval is False
