import json

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


def test_no_experience_file_switches_to_no_retrieval(tmp_path, monkeypatch):
    monkeypatch.setattr("scheduler.available_tools", lambda s: [])
    s = _make(tmp_path, experience_path=None)
    assert s.with_retrieval is False
    assert s.schedule_experience == ""
    assert s.experience_source is None


def test_missing_experience_file_switches(tmp_path, monkeypatch):
    monkeypatch.setattr("scheduler.available_tools", lambda s: [])
    s = _make(tmp_path, experience_path=tmp_path / "missing.json")
    assert s.with_retrieval is False


def test_empty_distilled_switches(tmp_path, monkeypatch):
    monkeypatch.setattr("scheduler.available_tools", lambda s: [])
    exp = tmp_path / "exp.json"
    exp.write_text(json.dumps({"distilled": ""}))
    s = _make(tmp_path, experience_path=exp)
    assert s.with_retrieval is False


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
