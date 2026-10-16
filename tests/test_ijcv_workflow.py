import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from expert import Expert
from scheduler import Scheduler
from trajectory import Trajectory
from test_expert import FakeTool, _expert, _img
from test_scheduler import FakeDepictQA, _make_img


@pytest.mark.parametrize("choice", ["former", "neither"])
def test_single_failed_candidate_can_retain_input(tmp_path, choice):
    expert = _expert(tmp_path, [FakeTool("xrestormer")],
                     ["high", "medium"], compare=choice)
    image = _img(tmp_path)
    assert expert.restore(image, tmp_path / "run") == image.resolve()
    assert len(expert.gpt4.calls) == 2


def test_later_candidate_applies_to_original_and_compares_with_best(tmp_path):
    calls = []

    class Tool(FakeTool):
        def __call__(self, input_dir=None, **kwargs):
            calls.append(input_dir)
            super().__call__(input_dir=input_dir, **kwargs)

    expert = _expert(tmp_path, [Tool("tool1"), Tool("tool2")],
                     ["high", "medium", "medium"], compare="latter",
                     gpt4_responses=[json.dumps({"thought": "try both", "candidates": ["tool1", "tool2"]})])
    image = _img(tmp_path)
    output = expert.restore(image, tmp_path / "run")
    assert calls == [image.resolve().parent] * 2
    comparisons = [c for c in expert.gpt4.calls if isinstance(c["img_path"], list)]
    assert "tool-tool1" in str(comparisons[1]["img_path"][0])
    assert "tool-tool2" in str(output)


def test_threshold_success_returns_without_comparison(tmp_path):
    expert = _expert(tmp_path, [FakeTool("xrestormer")], ["high", "very low"])
    assert expert.restore(_img(tmp_path), tmp_path / "run") != tmp_path / "input.png"
    assert len(expert.gpt4.calls) == 1


def test_selection_receives_remaining_degradations_and_allows_four(tmp_path):
    names = ["tool1", "tool2", "tool3", "tool4"]
    expert = _expert(tmp_path, [FakeTool(n) for n in names], [],
                     gpt4_responses=[json.dumps({"thought": "all needed", "candidates": names})])
    assert expert._select_tools(_img(tmp_path), "high", "preserve texture", ["rain", "noise"]) == names
    assert "Remaining Degradation(s): ['rain', 'noise']" in expert.gpt4.calls[0]["prompt"]


def test_instruction_can_add_low_severity_task_and_preserves_stage_order(monkeypatch, tmp_path):
    monkeypatch.setattr("scheduler.available_tools", lambda s: ["test"])
    replies = iter([
        {"thought": "remove explicitly requested noise", "updated degradations": ["rain", "noise", "jpeg compression artifact"]},
        {"thought": "compression first", "order": ["jpeg compression artifact removal"]},
        {"thought": "imaging next", "order": ["denoising"]},
        {"thought": "scene last", "order": ["deraining"]},
    ])

    def gpt(**kwargs):
        result = next(replies)
        kwargs["format_check"](result)
        return json.dumps(result)

    trace = Trajectory(tmp_path / "trace")
    scheduler = Scheduler(config_path=tmp_path / "none.yml", depictqa=FakeDepictQA([]),
                          gpt4=gpt, trajectory=trace)
    plan = scheduler.plan(_make_img(tmp_path), [("rain", "high"), ("noise", "low")], "remove noise too")
    assert plan == ["jpeg compression artifact removal", "denoising", "deraining"]
    assert any(e["step"] == "experience_loaded" and len(e["sha256"]) == 64 for e in trace.entries)


@pytest.mark.parametrize("payload", [[], {"distilled": 42, "source": "x"},
                                     {"distilled": "x"}, {"distilled": "x", "source": " "}])
def test_malformed_experience_rejected(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        Scheduler(config_path=tmp_path / "none.yml", experience_path=path,
                  depictqa=object(), gpt4=object())


def test_strict_preflight_cannot_claim_missing_tools(monkeypatch):
    import paper_resources
    monkeypatch.setattr(paper_resources, "inventory", lambda: [
        {"subtask": "brightening", "tool": "retinexformer", "available": False}])
    with pytest.raises(RuntimeError, match="retinexformer"):
        paper_resources.validate_paper_resources(SimpleNamespace(gpt4=SimpleNamespace(model="gpt-4o"), with_retrieval=True))


def test_alternative_model_can_keep_paper_tools_only(monkeypatch, tmp_path):
    monkeypatch.setattr("scheduler.available_tools", lambda s: ["ifan", "drbnet"])
    cfg = tmp_path / "alternative.yml"
    cfg.write_text("USE_PAPER_TOOLSET: true\nREQUIRE_PAPER_RESOURCES: false\n")
    scheduler = Scheduler(config_path=cfg, depictqa=object(), gpt4=object())
    assert scheduler._available_tools("defocus deblurring") == ["drbnet"]


def test_maxim_disabled_until_explicit_runtime_opt_in(monkeypatch, tmp_path):
    monkeypatch.setattr("scheduler.available_tools", lambda s: ["maxim", "restormer"])
    scheduler = Scheduler(config_path=tmp_path / "none.yml", depictqa=object(), gpt4=object())
    assert scheduler._available_tools("deraining") == ["restormer"]
    scheduler._cfg["ENABLE_MAXIM"] = True
    assert scheduler._available_tools("deraining") == ["maxim", "restormer"]


def test_mock_end_to_end_planning_and_experts(monkeypatch, tmp_path):
    """Real image files, scheduler, tools, expert reflection and saved trajectory."""
    import registry
    monkeypatch.setattr("scheduler.available_tools", lambda s: ["xrestormer"])
    image = _make_img(tmp_path)
    trace = Trajectory(tmp_path / "run", image)

    def gpt(img_path=None, prompt="", format_check=None):
        if "these tasks: ['denoising']" in prompt:
            result = {"thought": "imaging", "order": ["denoising"]}
        elif "these tasks: ['deraining']" in prompt:
            result = {"thought": "scene", "order": ["deraining"]}
        else:
            result = {"thought": "appropriate tool", "candidates": ["xrestormer"]}
        format_check(result)
        return json.dumps(result)

    class CopyTool:
        tool_name = "xrestormer"

        def __call__(self, input_dir, output_dir, silent):
            image_path = next(input_dir.glob("*.png"))
            assert cv2.imwrite(str(output_dir / "output.png"), cv2.imread(str(image_path)))

    scheduler = Scheduler(config_path=tmp_path / "none.yml", depictqa=FakeDepictQA([]), gpt4=gpt, trajectory=trace)
    plan = scheduler.plan(image, [("rain", "high"), ("noise", "medium")])
    current = image
    from test_expert import FakeDepictQA as FineQA
    for index, subtask in enumerate(plan):
        expert = Expert(subtask, scheduler.subtask_degra_dict[subtask], config_path=tmp_path / "none.yml",
                        toolbox=[CopyTool()], depictqa=FineQA(["high", "very low"]), gpt4=gpt, trajectory=trace)
        current = expert.restore(current, tmp_path / "run", remaining_degradations=[scheduler.subtask_degra_dict[s] for s in plan[index:]])
    assert np.array_equal(cv2.imread(str(current)), cv2.imread(str(image)))
    saved = json.loads(trace.save().read_text(encoding="utf-8"))
    assert len([e for e in saved if e["step"] == "stop"]) == 2
