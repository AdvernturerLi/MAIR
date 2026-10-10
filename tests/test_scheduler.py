import json

import numpy as np
import cv2

from scheduler import Scheduler


class FakeDepictQA:
    def __init__(self, evaluation):
        self.evaluation = json.dumps([[d, l] for d, l in evaluation])
        self.calls = []

    def __call__(self, img_path=None, task=None, degradation=None):
        self.calls.append({"task": task, "img_path": img_path})
        return self.evaluation


class FakeGPT4:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, img_path=None, prompt="", format_check=None):
        self.calls.append({"img_path": img_path, "prompt": prompt})
        return self.responses.pop(0)


def _make_img(tmp_path, h=400, w=400):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    p = tmp_path / "input.png"
    cv2.imwrite(str(p), img)
    return p


def _scheduler(tmp_path, evaluation, fake_gpt4=None, experience_path=None, with_retrieval=False):
    return Scheduler(
        config_path=tmp_path / "nonexistent.yml",
        experience_path=experience_path,
        with_retrieval=with_retrieval,
        depictqa=FakeDepictQA(evaluation),
        gpt4=fake_gpt4 if fake_gpt4 is not None else FakeGPT4([]),
    )


def test_plan_groups_three_stages(monkeypatch, tmp_path):
    monkeypatch.setattr("scheduler.available_tools", lambda s: ["t"])
    img = _make_img(tmp_path)
    evaluation = [("jpeg compression artifact", "high"), ("noise", "medium"), ("rain", "medium")]
    s = _scheduler(tmp_path, evaluation, with_retrieval=False)
    plan = s.plan(img, evaluation)
    assert plan == ["jpeg compression artifact removal", "denoising", "deraining"]


def test_plan_instruction_keep_rain(monkeypatch, tmp_path):
    monkeypatch.setattr("scheduler.available_tools", lambda s: ["t"])
    img = _make_img(tmp_path)
    evaluation = [("jpeg compression artifact", "high"), ("rain", "medium")]
    s = _scheduler(tmp_path, evaluation, with_retrieval=False)
    plan = s.plan(img, evaluation, instruction="保留雨")
    assert plan == ["jpeg compression artifact removal"]


def test_plan_only_rain(monkeypatch, tmp_path):
    monkeypatch.setattr("scheduler.available_tools", lambda s: ["t"])
    img = _make_img(tmp_path)
    evaluation = [("rain", "medium"), ("haze", "medium")]
    s = _scheduler(tmp_path, evaluation, with_retrieval=False)
    plan = s.plan(img, evaluation, instruction="只去雨")
    assert plan == ["deraining"]


def test_plan_skips_degradation_without_tools(monkeypatch, tmp_path):
    # noise 无可用工具 → 跳过（M07）
    monkeypatch.setattr("scheduler.available_tools", lambda s: ["t"] if s == "deraining" else [])
    img = _make_img(tmp_path)
    evaluation = [("rain", "medium"), ("noise", "medium")]
    s = _scheduler(tmp_path, evaluation, with_retrieval=False)
    plan = s.plan(img, evaluation)
    assert plan == ["deraining"]


def test_plan_low_res_heuristic(monkeypatch, tmp_path):
    monkeypatch.setattr("scheduler.available_tools", lambda s: ["t"])
    img = _make_img(tmp_path, h=200, w=200)
    s = _scheduler(tmp_path, [], with_retrieval=False)
    plan = s.plan(img, [])
    assert plan == ["super-resolution"]


def test_plan_size_heuristic_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr("scheduler.available_tools", lambda s: ["t"])
    img = _make_img(tmp_path, h=200, w=200)
    cfg = tmp_path / "cfg.yml"
    cfg.write_text("USE_SIZE_HEURISTIC_FOR_SR: false\n")
    s = Scheduler(config_path=cfg, with_retrieval=False, depictqa=FakeDepictQA([]), gpt4=FakeGPT4([]))
    plan = s.plan(img, [])
    assert plan == []


def test_plan_wo_retrieval_multimodal(monkeypatch, tmp_path):
    # M02：阶段内规划的多模态请求收到原图 + 粗感知 + 指令
    monkeypatch.setattr("scheduler.available_tools", lambda s: ["t1", "t2"])
    img = _make_img(tmp_path)
    evaluation = [("noise", "medium"), ("motion blur", "medium")]
    fake_gpt4 = FakeGPT4(["deblur before denoise", json.dumps(["motion deblurring", "denoising"])])
    s = _scheduler(tmp_path, evaluation, fake_gpt4=fake_gpt4, with_retrieval=False)
    plan = s.plan(img, evaluation, instruction="尽量快更清晰")
    assert plan == ["motion deblurring", "denoising"]
    assert len(fake_gpt4.calls) == 2
    for call in fake_gpt4.calls:
        assert call["img_path"] == img
        assert "尽量快更清晰" in call["prompt"]
        assert "noise: medium" in call["prompt"]


def test_plan_w_retrieval_multimodal_with_experience(monkeypatch, tmp_path):
    # M02 + M03 + M04：图像、指令、经验都进入规划请求
    monkeypatch.setattr("scheduler.available_tools", lambda s: ["t1", "t2"])
    img = _make_img(tmp_path)
    evaluation = [("noise", "medium"), ("motion blur", "medium")]
    exp = tmp_path / "exp.json"
    exp.write_text(json.dumps({"source": "offline distillation", "distilled": "deblur first"}))
    fake_gpt4 = FakeGPT4([json.dumps({"thought": "t", "order": ["motion deblurring", "denoising"]})])
    s = _scheduler(tmp_path, evaluation, fake_gpt4=fake_gpt4,
                   experience_path=exp, with_retrieval=True)
    plan = s.plan(img, evaluation, instruction="尽量快")
    assert plan == ["motion deblurring", "denoising"]
    call = fake_gpt4.calls[0]
    assert call["img_path"] == img
    assert "尽量快" in call["prompt"]
    assert "deblur first" in call["prompt"]
    assert "noise: medium" in call["prompt"]


def test_plan_skips_sr_without_tools(monkeypatch, tmp_path):
    # 尺寸触发超分，但超分无可用工具 → 跳过（M07，与普通退化一致）
    monkeypatch.setattr(
        "scheduler.available_tools",
        lambda s: [] if s == "super-resolution" else ["t"],
    )
    img = _make_img(tmp_path, h=200, w=200)
    s = _scheduler(tmp_path, [], with_retrieval=False)
    plan = s.plan(img, [])
    assert plan == []


def test_order_within_stage_falls_back_on_llm_failure(monkeypatch, tmp_path):
    # P2：LLM 阶段内排序失败时回退到严重度降序，不中断整条流程
    monkeypatch.setattr("scheduler.available_tools", lambda s: ["t"])

    class BoomGPT4:
        def __call__(self, *a, **k):
            raise RuntimeError("Too many format-check failures")

    img = _make_img(tmp_path)
    evaluation = [("noise", "medium"), ("motion blur", "high")]
    s = _scheduler(tmp_path, evaluation, fake_gpt4=BoomGPT4(), with_retrieval=False)
    plan = s.plan(img, evaluation)
    assert plan == ["motion deblurring", "denoising"]  # high 先于 medium


def test_plan_dedups_duplicate_tasks(monkeypatch, tmp_path):
    # 计划不能含重复任务（P2）；构造 duplicate 退化输入也应去重
    monkeypatch.setattr("scheduler.available_tools", lambda s: ["t"])
    img = _make_img(tmp_path)
    evaluation = [("rain", "medium"), ("rain", "medium")]
    s = _scheduler(tmp_path, evaluation, with_retrieval=False)
    plan = s.plan(img, evaluation)
    assert plan == ["deraining"]
