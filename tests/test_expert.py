import json

from expert import Expert


class FakeDepictQA:
    def __init__(self, levels, compare="former"):
        self.levels = list(levels)
        self.compare = compare
        self.calls = []

    def __call__(self, img_path=None, task=None, degradation=None):
        self.calls.append({"task": task, "degradation": degradation})
        if task == "comp_quality":
            return self.compare
        level = self.levels.pop(0) if self.levels else "medium"
        return json.dumps([[degradation, level]])


class FakeGPT4:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, img_path=None, prompt="", format_check=None):
        self.calls.append({"img_path": img_path, "prompt": prompt})
        return self.responses.pop(0)


class FakeTool:
    def __init__(self, name, fail=False):
        self.tool_name = name
        self.fail = fail

    def __call__(self, input_dir=None, output_dir=None, silent=False):
        if self.fail:
            raise RuntimeError("boom")
        (output_dir / "output.png").write_bytes(b"fake")


def _expert(tmp_path, toolbox, levels, compare="former", gpt4_responses=None,
            stop_threshold="low", subtask="deraining", degradation="rain"):
    gpt4 = FakeGPT4(gpt4_responses or [])
    return Expert(
        subtask=subtask,
        degradation=degradation,
        config_path=tmp_path / "nope.yml",
        depictqa=FakeDepictQA(levels, compare=compare),
        gpt4=gpt4,
        toolbox=toolbox,
        stop_threshold=stop_threshold,
    )


def _img(tmp_path, name="input.png"):
    p = tmp_path / name
    p.write_bytes(b"fake")
    return p


def test_skip_when_input_below_threshold(tmp_path):
    tool = FakeTool("xrestormer")
    expert = _expert(tmp_path, [tool], levels=["very low"])
    img = _img(tmp_path)
    out = expert.restore(img, tmp_path / "work")
    assert out == img.resolve()          # 输入已低于阈值，原样返回
    assert expert.gpt4.calls == []       # 未选工具、未执行


def test_compare_all_failed_candidates(tmp_path):
    # M06：两个候选都未达阈值 → 所有有效失败候选两两比较，选后者
    t1 = FakeTool("tool1")
    t2 = FakeTool("tool2")
    expert = _expert(
        tmp_path, [t1, t2],
        levels=["medium", "medium", "medium"],   # 输入 + tool1 输出 + tool2 输出
        compare="latter",
        gpt4_responses=[json.dumps({"thought": "t", "candidates": ["tool1", "tool2"]})],
    )
    img = _img(tmp_path)
    out = expert.restore(img, tmp_path / "work")
    assert "tool-tool2" in str(out)
    comp_calls = [c for c in expert.depictqa.calls if c["task"] == "comp_quality"]
    assert len(comp_calls) == 1          # 两两比较全部失败候选（1 次）


def test_error_recovery_continues(tmp_path):
    # P2：tool1 失败记录错误并继续 tool2，tool2 达阈值即停
    t1 = FakeTool("tool1", fail=True)
    t2 = FakeTool("tool2")
    expert = _expert(
        tmp_path, [t1, t2],
        levels=["medium", "very low"],
        gpt4_responses=[json.dumps({"thought": "t", "candidates": ["tool1", "tool2"]})],
    )
    img = _img(tmp_path)
    out = expert.restore(img, tmp_path / "work")
    assert "tool-tool2" in str(out)


def test_all_tools_failed_returns_none(tmp_path):
    t1 = FakeTool("tool1", fail=True)
    t2 = FakeTool("tool2", fail=True)
    expert = _expert(
        tmp_path, [t1, t2],
        levels=["medium"],
        gpt4_responses=[json.dumps({"thought": "t", "candidates": ["tool1", "tool2"]})],
    )
    img = _img(tmp_path)
    out = expert.restore(img, tmp_path / "work")
    assert out is None                    # 全部失败，明确返回 None


def test_select_tools_dedups_candidates(tmp_path):
    # P2：候选列表去重
    t1 = FakeTool("tool1")
    t2 = FakeTool("tool2")
    t3 = FakeTool("tool3")
    expert = _expert(
        tmp_path, [t1, t2, t3], levels=[],
        gpt4_responses=[json.dumps({"thought": "t", "candidates": ["tool1", "tool1", "tool2"]})],
    )
    cands = expert._select_tools(_img(tmp_path), "medium", instruction="")
    assert cands == ["tool1", "tool2"]


def test_select_tools_passes_instruction_to_prompt(tmp_path):
    # M05：四类信息（图像、严重度、指令、工具表单）都进入候选选择请求
    t1 = FakeTool("tool1")
    t2 = FakeTool("tool2")
    expert = _expert(
        tmp_path, [t1, t2], levels=[],
        gpt4_responses=[json.dumps({"thought": "t", "candidates": ["tool1", "tool2"]})],
    )
    img = _img(tmp_path)
    expert._select_tools(img, "medium", instruction="尽量快")
    call = expert.gpt4.calls[0]
    assert call["img_path"] == img
    assert "尽量快" in call["prompt"]
    assert "medium" in call["prompt"]


def test_single_tool_skips_smart_selection(tmp_path):
    # M07：单工具任务直接返回该工具，不假装智能候选选择
    t1 = FakeTool("xrestormer")
    expert = _expert(tmp_path, [t1], levels=[])
    cands = expert._select_tools(_img(tmp_path), "medium", instruction="")
    assert cands == ["xrestormer"]
    assert expert.gpt4.calls == []
