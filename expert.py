"""MAIR Level 2 专家（Expert）。

每个专家专精一种退化，负责：
  1. 细感知：DepictQA 评估该退化当前 severity。
  2. 候选工具选择：GPT-4o 读工具注册表、指令与剩余退化，优先只选一个。
  3. 逐个工具执行 + 反思（DepictQA 评估结果 severity），低于阈值就停。
  4. 未达阈值时由 GPT 比较当前最佳（初始为输入）与候选（Algorithm 3）。
"""
import json
import time
from pathlib import Path
from typing import Optional

import yaml

from executor import executor
from llm import DepictQA, GPT4
from utils.custom_types import Degradation, Level, Subtask
from trajectory import Trajectory
import prompts
import registry


class Expert:
    def __init__(
        self,
        subtask: Subtask,
        degradation: Degradation,
        config_path: Path = Path("config.yml"),
        silent: bool = False,
        depictqa=None,
        gpt4=None,
        toolbox=None,
        stop_threshold: Optional[Level] = None,
        trajectory: Optional[Trajectory] = None,
    ) -> None:
        self.subtask = subtask
        self.degradation = degradation
        self.levels: list[Level] = ["very low", "low", "medium", "high", "very high"]
        self.silent = silent
        self.trajectory = trajectory

        self._cfg: dict = {}
        if Path(config_path).exists():
            self._cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}

        self.depictqa = depictqa if depictqa is not None else DepictQA(
            silent=silent,
            low_res_as_blur=bool(self._cfg.get("LOW_RES_AS_BLUR", True)),
        )
        self.gpt4 = gpt4 if gpt4 is not None else GPT4(
            config_path=config_path, system_message=prompts.system_message, silent=silent
        )

        # 停止阈值：严重度严格低于该值即停（论文示例 "low" → 实际 "very low" 才停）
        if stop_threshold is not None:
            self.stop_threshold = stop_threshold
        else:
            self.stop_threshold = self._cfg.get("STOP_THRESHOLD", "low")
        assert self.stop_threshold in self.levels, f"非法停止阈值: {self.stop_threshold}"

        # 该 subtask 实际可用工具（代码+环境+权重检查，M07）；注入的 toolbox 优先
        if toolbox is not None:
            self.toolbox = list(toolbox)
        else:
            avail = registry.available_tools(self.subtask)
            self.toolbox = [
                t for t in executor.toolbox_router[self.subtask] if t.tool_name in avail
            ]
        self.tool_by_name = {t.tool_name: t for t in self.toolbox}
        if not self._cfg.get("ENABLE_MAXIM", False):
            self.toolbox = [t for t in self.toolbox if t.tool_name != "maxim"]
            self.tool_by_name = {t.tool_name: t for t in self.toolbox}
        if self._cfg.get("REQUIRE_PAPER_RESOURCES", False) or self._cfg.get("USE_PAPER_TOOLSET", False):
            from paper_resources import PAPER_TOOLS
            self.toolbox = [t for t in self.toolbox if t.tool_name in PAPER_TOOLS[self.subtask]]
            self.tool_by_name = {t.tool_name: t for t in self.toolbox}

    def _below_threshold(self, level: Level) -> bool:
        return self.levels.index(level) < self.levels.index(self.stop_threshold)

    def _record(self, step, **fields):
        if self.trajectory is not None:
            self.trajectory.record(step, subtask=self.subtask, **fields)

    # —— 细感知 ——
    def _fine_perceive(self, img_path: Path) -> Level:
        res = json.loads(
            self.depictqa(img_path, task="eval_degradation", degradation=self.degradation)
        )
        if not res:
            # DepictQA 无法评估该退化时退回中性默认，避免无条件取 res[0][1] 崩溃
            self._record("fine_perception_fallback", degradation=self.degradation)
            return "medium"
        return res[0][1]

    # —— 候选工具选择（注册表 + GPT-4o，含指令）——
    def _select_tools(self, img_path: Path, level: Level, instruction: str = "",
                      remaining_degradations=None) -> list[str]:
        if not self.toolbox:
            raise RuntimeError(f"No available tools for {self.subtask}")

        available_names = {t.tool_name for t in self.toolbox}
        forms = [f for f in registry.REGISTRY.get(self.subtask, []) if f["name"] in available_names]
        if {f["name"] for f in forms} != available_names:
            raise ValueError(f"Missing registry forms for {available_names - {f['name'] for f in forms}}")
        reg_text = "\n".join(
            f"- {f['name']}: {f['desc']}；适用：{f['suitable']}；效率：{f['speed']}"
            + (f"；注意：{f['note']}" if f["note"] else "")
            for f in forms
        )

        def check(sel):
            assert isinstance(sel, dict) and set(sel.keys()) == {"thought", "candidates"}
            assert isinstance(sel["thought"], str)
            cands = sel["candidates"]
            assert isinstance(cands, list) and 1 <= len(cands) <= len(available_names)
            assert all(isinstance(c, str) for c in cands)
            assert len(cands) == len(set(cands))  # 候选不重复（P2）
            assert all(c in available_names for c in cands)

        sel = self.gpt4(
            img_path=img_path,
            prompt=prompts.expert_select_tools_prompt.format(
                degradation=self.degradation, severity=level,
                instruction=instruction, registry=reg_text,
                remaining_degradations=list(remaining_degradations or []),
            ),
            format_check=check,
        )
        parsed = json.loads(sel)
        check(parsed)
        self._record("tool_selection", level=level, instruction=instruction,
                     remaining_degradations=list(remaining_degradations or []),
                     thought=parsed["thought"], candidates=parsed["candidates"],
                     registry_sources={f["name"]: f.get("source") for f in forms})
        return parsed["candidates"]

    # —— 执行 + 反思 ——
    def _apply_and_reflect(self, img_path: Path, subtask_dir: Path, candidates: list[str],
                           input_level: Level):
        """逐个执行候选工具，反思 severity。返回 (best_img_path, best_level, tool_name)。"""
        best_img, best_level, best_tool = img_path, input_level, None
        valid_outputs = 0
        for tool_name in candidates:
            tool = self.tool_by_name[tool_name]
            tool_dir = subtask_dir / f"tool-{tool_name}"
            output_dir = tool_dir / "0-img"
            output_dir.mkdir(parents=True, exist_ok=True)

            # 执行失败记录原始错误并继续后续候选（P2）
            start = time.monotonic()
            self._record("tool_execution", tool=tool_name, input=str(img_path),
                         python=str(tool._python_path()) if hasattr(tool, "_python_path") else None)
            try:
                tool(input_dir=img_path.parent, output_dir=output_dir, silent=True)
            except Exception as e:
                self._record("tool_error", tool=tool_name, error=f"{type(e).__name__}: {e}")
                print(f"[Expert] {self.subtask} 工具 {tool_name} 失败："
                      f"{type(e).__name__}: {e}", flush=True)
                continue

            outs = sorted(output_dir.glob("*"))
            if not outs:
                self._record("tool_error", tool=tool_name, error="no output produced")
                continue
            out_path = outs[0]

            level = self._fine_perceive(out_path)
            valid_outputs += 1
            self._record("tool_reflection", tool=tool_name, level=level, output=str(out_path),
                         duration_s=round(time.monotonic() - start, 3))
            if self._below_threshold(level):
                self._record("stop", tool=tool_name, level=level,
                             reason=f"below threshold ({self.stop_threshold})")
                return out_path, level, tool_name

            choice = self._compare(best_img, out_path)
            if choice == "latter":
                best_img, best_level, best_tool = out_path, level, tool_name

        if not valid_outputs:
            self._record("all_tools_failed", candidates=candidates)
            raise RuntimeError(f"All candidate tools failed for {self.subtask}")
        self._record("expert_output", output=str(best_img), tool=best_tool,
                     retained_input=best_tool is None, level=best_level)
        return best_img, best_level, best_tool

    def _compare(self, former: Path, latter: Path) -> str:
        def check(obj):
            assert isinstance(obj, dict) and set(obj) == {"thought", "choice"}
            assert isinstance(obj["thought"], str)
            assert obj["choice"] in ("former", "latter", "neither")

        result = json.loads(self.gpt4(
            img_path=[former, latter], prompt=prompts.compare_quality_prompt,
            format_check=check,
        ))
        check(result)
        self._record("quality_comparison", former=str(former), latter=str(latter),
                     choice=result["choice"], thought=result["thought"],
                     comparator=getattr(self.gpt4, "model", "multimodal_llm"))
        return result["choice"]

    # —— 对外入口 ——
    def restore(
        self, img_path: Path, work_dir: Path, instruction: str = "",
        remaining_degradations=None,
    ) -> Optional[Path]:
        """恢复当前退化。img_path 为输入图，work_dir 为本次 MAIR 运行的输出根。"""
        work_dir = Path(work_dir).resolve()
        img_path = Path(img_path).resolve()
        level = self._fine_perceive(img_path)
        self._record("fine_perception", level=level, input=str(img_path))

        # Algorithm 3 lines 3-6: every expert obeys the same input threshold.
        # LR perception remains an explicitly documented implementation assumption.
        if self._below_threshold(level):
            self._record("skip_below_threshold", level=level, threshold=self.stop_threshold)
            print(f"[Expert] {self.subtask} 输入 level={level} 已低于阈值 "
                  f"{self.stop_threshold}，跳过", flush=True)
            return img_path

        candidates = self._select_tools(img_path, level, instruction, remaining_degradations)
        self._record("candidate_tools", candidates=candidates)

        subtask_dir = work_dir / f"subtask-{self.subtask}"
        subtask_dir.mkdir(parents=True, exist_ok=True)

        best_img, best_level, best_tool = self._apply_and_reflect(
            img_path, subtask_dir, candidates, level
        )
        print(f"[Expert] {self.subtask} ({self.degradation}) 输入 level={level} "
              f"候选={candidates} 最终={best_tool} level={best_level}", flush=True)
        return best_img
