"""MAIR Level 2 专家（Expert）。

每个专家专精一种退化，负责：
  1. 细感知：DepictQA 评估该退化当前 severity。
  2. 候选工具选择：GPT-4o 读工具注册表，选 1~3 个候选工具。
  3. 逐个工具执行 + 反思（DepictQA 评估结果 severity），达 very low 就停。
  4. 全不达标则两两比较选最佳。
"""
from pathlib import Path
from typing import Optional

from executor import executor
from llm import DepictQA, GPT4
from utils.custom_types import Degradation, Level, Subtask
import prompts
import registry


class Expert:
    def __init__(
        self,
        subtask: Subtask,
        degradation: Degradation,
        config_path: Path = Path("config.yml"),
        silent: bool = False,
    ) -> None:
        self.subtask = subtask
        self.degradation = degradation
        self.levels: list[Level] = ["very low", "low", "medium", "high", "very high"]

        self.depictqa = DepictQA(silent=silent)
        self.gpt4 = GPT4(
            config_path=config_path,
            system_message=prompts.system_message,
            silent=silent,
        )
        # 该 subtask 实际注册的工具（executor 单例），过滤到可用工具（同 AgenticIR WORKING_TOOLS）
        self.toolbox = executor.toolbox_router[self.subtask]
        if self.subtask in registry.WORKING_TOOLS:
            names = set(registry.WORKING_TOOLS[self.subtask])
            self.toolbox = [t for t in self.toolbox if t.tool_name in names]
        self.tool_by_name = {t.tool_name: t for t in self.toolbox}

    # —— 细感知 ——
    def _fine_perceive(self, img_path: Path) -> Level:
        res = eval(
            self.depictqa(img_path, task="eval_degradation", degradation=self.degradation)
        )
        return res[0][1]

    # —— 候选工具选择（注册表 + GPT-4o）——
    def _select_tools(self, img_path: Path, level: Level) -> list[str]:
        if len(self.toolbox) <= 1:
            return [t.tool_name for t in self.toolbox]

        available_names = {t.tool_name for t in self.toolbox}
        # 只保留注册表里实际存在的工具
        forms = [f for f in registry.REGISTRY.get(self.subtask, []) if f["name"] in available_names]
        reg_text = "\n".join(
            f"- {f['name']}: {f['desc']}；适用：{f['suitable']}；效率：{f['speed']}"
            + (f"；注意：{f['note']}" if f["note"] else "")
            for f in forms
        )

        def check(sel: object):
            assert isinstance(sel, dict) and set(sel.keys()) == {"thought", "candidates"}
            cands = sel["candidates"]
            assert isinstance(cands, list) and 1 <= len(cands) <= 3
            assert all(c in available_names for c in cands)

        sel = self.gpt4(
            img_path=img_path,
            prompt=prompts.expert_select_tools_prompt.format(
                degradation=self.degradation, severity=level, registry=reg_text
            ),
            format_check=check,
        )
        return eval(sel)["candidates"]

    # —— 执行 + 反思 ——
    def _apply_and_reflect(self, img_path: Path, subtask_dir: Path, candidates: list[str]):
        """逐个执行候选工具，反思 severity。返回 (best_img_path, best_level, tool_name)。"""
        res_by_level: dict[str, list[tuple[Path, str]]] = {}
        last_result = None
        for tool_name in candidates:
            tool = self.tool_by_name[tool_name]
            tool_dir = subtask_dir / f"tool-{tool_name}"
            output_dir = tool_dir / "0-img"
            output_dir.mkdir(parents=True, exist_ok=True)

            # 执行（input_dir = 输入图所在目录，output_dir 必须为空）
            tool(input_dir=img_path.parent, output_dir=output_dir, silent=True)

            outs = sorted(output_dir.glob("*"))
            if not outs:
                continue
            out_path = outs[0]
            last_result = (out_path, tool_name)

            # 反思
            level = self._fine_perceive(out_path)
            res_by_level.setdefault(level, []).append((out_path, tool_name))
            if level == "very low":
                return out_path, "very low", tool_name

        # 没有 very low：从最低可用严重度里两两比较选最佳
        for lvl in self.levels[1:]:  # low → very high
            if lvl in res_by_level:
                cands = res_by_level[lvl]
                best = self._search_best(cands)
                return best[0], lvl, best[1]

        # 一个候选都没有成功执行
        if last_result is not None:
            return last_result[0], "unknown", last_result[1]
        return None, "unknown", None

    def _search_best(self, cands: list[tuple[Path, str]]) -> tuple[Path, str]:
        """两两比较（锦标赛），返回最佳 (img_path, tool_name)。"""
        best_img, best_name = cands[0]
        for img, name in cands[1:]:
            choice = self.depictqa(img_path=[best_img, img], task="comp_quality")
            if choice == "latter":
                best_img, best_name = img, name
        return best_img, best_name

    # —— 对外入口 ——
    def restore(
        self, img_path: Path, work_dir: Path, instruction: str = ""
    ) -> Optional[Path]:
        """恢复当前退化。img_path 为输入图，work_dir 为本次 MAIR 运行的输出根。"""
        work_dir = Path(work_dir).resolve()
        img_path = Path(img_path).resolve()
        level = self._fine_perceive(img_path)
        candidates = self._select_tools(img_path, level)

        subtask_dir = work_dir / f"subtask-{self.subtask}"
        subtask_dir.mkdir(parents=True, exist_ok=True)

        best_img, best_level, best_tool = self._apply_and_reflect(
            img_path, subtask_dir, candidates
        )
        print(f"[Expert] {self.subtask} ({self.degradation}) 输入 level={level} "
              f"候选={candidates} 最终={best_tool} level={best_level}", flush=True)
        return best_img
