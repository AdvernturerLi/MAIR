"""MAIR Level 1 调度器（Scheduler）。

职责：粗感知退化 + 按三阶段先验规划。
  - perceive：用 DepictQA 评估全部 7 类退化的严重度（粗感知）。
  - plan：过滤 medium+ → 按三阶段（compression→imaging→scene）分组 →
          阶段内仍由 GPT-4o(+经验) 排序 → 拼接成执行序列。
"""
import json
from pathlib import Path
from typing import Optional

import cv2

from llm import DepictQA, GPT4
from registry import WORKING_DEGRADATIONS
from stage_prior import subtask_stage
from utils.custom_types import Degradation, Level, Subtask
import prompts


class Scheduler:
    def __init__(
        self,
        config_path: Path = Path("config.yml"),
        experience_path: Optional[Path] = None,
        with_retrieval: bool = True,
        silent: bool = False,
    ) -> None:
        self.degra_subtask_dict: dict[Degradation, Subtask] = {
            "low resolution": "super-resolution",
            "noise": "denoising",
            "motion blur": "motion deblurring",
            "defocus blur": "defocus deblurring",
            "haze": "dehazing",
            "rain": "deraining",
            "dark": "brightening",
            "jpeg compression artifact": "jpeg compression artifact removal",
        }
        self.subtask_degra_dict = {v: k for k, v in self.degra_subtask_dict.items()}
        self.levels: list[Level] = ["very low", "low", "medium", "high", "very high"]
        self.with_retrieval = with_retrieval

        self.depictqa = DepictQA(silent=silent)
        self.gpt4 = GPT4(
            config_path=config_path,
            system_message=prompts.system_message,
            silent=silent,
        )

        if experience_path is not None and Path(experience_path).exists():
            self.schedule_experience = json.load(Path(experience_path).open())["distilled"]
        else:
            self.schedule_experience = ""

    # —— 粗感知 ——
    def perceive(self, lq_path: Path) -> list[tuple[Degradation, Level]]:
        """评估全部 7 类退化（不含 low resolution，其由图像尺寸判断）。"""
        evaluation = eval(self.depictqa(lq_path, task="eval_degradation"))
        return evaluation

    # —— 三阶段约束规划 ——
    def plan(self, lq_path: Path, evaluation: list[tuple[Degradation, Level]]) -> list[Subtask]:
        """过滤 medium+ → 按三阶段分组 → 阶段内排序 → 拼接。"""
        agenda: list[Subtask] = []

        # low resolution 由尺寸判断（同 AgenticIR 启发式）
        img_shape = cv2.imread(str(lq_path)).shape[:2]
        if max(img_shape) < 300:
            agenda.append("super-resolution")

        for degradation, severity in evaluation:
            if degradation not in WORKING_DEGRADATIONS:
                continue  # 对齐 AgenticIR，跳过 defocus blur / dark
            if self.levels.index(severity) >= 2:  # medium 及以上
                agenda.append(self.degra_subtask_dict[degradation])

        # 按三阶段分组（compression → imaging → scene）
        grouped: dict[int, list[Subtask]] = {1: [], 2: [], 3: []}
        for subtask in agenda:
            grouped[subtask_stage(subtask)].append(subtask)

        plan: list[Subtask] = []
        for stage in (1, 2, 3):
            stage_agenda = grouped[stage]
            if len(stage_agenda) <= 1:
                plan.extend(stage_agenda)
            else:
                plan.extend(self._order_within_stage(stage_agenda))
        return plan

    # —— 阶段内排序（复用 AgenticIR 的 LLM(+经验) 调度逻辑）——
    def _order_within_stage(self, agenda: list[Subtask]) -> list[Subtask]:
        degradations = [self.subtask_degra_dict[s] for s in agenda]
        if self.with_retrieval:
            return self._schedule_w_retrieval(degradations, agenda)
        return self._schedule_wo_retrieval(degradations, agenda)

    def _schedule_w_retrieval(self, degradations: list[Degradation], agenda: list[Subtask]) -> list[Subtask]:
        def check_order(schedule: object):
            assert isinstance(schedule, dict) and set(schedule.keys()) == {"thought", "order"}
            assert set(schedule["order"]) == set(agenda)

        schedule = self.gpt4(
            prompt=prompts.schedule_w_retrieval_prompt.format(
                degradations=degradations, agenda=agenda,
                experience=self.schedule_experience,
            ),
            format_check=check_order,
        )
        schedule = eval(schedule)
        return schedule["order"]

    def _schedule_wo_retrieval(self, degradations: list[Degradation], agenda: list[Subtask]) -> list[Subtask]:
        insights = self.gpt4(
            prompt=prompts.reason_to_schedule_prompt.format(
                degradations=degradations, agenda=agenda
            )
        )

        def check_order(order: object):
            assert isinstance(order, list) and set(order) == set(agenda)

        order = self.gpt4(
            prompt=prompts.schedule_wo_retrieval_prompt.format(
                degradations=degradations, agenda=agenda, insights=insights
            ),
            format_check=check_order,
        )
        return eval(order)
