"""MAIR Level 1 调度器（Scheduler）。

职责：粗感知退化 + 结合三阶段先验与用户指令规划。
  - perceive：用 DepictQA 评估 7 类退化的严重度（粗感知）。
  - plan：按指令过滤 → 按三阶段（compression→imaging→scene）分组 →
          阶段内由 GPT-4o(+经验) 排序（多模态：原图+粗感知+指令）→ 拼接。
"""
import json
from pathlib import Path
from typing import Optional

import cv2
import yaml

from llm import DepictQA, GPT4
from registry import available_tools
from stage_prior import DEGRADATION_TO_SUBTASK, subtask_stage
from utils.custom_types import Degradation, Level, Subtask
from instruction import InstructionSpec
from trajectory import Trajectory
import prompts


class Scheduler:
    def __init__(
        self,
        config_path: Path = Path("config.yml"),
        experience_path: Optional[Path] = None,
        with_retrieval: bool = True,
        silent: bool = False,
        depictqa=None,
        gpt4=None,
        trajectory: Optional[Trajectory] = None,
    ) -> None:
        self.degra_subtask_dict = DEGRADATION_TO_SUBTASK
        self.subtask_degra_dict = {v: k for k, v in self.degra_subtask_dict.items()}
        self.levels: list[Level] = ["very low", "low", "medium", "high", "very high"]
        self.silent = silent
        self.trajectory = trajectory

        self.depictqa = depictqa if depictqa is not None else DepictQA(silent=silent)
        self.gpt4 = gpt4 if gpt4 is not None else GPT4(
            config_path=config_path, system_message=prompts.system_message, silent=silent
        )

        # 配置：低分辨率尺寸启发式（来源 AgenticIR，非论文正文，见 M09）
        self._cfg: dict = {}
        if Path(config_path).exists():
            self._cfg = yaml.safe_load(Path(config_path).read_text()) or {}
        self.low_res_max_side = int(self._cfg.get("LOW_RES_MAX_SIDE", 300))
        self.use_size_heuristic_sr = bool(self._cfg.get("USE_SIZE_HEURISTIC_FOR_SR", True))

        # 经验：要求真实、非空、有来源；否则显式切 no-experience 档位（M04）
        self.with_retrieval, self.schedule_experience, self.experience_source = (
            self._load_experience(experience_path, with_retrieval)
        )

    # —— 经验加载与校验 ——
    def _load_experience(self, experience_path, with_retrieval):
        if not with_retrieval:
            return False, "", None
        if experience_path is None:
            self._log("经验模式：未提供经验文件，显式切到 no-experience 档位。")
            return False, "", None
        p = Path(experience_path)
        if not p.exists():
            self._log(f"经验模式：文件不存在 {p}，显式切到 no-experience 档位。")
            return False, "", None
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            self._log(f"经验文件无法解析 {p}: {e}，显式切到 no-experience 档位。")
            return False, "", None
        distilled = (data.get("distilled") or "").strip()
        if not distilled:
            self._log(f"经验文件 {p} 的 distilled 字段为空，显式切到 no-experience 档位。")
            return False, "", None
        source = data.get("source") or (data.get("_meta") or {}).get("source")
        if not source:
            self._log(f"经验文件 {p} 缺少来源记录，标记来源未知。")
            source = "unknown"
        return True, distilled, source

    def _log(self, msg):
        if not self.silent:
            print(f"[Scheduler] {msg}", flush=True)
        if self.trajectory is not None:
            self.trajectory.record("experience", message=msg)

    # —— 粗感知 ——
    def perceive(self, lq_path: Path) -> list[tuple[Degradation, Level]]:
        """评估全部 7 类退化（不含 low resolution，其由图像尺寸判断）。"""
        evaluation = json.loads(self.depictqa(lq_path, task="eval_degradation"))
        if self.trajectory is not None:
            self.trajectory.record("coarse_perception", evaluation=evaluation)
        return evaluation

    # —— 三阶段约束规划 ——
    def plan(self, lq_path: Path, evaluation: list[tuple[Degradation, Level]],
             instruction: str = "") -> list[Subtask]:
        """按指令过滤 medium+ → 按三阶段分组 → 阶段内多模态排序 → 拼接。"""
        inst = InstructionSpec.parse(instruction)
        agenda: list[Subtask] = []

        # low resolution 由尺寸启发式判断（可配置，见 M09）；同样受可用工具检查约束
        if self.use_size_heuristic_sr:
            img_shape = cv2.imread(str(lq_path)).shape[:2]
            if max(img_shape) < self.low_res_max_side:
                if not inst.should_process("low resolution"):
                    self._record_skip("low resolution", "instruction")
                elif not available_tools("super-resolution"):
                    self._record_skip("low resolution", "no_available_tools")
                else:
                    agenda.append("super-resolution")

        for degradation, severity in evaluation:
            subtask = self.degra_subtask_dict.get(degradation)
            if subtask is None:
                continue
            if self.levels.index(severity) < 2:  # 低于 medium，不处理
                continue
            if not available_tools(subtask):  # 无可用工具，跳过（M07）
                self._record_skip(degradation, "no_available_tools")
                continue
            if not inst.should_process(degradation):
                self._record_skip(degradation, "instruction")
                continue
            agenda.append(subtask)

        agenda = list(dict.fromkeys(agenda))  # 去重（P2）

        grouped: dict[int, list[Subtask]] = {1: [], 2: [], 3: []}
        for subtask in agenda:
            grouped[subtask_stage(subtask)].append(subtask)

        plan: list[Subtask] = []
        for stage in (1, 2, 3):
            stage_agenda = grouped[stage]
            if len(stage_agenda) <= 1:
                plan.extend(stage_agenda)
            else:
                plan.extend(self._order_within_stage(lq_path, evaluation, inst, stage_agenda))

        if self.trajectory is not None:
            self.trajectory.record("plan", plan=plan, instruction=instruction,
                                   experience_source=self.experience_source,
                                   with_retrieval=self.with_retrieval)
        return plan

    def _record_skip(self, degradation, reason):
        if self.trajectory is not None:
            self.trajectory.record("skip_degradation", degradation=degradation, reason=reason)

    # —— 阶段内排序 ——
    def _order_within_stage(self, lq_path, evaluation, inst, agenda):
        degradations = [self.subtask_degra_dict[s] for s in agenda]
        perception = ", ".join(
            f"{d}: {lvl}" for d, lvl in evaluation if d in degradations
        )
        try:
            if self.with_retrieval:
                return self._schedule_w_retrieval(lq_path, degradations, agenda, perception, inst)
            return self._schedule_wo_retrieval(lq_path, degradations, agenda, perception, inst)
        except Exception as e:
            # LLM 排序/格式解析失败：回退到确定性顺序（严重度降序），不中断整条流程
            self._log(f"阶段内排序失败，回退到严重度降序: {type(e).__name__}: {e}")
            sev = dict(evaluation)
            return sorted(
                agenda,
                key=lambda s: self.levels.index(
                    sev.get(self.subtask_degra_dict[s], "very low")
                ),
                reverse=True,
            )

    def _schedule_w_retrieval(self, lq_path, degradations, agenda, perception, inst):
        def check_order(schedule):
            assert isinstance(schedule, dict) and set(schedule.keys()) == {"thought", "order"}
            order = schedule["order"]
            assert isinstance(order, list)
            assert len(order) == len(agenda) and len(set(order)) == len(agenda)
            assert set(order) == set(agenda)

        schedule = self.gpt4(
            img_path=lq_path,
            prompt=prompts.schedule_w_retrieval_prompt.format(
                degradations=degradations, agenda=agenda,
                perception=perception, instruction=inst.text,
                experience=self.schedule_experience,
            ),
            format_check=check_order,
        )
        return json.loads(schedule)["order"]

    def _schedule_wo_retrieval(self, lq_path, degradations, agenda, perception, inst):
        insights = self.gpt4(
            img_path=lq_path,
            prompt=prompts.reason_to_schedule_prompt.format(
                degradations=degradations, agenda=agenda,
                perception=perception, instruction=inst.text,
            )
        )

        def check_order(order):
            assert isinstance(order, list)
            assert len(order) == len(agenda) and len(set(order)) == len(agenda)
            assert set(order) == set(agenda)

        order = self.gpt4(
            img_path=lq_path,
            prompt=prompts.schedule_wo_retrieval_prompt.format(
                degradations=degradations, agenda=agenda,
                perception=perception, instruction=inst.text, insights=insights
            ),
            format_check=check_order,
        )
        return json.loads(order)
