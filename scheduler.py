"""MAIR Level 1 调度器（Scheduler）。

职责：粗感知退化 + 结合三阶段先验与用户指令规划。
  - perceive：用 DepictQA 评估 7 类退化的严重度（粗感知）。
  - plan：按指令过滤 → 按三阶段（compression→imaging→scene）分组 →
          阶段内由 GPT-4o(+经验) 排序（多模态：原图+粗感知+指令）→ 拼接。
"""
import json
import hashlib
from pathlib import Path
from typing import Optional

import cv2
import yaml

from llm import DepictQA, GPT4
from registry import available_tools
from stage_prior import DEGRADATION_TO_SUBTASK, subtask_stage, STAGE_NAMES
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
            self._cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        self.low_res_max_side = int(self._cfg.get("LOW_RES_MAX_SIDE", 300))
        self.use_size_heuristic_sr = bool(self._cfg.get("USE_SIZE_HEURISTIC_FOR_SR", True))

        # Default to the published Sec.8 experience; invalid explicit files fail.
        self.with_retrieval, self.schedule_experience, self.experience_source = (
            self._load_experience(experience_path, with_retrieval)
        )

    # —— 经验加载与校验 ——
    def _load_experience(self, experience_path, with_retrieval):
        self.experience_hash = None
        self.experience_path = None
        if not with_retrieval:
            return False, "", None
        if experience_path is None:
            experience_path = Path(__file__).parent / "experience" / "paper_ijcv2026.json"
        p = Path(experience_path)
        if not p.exists():
            raise ValueError(f"Experience file does not exist: {p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"Invalid experience file: {p}") from e
        if not isinstance(data, dict):
            raise ValueError("Experience must be a JSON object")
        distilled, source = data.get("distilled"), data.get("source")
        if not isinstance(distilled, str) or not distilled.strip():
            raise ValueError("Experience requires a nonempty distilled string")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Experience requires a nonempty source string")
        self.experience_path = p.resolve()
        self.experience_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        if self.trajectory is not None:
            self.trajectory.record("experience_loaded", path=self.experience_path,
                                   sha256=self.experience_hash, source=source)
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
        """粗感知清单 → 指令语义更新 → 三阶段内规划；模型失败直接报告。"""
        inst = InstructionSpec.parse(instruction)
        agenda: list[Subtask] = []

        # low resolution 由尺寸启发式判断（可配置，见 M09）；同样受可用工具检查约束
        if self.use_size_heuristic_sr:
            image = cv2.imread(str(lq_path))
            if image is None:
                raise ValueError(f"Cannot read image: {lq_path}")
            img_shape = image.shape[:2]
            if max(img_shape) < self.low_res_max_side:
                if not inst.should_process("low resolution"):
                    self._record_skip("low resolution", "instruction")
                elif not self._available_tools("super-resolution"):
                    self._record_skip("low resolution", "no_available_tools")
                else:
                    agenda.append("super-resolution")

        perception_threshold = self._cfg.get("PERCEPTION_THRESHOLD", "medium")
        if perception_threshold not in self.levels:
            raise ValueError("Invalid PERCEPTION_THRESHOLD")
        for degradation, severity in evaluation:
            subtask = self.degra_subtask_dict.get(degradation)
            if subtask is None:
                continue
            if self.levels.index(severity) < self.levels.index(perception_threshold):
                continue
            if not self._available_tools(subtask):
                self._record_skip(degradation, "no_available_tools")
                continue
            if not inst.should_process(degradation):
                self._record_skip(degradation, "instruction")
                continue
            agenda.append(subtask)

        agenda = list(dict.fromkeys(agenda))  # 去重（P2）

        # Sec.8 tschedule: instructions may update (not merely reorder) the agenda.
        # With no instruction, the update is an identity operation.
        if instruction.strip():
            agenda = self._update_degradations(lq_path, evaluation, agenda, instruction)

        grouped: dict[int, list[Subtask]] = {1: [], 2: [], 3: []}
        for subtask in agenda:
            grouped[subtask_stage(subtask)].append(subtask)

        plan: list[Subtask] = []
        for stage in (1, 2, 3):
            stage_agenda = grouped[stage]
            if not stage_agenda:
                continue
            plan.extend(self._paper_order(lq_path, evaluation, inst, stage_agenda, stage, plan))

        if self.trajectory is not None:
            self.trajectory.record("plan", plan=plan, instruction=instruction,
                                   experience_source=self.experience_source,
                                   experience_sha256=getattr(self, "experience_hash", None),
                                   with_retrieval=self.with_retrieval)
        return plan

    def _update_degradations(self, image, evaluation, agenda, instruction):
        allowed = list(self.degra_subtask_dict)

        def check(obj):
            assert isinstance(obj, dict) and set(obj) == {"thought", "updated degradations"}
            assert isinstance(obj["thought"], str)
            values = obj["updated degradations"]
            assert isinstance(values, list) and all(isinstance(v, str) for v in values)
            assert len(values) == len(set(values))
            assert all(v in allowed for v in values)

        result = json.loads(self.gpt4(img_path=image,
            prompt=prompts.update_degradations_prompt.format(
                instruction=instruction, degradations=[self.subtask_degra_dict[s] for s in agenda],
                perception=evaluation, allowed=allowed), format_check=check))
        check(result)
        updated = []
        for degradation in result["updated degradations"]:
            subtask = self.degra_subtask_dict[degradation]
            if not self._available_tools(subtask):
                self._record_skip(degradation, "no_available_tools")
                if self._cfg.get("REQUIRE_PAPER_RESOURCES", False):
                    raise RuntimeError(f"Requested degradation has no available tools: {degradation}")
                continue
            updated.append(subtask)
        if self.trajectory is not None:
            self.trajectory.record("instruction_update", before=agenda, after=updated,
                                   instruction=instruction, thought=result["thought"])
        return updated

    def _paper_order(self, image, evaluation, inst, agenda, stage, previous_plan):
        def check(obj):
            assert isinstance(obj, dict) and set(obj) == {"thought", "order"}
            assert isinstance(obj["thought"], str)
            order = obj["order"]
            assert isinstance(order, list) and all(isinstance(s, str) for s in order)
            assert len(order) == len(agenda) and set(order) == set(agenda)

        result = json.loads(self.gpt4(img_path=image,
            prompt=prompts.paper_schedule_prompt.format(
                degradations=[self.subtask_degra_dict[s] for s in agenda], agenda=agenda,
                stage=STAGE_NAMES[stage], previous_plan=previous_plan, perception=evaluation,
                instruction=inst.text,
                experience=self.schedule_experience if self.with_retrieval else "No experience (ablation)."),
            format_check=check))
        check(result)
        if self.trajectory is not None:
            self.trajectory.record("stage_plan", stage=stage, order=result["order"],
                                   thought=result["thought"])
        return result["order"]

    def _record_skip(self, degradation, reason):
        if self.trajectory is not None:
            self.trajectory.record("skip_degradation", degradation=degradation, reason=reason)

    def _available_tools(self, subtask):
        names = available_tools(subtask)
        if not self._cfg.get("ENABLE_MAXIM", False):
            names = [n for n in names if n != "maxim"]
        if self._cfg.get("REQUIRE_PAPER_RESOURCES", False) or self._cfg.get("USE_PAPER_TOOLSET", False):
            from paper_resources import PAPER_TOOLS
            names = [n for n in names if n in PAPER_TOOLS[subtask]]
        return names
