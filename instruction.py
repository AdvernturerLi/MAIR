"""MAIR 用户指令：结构与确定性解析。

论文 Fig.7 展示 MAIR 支持自然语言指令（"保留雾但去雨"、"只去雨"、
"尽量快且更清晰"）。指令既会作为文本进入 Scheduler/Expert 的 GPT-4o 请求，
也需在确定性三阶段约束下，把"保留/仅处理某退化"落实到任务清单（M03）。

本模块只做保守、可解释的确定性解析，覆盖论文与任务书示例；无法确定时
返回空集、不臆造。真正的语义理解仍交给 GPT-4o（指令全文进 prompt）。
"""
import re
from dataclasses import dataclass, field
from typing import Optional

from utils.custom_types import Degradation


# 退化 → 常见中英文关键词（长词在前，避免 "compression" 先误命中 "artifact" 之类）
_DEGRADATION_KEYWORDS: dict[Degradation, list[str]] = {
    "jpeg compression artifact": ["jpeg", "compression artifact", "jpeg artifact", "伪影"],
    "noise": ["noise", "denoise", "噪声", "噪点", "去噪"],
    "motion blur": ["motion blur", "deblur", "运动模糊", "去模糊"],
    "defocus blur": ["defocus blur", "离焦", "失焦"],
    "low resolution": ["low resolution", "super-resolution", "低分辨率", "超分", "分辨率"],
    "rain": ["rain", "derain", "雨"],
    "haze": ["haze", "dehaze", "雾"],
    "dark": ["dark", "brighten", "low light", "低光", "暗"],
}

# "保留/跳过/不处理"标记（子句级）
_KEEP_MARKERS = ["keep", "retain", "preserve", "except", "保留", "不要处理", "不处理",
                 "跳过", "无需处理", "除了", "除"]
# "只处理/仅处理"标记
_ONLY_MARKERS = ["only", "just", "只", "仅", "只要"]

# 子句分隔符：只切明确转折/列举，不切 "和/与/and"（避免把 "保留雾和雨" 拆错）
_CLAUSE_RE = re.compile(r"[。；;,，]|但是|但|but|而", re.IGNORECASE)


def _split_clauses(instruction: str) -> list[str]:
    """把指令切成子句。"""
    return [c.strip() for c in _CLAUSE_RE.split(instruction) if c.strip()]


def _match_degradations(text: str) -> set[Degradation]:
    """返回文本中命中的退化集合（小写匹配）。"""
    s = text.lower()
    found: set[Degradation] = set()
    for deg, kws in _DEGRADATION_KEYWORDS.items():
        if any(kw in s for kw in kws):
            found.add(deg)
    return found


def parse_keep(instruction: str) -> set[Degradation]:
    """解析"保留/不处理"的退化集合（子句级）。"""
    kept: set[Degradation] = set()
    for clause in _split_clauses(instruction):
        c = clause.lower()
        if any(m in c for m in _KEEP_MARKERS):
            kept |= _match_degradations(clause)
    return kept


def parse_only(instruction: str) -> Optional[set[Degradation]]:
    """解析"只处理 X"的退化集合；非 only 指令返回 None。

    注意区分「只处理 X」与「只保留 X」：子句里同时出现"保留/keep"等
    保留标记时，属于"只保留"，归 keep 处理，不当作 only-process，
    避免把"只保留雾"误解析成"只处理雾"（反向）。
    """
    if not any(m in instruction.lower() for m in _ONLY_MARKERS):
        return None
    only: set[Degradation] = set()
    for clause in _split_clauses(instruction):
        c = clause.lower()
        has_only = any(m in c for m in _ONLY_MARKERS)
        has_keep = any(m in c for m in _KEEP_MARKERS)
        if has_only and not has_keep:
            only |= _match_degradations(clause)
    return only or None


@dataclass
class InstructionSpec:
    """指令解析结果。"""
    text: str = ""                                        # 原始指令
    keep: set[Degradation] = field(default_factory=set)   # 保留（不处理）
    only: Optional[set[Degradation]] = None               # 仅处理（非 None 时生效）

    @staticmethod
    def parse(text: str) -> "InstructionSpec":
        text = text or ""
        return InstructionSpec(text=text, keep=parse_keep(text), only=parse_only(text))

    def should_process(self, degradation: Degradation) -> bool:
        """给定退化是否应被处理。"""
        if self.only is not None:
            return degradation in self.only
        return degradation not in self.keep
