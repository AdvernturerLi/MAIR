"""MAIR 工具注册表（Tool Registry）。

把每个工具的功能、适用场景、效率、特性写成表单，喂给 GPT-4o 让专家智能
选候选工具（替代 AgenticIR 的「跑全部工具」）。字段：
  name   —— 工具名（与 executor.toolbox_router 一致）
  desc   —— 功能描述
  suitable —— 适用场景（退化程度）
  speed  —— 推理效率（快/中/慢）
  note   —— 特性/副作用
"""
from utils.custom_types import Subtask
from executor import executor

# subtask → 工具表单列表
REGISTRY: dict[Subtask, list[dict]] = {
    "super-resolution": [
        {"name": "diffbir", "desc": "DiffBIR 生成式超分", "suitable": "严重低分辨率、需生成细节", "speed": "慢", "note": "生成式，保真度(PSNR)一般但细节强"},
        {"name": "xrestormer", "desc": "X-Restormer 超分", "suitable": "一般低分辨率", "speed": "中", "note": "判别式，过度平滑"},
        {"name": "swinir_gan", "desc": "SwinIR 超分（GAN 版）", "suitable": "中等低分辨率", "speed": "中", "note": "细节丰富，可能有伪影"},
        {"name": "swinir_psnr", "desc": "SwinIR 超分（PSNR 版）", "suitable": "轻度低分辨率", "speed": "中", "note": "保真优先"},
        {"name": "hat", "desc": "HAT 超分", "suitable": "中等低分辨率", "speed": "中", "note": "混合注意力，性能好"},
    ],
    "denoising": [
        {"name": "xrestormer", "desc": "X-Restormer 去噪", "suitable": "一般噪声", "speed": "中", "note": "判别式，平滑"},
        {"name": "swinir_15", "desc": "SwinIR 去噪（噪声15）", "suitable": "轻度噪声", "speed": "中", "note": "针对低噪声"},
        {"name": "swinir_50", "desc": "SwinIR 去噪（噪声50）", "suitable": "重度噪声", "speed": "中", "note": "针对高噪声"},
        {"name": "mprnet", "desc": "MPRNet 去噪", "suitable": "一般噪声", "speed": "中", "note": "多阶段渐进"},
        {"name": "maxim", "desc": "MAXIM 去噪", "suitable": "一般噪声", "speed": "慢", "note": "JAX 实现，慢"},
        {"name": "restormer", "desc": "Restormer 去噪", "suitable": "一般噪声", "speed": "中", "note": "Transformer"},
    ],
    "motion deblurring": [
        {"name": "restormer", "desc": "Restormer 运动去模糊", "suitable": "一般运动模糊", "speed": "中", "note": ""},
        {"name": "mprnet", "desc": "MPRNet 去模糊", "suitable": "一般运动模糊", "speed": "中", "note": ""},
        {"name": "maxim", "desc": "MAXIM 去模糊", "suitable": "一般运动模糊", "speed": "慢", "note": "JAX，慢"},
        {"name": "xrestormer", "desc": "X-Restormer 去模糊", "suitable": "一般运动模糊", "speed": "中", "note": ""},
    ],
    "defocus deblurring": [
        {"name": "drbnet", "desc": "DRBNet 离焦去模糊", "suitable": "离焦模糊", "speed": "中", "note": "专门离焦"},
        {"name": "restormer", "desc": "Restormer 单图离焦去模糊", "suitable": "离焦模糊", "speed": "中", "note": ""},
        {"name": "ifan", "desc": "IFAN 离焦去模糊", "suitable": "离焦模糊", "speed": "慢", "note": "需 LPIPS 权重"},
    ],
    "deraining": [
        {"name": "maxim", "desc": "MAXIM 去雨", "suitable": "一般雨", "speed": "慢", "note": "JAX，慢"},
        {"name": "xrestormer", "desc": "X-Restormer 去雨", "suitable": "一般雨", "speed": "中", "note": ""},
        {"name": "restormer", "desc": "Restormer 去雨", "suitable": "一般雨", "speed": "中", "note": ""},
        {"name": "mprnet", "desc": "MPRNet 去雨", "suitable": "一般雨", "speed": "中", "note": ""},
    ],
    "dehazing": [
        {"name": "xrestormer", "desc": "X-Restormer 去雾", "suitable": "一般雾", "speed": "中", "note": ""},
        {"name": "ridcp", "desc": "RIDCP 去雾", "suitable": "一般雾", "speed": "中", "note": ""},
        {"name": "dehazeformer", "desc": "DehazeFormer 去雾", "suitable": "一般雾", "speed": "中", "note": "Transformer"},
        {"name": "dehazeformer_outdoor_b", "desc": "DehazeFormer Outdoor-B 去雾", "suitable": "室外场景；Table 15 明确列出此变体", "speed": "未核实", "note": "权重 saved_models/outdoor/dehazeformer-b.pth"},
        {"name": "dehazeformer_indoor_b", "desc": "DehazeFormer Indoor-B 去雾", "suitable": "室内场景；Table 15 明确列出此变体", "speed": "未核实", "note": "权重 saved_models/indoor/dehazeformer-b.pth"},
        {"name": "dehazeformer_indoor_l", "desc": "DehazeFormer Indoor-L 去雾", "suitable": "室内场景；Table 15 明确列出此变体", "speed": "未核实", "note": "权重 saved_models/indoor/dehazeformer-l.pth"},
        {"name": "maxim", "desc": "MAXIM 去雾", "suitable": "一般雾", "speed": "慢", "note": "JAX，慢"},
    ],
    "brightening": [
        {"name": "histogram_equalization", "desc": "直方图均衡(CLAHE)", "suitable": "轻度低光", "speed": "快", "note": "纯 OpenCV"},
        {"name": "gamma_correction", "desc": "Gamma 校正(gamma=1.5)", "suitable": "轻度低光", "speed": "快", "note": "纯 OpenCV"},
        {"name": "constant_shift", "desc": "V 通道常量提升(+40)", "suitable": "轻度低光", "speed": "快", "note": "纯 OpenCV"},
    ],
    "jpeg compression artifact removal": [
        {"name": "fbcnn_blind", "desc": "FBCNN 去 JPEG 伪影(盲)", "suitable": "未知质量因子", "speed": "中", "note": "盲估计 QF"},
        {"name": "fbcnn_5", "desc": "FBCNN 去 JPEG 伪影(QF=5)", "suitable": "重度 JPEG 压缩", "speed": "中", "note": ""},
        {"name": "fbcnn_90", "desc": "FBCNN 去 JPEG 伪影(QF=90)", "suitable": "轻度 JPEG 压缩", "speed": "中", "note": ""},
        {"name": "swinir_40", "desc": "SwinIR 去 JPEG 伪影(jpeg40)", "suitable": "中度 JPEG 压缩", "speed": "中", "note": ""},
    ],
}


from paper_registry import FORMS as PAPER_FORMS

# Keep independently written forms distinguishable from the published examples.
for _task, _forms in REGISTRY.items():
    for _form in _forms:
        _form["source"] = "locally authored; not a published MAIR registry form"
for _task, _forms in PAPER_FORMS.items():
    for _form in _forms:
        _existing = next((i for i, f in enumerate(REGISTRY[_task]) if f["name"] == _form["name"]), None)
        if _existing is None:
            REGISTRY[_task].append(_form)
        else:
            REGISTRY[_task][_existing] = _form

# 注意：不再维护硬编码白名单（M07）。运行时以 available_tools() 的
# 「代码 + 环境 + 权重」可用性检查为准，新增工具经检查即可被选择，
# 不需要悄悄改白名单。


def available_tools(subtask: Subtask) -> list[str]:
    """某 subtask 在注册表里有表单、且实际可用（代码+环境+权重）的工具名列表。

    按注册表顺序返回，作为专家选工具与调度器判定退化是否可处理的唯一来源。
    """
    names = tool_names(subtask)
    result = []
    for t in executor.toolbox_router.get(subtask, []):
        if t.tool_name in names and t.available():
            result.append(t.tool_name)
    result.sort(key=lambda n: names.index(n))
    return result


def registry_text(subtask: Subtask) -> str:
    """把某 subtask 的工具表单格式化成文本，喂给 GPT-4o。"""
    tools = REGISTRY.get(subtask, [])
    lines = []
    for t in tools:
        lines.append(
            f"- {t['name']}: {t['desc']}；适用：{t['suitable']}；效率：{t['speed']}；"
            f"{'注意：' + t['note'] if t['note'] else ''}"
        )
    return "\n".join(lines)


def tool_names(subtask: Subtask) -> list[str]:
    """某 subtask 在注册表里的工具名列表。"""
    return [t["name"] for t in REGISTRY.get(subtask, [])]
