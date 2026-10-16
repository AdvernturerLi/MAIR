"""Table 15 inventory contract. Missing adapters are reported, never fabricated."""
import registry
from executor import executor

ALGORITHM_VERSION = "mair-ijcv2026-algorithm3-v2"
PAPER_TOOLS = {
    "super-resolution": ["diffbir", "diffbir_realesrgan", "hat", "swinir_gan", "swinir_psnr", "xrestormer"],
    "denoising": ["swinir_15", "swinir_50", "maxim", "mprnet", "restormer", "xrestormer"],
    "jpeg compression artifact removal": ["swinir_40", "fbcnn_90", "fbcnn_5", "fbcnn_blind"],
    "motion deblurring": ["restormer", "maxim", "mprnet", "xrestormer"],
    "defocus deblurring": ["drbnet", "restormer"],
    "deraining": ["restormer", "maxim", "mprnet", "xrestormer"],
    "dehazing": ["dehazeformer_outdoor_b", "dehazeformer_indoor_l", "dehazeformer_indoor_b",
                 "ridcp", "maxim", "xrestormer", "dwgan"],
    "brightening": ["gamma_correction", "constant_shift", "histogram_equalization", "retinexformer", "cotf"],
}


def inventory():
    rows = []
    for subtask, names in PAPER_TOOLS.items():
        available = set(registry.available_tools(subtask))
        registered = set(registry.tool_names(subtask))
        implemented = {t.tool_name for t in executor.toolbox_router.get(subtask, [])}
        for name in names:
            rows.append({"subtask": subtask, "tool": name,
                         "registered": name in registered,
                         "implemented": name in implemented, "available": name in available})
    return rows


def validate_paper_resources(scheduler):
    gaps = []
    if getattr(scheduler.gpt4, "model", None) != "gpt-4o":
        gaps.append("GPT-4o required; check OPENAI_MODEL environment override")
    if not scheduler.with_retrieval:
        gaps.append("Published experience is disabled")
    if not getattr(scheduler, "_cfg", {}).get("ENABLE_MAXIM", False):
        gaps.append("MAXIM runtime not enabled; verify its isolated JAX/Flax environment first")
    gaps.extend(f"{r['subtask']}/{r['tool']}" for r in inventory() if not r["available"])
    if gaps:
        raise RuntimeError("Paper configuration prerequisites missing: " + "; ".join(gaps))
