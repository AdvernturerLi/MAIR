"""MAIR 工具盘点（M07）。

打印每个 subtask 下注册工具的 code / env / weights 可用性，以及每个任务的
可用候选数。不启动 GPU，仅做文件系统与环境检查，作为"哪些工具真正可用"
的交付证据。
"""
from executor import executor
from registry import REGISTRY, tool_names, available_tools
from stage_prior import DEGRADATION_TO_SUBTASK


def main() -> None:
    print("=" * 84)
    print("MAIR 工具盘点（code / env / weights）")
    print("=" * 84)
    for subtask in sorted(executor.subtasks):
        tools = executor.toolbox_router.get(subtask, [])
        if not tools:
            continue
        avail = available_tools(subtask)
        print(f"\n[{subtask}] 注册表单 {len(tool_names(subtask))} 个，可用 {len(avail)} 个")
        print(f"  可用候选: {avail if avail else '(无)'}")
        for t in tools:
            info = t.availability()
            marks = [f"{k}={info[k]}" for k in ("code", "env", "weights")]
            status = "AVAILABLE" if info["available"] else "missing"
            print(f"  - {t.tool_name:26s} {' | '.join(marks):44s} {status}")
    print("\n" + "=" * 84)
    working = [d for d, s in DEGRADATION_TO_SUBTASK.items() if available_tools(s)]
    print(f"可处理退化（≥1 可用工具）: {working}")
    print("=" * 84)


if __name__ == "__main__":
    main()
