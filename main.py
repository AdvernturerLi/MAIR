"""MAIR 入口：调度器感知规划 → 专家顺序执行，并保存机器可读轨迹（M10）。"""
import argparse
import shutil
from pathlib import Path

import yaml

from scheduler import Scheduler
from expert import Expert
from trajectory import Trajectory


def _is_within(path: Path, ancestor: Path) -> bool:
    """path 是否位于 ancestor 目录内（含相等）。"""
    path = path.resolve()
    ancestor = ancestor.resolve()
    return path == ancestor or ancestor in path.parents


def _require_empty_output(out: Path, input_path: Path, force: bool) -> None:
    """运行隔离：输出目录已存在且非空时，除非 --force，否则明确失败。

    --force 前做安全校验：禁止删除项目根目录、输入图文件（及其所在目录）、
    以及这些路径之上的任意祖先目录。
    """
    out = out.resolve()
    if not out.exists() or not any(out.iterdir()):
        return
    if not force:
        raise SystemExit(
            f"输出目录非空: {out}。请换一个新目录，或用 --force 清空后重跑。"
        )
    project_root = Path(__file__).resolve().parent
    input_file = Path(input_path).resolve()
    for protected in (project_root, input_file):
        if _is_within(protected, out):
            raise SystemExit(
                f"拒绝删除受保护路径: {out}（其包含 {protected}）。请另选输出目录。"
            )
    shutil.rmtree(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="MAIR: multi-agent image restoration")
    parser.add_argument("--input", type=Path, required=True, help="LQ 输入图")
    parser.add_argument("--output", type=Path, default=Path("output"), help="输出根目录")
    parser.add_argument("--config", type=Path, default=Path("config.yml"))
    parser.add_argument("--experience", type=Path, default=None,
                        help="调度经验 json（含 distilled 与 source）")
    parser.add_argument("--instruction", type=str, default="",
                        help="用户指令（如：保留雾但去雨）")
    parser.add_argument("--no-retrieval", action="store_true", help="阶段内排序不用经验")
    parser.add_argument("--force", action="store_true", help="输出目录非空时清空重跑")
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()

    _require_empty_output(args.output, args.input, args.force)
    args.output.mkdir(parents=True, exist_ok=True)

    trajectory = Trajectory(args.output, input_path=args.input)

    scheduler = Scheduler(
        config_path=args.config,
        experience_path=args.experience,
        with_retrieval=not args.no_retrieval,
        silent=args.silent,
        trajectory=trajectory,
    )

    # 记录实际生效的模型/配置/指令（M08/M10）；模型取已构造客户端的解析结果
    # （gpt4 内部已处理环境变量覆盖，避免轨迹记错模型）
    cfg = {}
    if args.config.exists():
        cfg = yaml.safe_load(args.config.read_text()) or {}
    trajectory.record(
        "config",
        provider=getattr(scheduler.gpt4, "base_url", None),
        model=getattr(scheduler.gpt4, "model", None),
        low_res_max_side=cfg.get("LOW_RES_MAX_SIDE", 300),
        stop_threshold=cfg.get("STOP_THRESHOLD", "low"),
        instruction=args.instruction,
        input_hash=trajectory.input_hash,
        experience=args.experience,
    )

    try:
        # 1. 粗感知 + 三阶段规划
        evaluation = scheduler.perceive(args.input)
        plan = scheduler.plan(args.input, evaluation, instruction=args.instruction)
        print(f"[Scheduler] 感知: {evaluation}")
        print(f"[Scheduler] 计划(三阶段): {plan}")

        # 2. 准备干净的输入目录（工具要求 input_dir 恰好 1 张图）
        clean_input_dir = args.output / "0-img"
        clean_input_dir.mkdir(parents=True, exist_ok=True)
        cur_img = clean_input_dir / "input.png"
        shutil.copy(args.input, cur_img)

        # 3. 依次执行专家
        for subtask in plan:
            degradation = scheduler.subtask_degra_dict[subtask]
            expert = Expert(subtask, degradation, config_path=args.config,
                            silent=args.silent, trajectory=trajectory)
            out = expert.restore(cur_img, args.output, instruction=args.instruction)
            if out is None:
                print(f"[MAIR] 专家 {subtask} 失败，中止", flush=True)
                trajectory.record("abort", subtask=subtask, reason="expert returned None")
                trajectory.save()
                return
            cur_img = out

        # 4. 保存最终结果
        final = args.output / "result.png"
        shutil.copy(cur_img, final)
        trajectory.record("final", result=str(final))
        traj_path = trajectory.save()
        print(f"[MAIR] 最终结果: {final}")
        print(f"[MAIR] 轨迹: {traj_path}")
    except Exception as e:
        # 感知/规划/执行抛异常时，也保存失败轨迹（M10）
        trajectory.record("error", error=f"{type(e).__name__}: {e}")
        trajectory.save()
        raise


if __name__ == "__main__":
    main()
