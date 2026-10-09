"""MAIR 入口：调度器感知规划 → 专家顺序执行。"""
import argparse
import shutil
from pathlib import Path

from scheduler import Scheduler
from expert import Expert


def main() -> None:
    parser = argparse.ArgumentParser(description="MAIR: multi-agent image restoration")
    parser.add_argument("--input", type=Path, required=True, help="LQ 输入图")
    parser.add_argument("--output", type=Path, default=Path("output"), help="输出根目录")
    parser.add_argument("--config", type=Path, default=Path("config.yml"))
    parser.add_argument("--experience", type=Path, default=None, help="调度经验 json（distilled 字段）")
    parser.add_argument("--no-retrieval", action="store_true", help="阶段内排序不用经验")
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()

    scheduler = Scheduler(
        config_path=args.config,
        experience_path=args.experience,
        with_retrieval=not args.no_retrieval,
        silent=args.silent,
    )

    # 1. 粗感知 + 三阶段规划
    evaluation = scheduler.perceive(args.input)
    plan = scheduler.plan(args.input, evaluation)
    print(f"[Scheduler] 感知: {evaluation}")
    print(f"[Scheduler] 计划(三阶段): {plan}")

    # 2. 准备干净的输入目录（工具要求 input_dir 恰好 1 张图）
    args.output.mkdir(parents=True, exist_ok=True)
    clean_input_dir = args.output / "0-img"
    clean_input_dir.mkdir(parents=True, exist_ok=True)
    cur_img = clean_input_dir / "input.png"
    shutil.copy(args.input, cur_img)

    # 3. 依次执行专家
    for subtask in plan:
        degradation = scheduler.subtask_degra_dict[subtask]
        expert = Expert(subtask, degradation, config_path=args.config, silent=args.silent)
        out = expert.restore(cur_img, args.output)
        if out is None:
            print(f"[MAIR] 专家 {subtask} 失败，中止", flush=True)
            return
        cur_img = out

    # 4. 保存最终结果
    final = args.output / "result.png"
    shutil.copy(cur_img, final)
    print(f"[MAIR] 最终结果: {final}")


if __name__ == "__main__":
    main()
