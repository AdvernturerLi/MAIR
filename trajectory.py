"""MAIR 机器可读运行轨迹（M10）。

每张图保存一份 JSON 轨迹，记录输入哈希、模型/配置、粗感知、指令、经验文件
哈希、阶段计划、每个 Expert 的细感知与候选、工具调用与输出路径、停止原因、
错误与耗时，用于确认运行的确是所声称的算法。
"""
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional


class Trajectory:
    def __init__(self, out_dir: Path, input_path: Optional[Path] = None):
        self.out_dir = Path(out_dir)
        self.entries: list[dict[str, Any]] = []
        self.start = time.time()
        self.input_hash = self._file_hash(input_path) if input_path is not None else None

    @staticmethod
    def _file_hash(path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _jsonable(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (list, tuple)):
            return [Trajectory._jsonable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: Trajectory._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (set, frozenset)):
            return sorted(obj)
        return obj

    def record(self, step: str, **fields: Any) -> None:
        self.entries.append({
            "step": step,
            "elapsed_s": round(time.time() - self.start, 3),
            **{k: self._jsonable(v) for k, v in fields.items()},
        })

    def save(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "trajectory.json"
        path.write_text(
            json.dumps(self.entries, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path
