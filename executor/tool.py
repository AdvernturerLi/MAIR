import os
from pathlib import *
import subprocess
import time
from typing import Optional


# 权重文件扩展名（用于 M07 可用性检查）
WEIGHT_EXTS = {".pth", ".ckpt", ".npz", ".pytorch", ".safetensors", ".pt", ".onnx"}

# 工具可用性缓存（工具是 executor 单例，进程内稳定）
_availability_cache: dict = {}


class Tool:
    """Abstract class for a tool.

    Args:
        tool_name (str): Tool name, a valid identifier serving as the name of environment, configuration file, etc. An exception is that if there's '_' in the name, the environment name will be the part before '_'.
        subtask (str): Subtask name, serving as the name of the directory for the subtask.
        work_dir (str | None, optional): Basename of working directory. Defaults to None.
        script_rel_path (Path | str | None, optional): Path relative to the working directory of the script to run. Defaults to None.
    """

    def __init__(self,
                 tool_name: str,
                 subtask: str, 
                 work_dir: Optional[Path] = None,
                 script_rel_path: Optional[Path | str] = None,
                 ):
        self.tool_name: str = tool_name
        self.subtask: str = subtask
        self.work_dir: Optional[Path] = None
        self.script_path: Optional[Path] = None
        # 具体权重文件（相对 work_dir），用于 M07 可用性检查；为空则退回"任意权重"检查
        self.required_weight_files: list[Path] = []
        if work_dir is not None:
            assert script_rel_path is not None, "If `work_dir` is provided, `script_rel_path` should also be provided."
            self.work_dir: Path = Path().resolve() / 'executor' / subtask / 'tools' / work_dir
            self.script_path: Path = self.work_dir / script_rel_path

    def __call__(self, input_dir: Path, output_dir: Path, silent: bool = False, *args) -> None:
        """Executes the tool. `input_dir` should be absolute and only contain the input image, and `output_dir` should be empty, which will only contain the output image named `output.png` after the execution."""
        if not silent:
            print('-'*100)
            print(f"Subtask\t: {self.subtask}")
            print(f"Tool\t: {self.tool_name}")
            print(f"Input\t: {list(input_dir.glob('*'))[0]}")
        start_time = time.time()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self._precheck()
        self._invoke(*args)
        self._postcheck()
        end_time = time.time()
        if not silent:
            print(f"Output\t: {list(output_dir.glob('*'))[0]}")
            print(f"Time\t: {round(end_time - start_time, 3)}s")

    def _precheck(self) -> None:
        """Checks whether `input_dir` contains the input image named `input.png` only and `output_dir` is empty."""
        assert len(os.listdir(self.input_dir)) == 1, "The input directory should contain the input only."
        assert os.listdir(self.output_dir) == [], "The output directory should be empty."

    def _postcheck(self) -> None:
        """Ensures that `output_dir` contains only the output image named `output.png`."""
        output = list(self.output_dir.glob('*'))
        assert len(output) == 1, "There're other files in the same directory as the output image."
        if output[0].name != 'output.png':
            # rename to `output.png`
            output[0].replace(self.output_dir / 'output.png')

    def _invoke(self) -> None:
        self._preprocess()
        cmd = self._get_cmd()
        proc = subprocess.run(cmd, cwd=self.work_dir, shell=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Tool {self.tool_name} failed (exit {proc.returncode}): {stderr[-2000:]}"
            )
        self._postprocess()

    def _get_cmd(self) -> str:
        opts = self._get_cmd_opts()
        env_name = self.tool_name.split('_')[0]
        # 直接用 env 的 python，避免 conda run 的 exit-code/缓冲问题与 PATH 依赖
        python = f"/root/miniconda3/envs/{env_name}/bin/python"
        cmd = f"'{python}' '{self.script_path}'"
        for opt in opts:
            cmd += f" '{opt}'"
        return cmd

    def _get_cmd_opts(self, *args) -> list[str]:
        raise NotImplementedError

    # —— 可用性检查（M07）——
    @staticmethod
    def _exists(p: Path) -> bool:
        """exists() 的安全包装：无权限/OSError 视为不存在（不崩溃）。"""
        try:
            return Path(p).exists()
        except (PermissionError, OSError):
            return False

    def availability(self) -> dict:
        """返回 {code, env, weights, available}：脚本代码、conda 环境、权重是否齐备。

        纯 Python 工具（如 brightening）在主进程内运行，无外部脚本/环境/权重依赖。
        """
        key = id(self)
        if key in _availability_cache:
            return _availability_cache[key]
        if self.script_path is None:
            info = {"code": True, "env": True, "weights": True, "available": True}
        else:
            code = self._exists(self.script_path)
            env_name = self.tool_name.split("_")[0]
            env = self._exists(Path(f"/root/miniconda3/envs/{env_name}/bin/python"))
            weights = self._weights_present()
            info = {"code": code, "env": env, "weights": weights,
                    "available": code and env and weights}
        _availability_cache[key] = info
        return info

    def available(self) -> bool:
        return self.availability()["available"]

    def _weights_present(self) -> bool:
        """工具是否具备权重：声明了 required_weight_files 则逐个校验，否则
        退回"目录下存在任意权重文件"（此时同 work_dir 多变体工具可能被高估）。"""
        if self.work_dir is None:
            return False
        try:
            if not self.work_dir.exists():
                return False
            if self.required_weight_files:
                return all(
                    self._exists(self.work_dir / f) for f in self.required_weight_files
                )
            for p in self.work_dir.rglob("*"):
                try:
                    if p.is_file() and p.suffix.lower() in WEIGHT_EXTS:
                        return True
                except (PermissionError, OSError):
                    continue
        except (PermissionError, OSError):
            return False
        return False

    def _preprocess(self):
        """May be needed by the specific tool."""        
        pass

    def _postprocess(self):
        """May be needed by the specific tool."""        
        pass
