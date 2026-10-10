from pathlib import Path
import json
import requests
import logging
from typing import Optional

from .base_llm import BaseLLM
from pipeline.prompts import depictqa_evaluate_degradation_prompt, depictqa_compare_prompt
from utils.custom_types import Degradation, Level


class DepictQA(BaseLLM):
    """Parameters when called: img_path_lst, task (eval_degradation or comp_quality), degradations (if task is eval_degradation).

    感知/反思模型（论文：Vicuna-v1.5-7B 微调的 DepictQA）。本地以 HTTP 服务形式
    部署（5001=退化评估、5002=质量比较），调用带超时。
    """

    def __init__(
        self,
        log_path: Optional[Path | str] = None,
        logger: Optional[logging.Logger] = None,
        silent: bool = False,
        eval_url: str = "http://127.0.0.1:5001/evaluate_degradation",
        compare_url: str = "http://127.0.0.1:5002/compare_quality",
        timeout: float = 120.0,
        low_res_as_blur: bool = True,
    ):
        super().__init__(
            log_path=log_path, logger=logger, silent=silent
        )  # set attributes: cfg, logger, silent
        self.eval_url = eval_url
        self.compare_url = compare_url
        self.timeout = timeout
        # 超分专家用 "blur" 严重度代理 "低分辨率"（启发式，非论文正文，见 M09）
        self.low_res_as_blur = low_res_as_blur

    def query(
        self,
        img_path_lst: list[Path],
        task: str,
        degradation: Optional[Degradation] = None,
    ) -> tuple[str, str]:
        assert task in ["eval_degradation", "comp_quality"], f"Unexpected task: {task}"
        if task == "eval_degradation":
            assert (
                len(img_path_lst) == 1
            ), "Only one image should be provided for degradation evaluation."
            return self.eval_degradation(img_path_lst[0], degradation)
        else:
            assert (
                len(img_path_lst) == 2
            ), "Two images should be provided quality comparison."
            return self.compare_img_qual(img_path_lst[0], img_path_lst[1])

    def eval_degradation(
        self, img: Path, degradation: Optional[Degradation]
    ) -> tuple[str, str]:
        all_degradations: list[Degradation] = [
            "motion blur",
            "defocus blur",
            "rain",
            "haze",
            "dark",
            "noise",
            "jpeg compression artifact",
        ]
        if degradation is None:
            degradations_lst = all_degradations
        else:
            if degradation == "low resolution":
                if self.low_res_as_blur:
                    degradation = "blur"
                else:
                    # 无 blur 代理时，DepictQA 无法评估"低分辨率"，退回中性默认
                    # "medium"（非空结果，避免下游 IndexError），并如实记录
                    return depictqa_evaluate_degradation_prompt.format(
                        degradation=degradation
                    ), json.dumps([["low resolution", "medium"]])
            else:
                assert isinstance(
                    degradation, Degradation
                ), f"Unexpected type of degradations: {type(degradation)}"
                assert (
                    degradation in all_degradations
                ), f"Unexpected degradation: {degradation}"
            degradations_lst = [degradation]

        levels: set[Level] = {"very low", "low", "medium", "high", "very high"}
        res: list[list[str]] = []
        for degradation in degradations_lst:
            prompt = depictqa_evaluate_degradation_prompt.format(
                degradation=degradation
            )
            payload = {"imageA_path": img.resolve(), "prompt": prompt}
            rsp: str = requests.post(
                self.eval_url, data=payload, timeout=self.timeout
            ).json()["answer"]
            assert rsp in levels, f"Unexpected response from DepictQA: {list(rsp)}"
            res.append([degradation, rsp])

        prompt_to_display = depictqa_evaluate_degradation_prompt.format(
            degradation=degradations_lst
        )
        return prompt_to_display, json.dumps(res)

    def compare_img_qual(self, img1: Path, img2: Path) -> tuple[str, str]:
        prompt = depictqa_compare_prompt
        payload = {
            "imageA_path": img1.resolve(),
            "imageB_path": img2.resolve(),
            "prompt": prompt
        }
        rsp: str = requests.post(
            self.compare_url, data=payload, timeout=self.timeout
        ).json()["answer"]

        if "A" in rsp and "B" not in rsp:
            choice = "former"
        elif "B" in rsp and "A" not in rsp:
            choice = "latter"
        else:
            raise ValueError(f"Unexpected answer from DepictQA: {rsp}")

        return prompt, choice
