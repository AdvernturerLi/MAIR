import json

from llm.gpt4 import GPT4
from llm.depictqa import DepictQA


def test_low_res_no_blur_proxy_nonempty(tmp_path):
    # LOW_RES_AS_BLUR=false 时不再返回空列表导致下游 IndexError
    d = DepictQA(low_res_as_blur=False)
    _, rsp = d.eval_degradation(tmp_path / "x.png", "low resolution")
    assert json.loads(rsp) == [["low resolution", "medium"]]


def test_env_var_overrides_config_key(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yml"
    cfg.write_text(
        "OPENAI_API_KEY: file-key\n"
        "OPENAI_BASE_URL: https://file.example.com\n"
        "OPENAI_MODEL: file-model\n"
        "MAX_TOKENS: 100\n"
        "TEMPERATURE: 0.0\n"
    )
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    g = GPT4(config_path=cfg)
    assert g.api_key == "env-key"          # 环境变量优先
    assert g.model == "file-model"          # 未设置 env 时回退 config


def test_safe_parse_json_plain():
    assert GPT4._safe_parse_json('{"a": 1}') == {"a": 1}


def test_safe_parse_json_codeblock():
    assert GPT4._safe_parse_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}


def test_safe_parse_json_rejects_code_execution():
    # 替换 eval 后，恶意/非 JSON 输入应返回 None，而不是被执行
    assert GPT4._safe_parse_json('__import__("os").system("id")') is None
    assert GPT4._safe_parse_json("not valid json") is None
    assert GPT4._safe_parse_json("[1, 2, 3]") == [1, 2, 3]
