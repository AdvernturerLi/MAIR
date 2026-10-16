from unittest.mock import Mock
import pytest

from llm.gpt4 import GPT4


def client(responses):
    obj = GPT4.__new__(GPT4)
    obj.max_tokens = 8192
    obj.prompt_tokens = obj.completion_tokens = 0
    obj._log = Mock()
    obj._prepare_for_request = Mock(side_effect=lambda *a: ({}, {'max_tokens':8192, 'model':'unchanged'}))
    obj._send_request = Mock(side_effect=responses)
    return obj


def response(reason, content):
    obj = Mock()
    obj.json.return_value = {'usage': {'prompt_tokens':3,'completion_tokens':8192},
                             'choices':[{'finish_reason':reason,'message':{'content':content}}]}
    return obj


def test_truncation_increases_budget_without_changing_prompt_or_model():
    obj = client([response('length',''), response('stop','["denoising"]')])
    prompt, output = obj.query(prompt='original request', format_check=lambda value: None)
    assert output == '["denoising"]'
    calls = obj._send_request.call_args_list
    assert [c.args[1]['max_tokens'] for c in calls] == [8192,16384]
    assert all(c.args[1]['model']=='unchanged' for c in calls)
    assert all(c.args[0]=='original request' for c in obj._prepare_for_request.call_args_list)


def test_repeated_truncation_is_bounded_and_not_accepted_as_success():
    obj = client([response('length','') for _ in range(3)])
    with pytest.raises(RuntimeError, match='remains truncated'):
        obj.query(prompt='original')
    assert obj._send_request.call_count == 3
    assert [c.args[1]['max_tokens'] for c in obj._send_request.call_args_list] == [8192,16384,32768]
