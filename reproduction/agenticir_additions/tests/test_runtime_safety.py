from unittest.mock import Mock, patch
import pytest

from runtime_safety import parse_response, post_local


def test_response_parsing_supports_json_and_depictqa_tuples():
    assert parse_response('{"enabled":false,"value":null}') == {'enabled': False, 'value': None}
    assert parse_response("[('rain', 'high')]") == [('rain', 'high')]
    with pytest.raises((ValueError, SyntaxError)):
        parse_response("__import__('os').system('echo forbidden')")


def test_local_service_retry_and_timeout():
    import requests
    response = Mock()
    response.json.return_value = {'answer': 'low'}
    with patch('runtime_safety.requests.post', side_effect=[requests.Timeout(), response]) as post, patch('runtime_safety.time.sleep'):
        assert post_local('http://127.0.0.1:5001/evaluate_degradation', {}) == 'low'
    assert post.call_count == 2
    assert post.call_args.kwargs['timeout'] == (10, 180)


def test_local_service_stops_after_bounded_retries():
    import requests
    with patch('runtime_safety.requests.post', side_effect=requests.Timeout()) as post, patch('runtime_safety.time.sleep'):
        with pytest.raises(requests.Timeout):
            post_local('http://127.0.0.1:5001/evaluate_degradation', {})
    assert post.call_count == 3
