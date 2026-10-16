from pathlib import Path
from unittest.mock import Mock
import pytest

from pipeline.iragent import IRAgent


class FakeTool:
    def __init__(self, name, fails):
        self.tool_name, self.fails = name, fails

    def __call__(self, input_dir, output_dir, silent):
        if self.fails:
            (output_dir / 'partial.txt').write_text('diagnostic')
            raise RuntimeError('simulated tool failure')
        (output_dir / 'output.png').write_bytes(b'completed')


def make_agent(tmp_path, tools):
    agent = IRAgent.__new__(IRAgent)
    agent.plan = ['denoising']
    agent.work_mem = {'n_invocations': 0}
    agent.log_dir = tmp_path / 'logs'
    agent.log_dir.mkdir()
    task = tmp_path / 'img_tree/subtask1-denoising'
    task.mkdir(parents=True)
    root = {'img_path': str(tmp_path / 'input.png'), 'children': {
        'denoising': {'tools': {}, 'best_tool': None}}}
    agent.cur_node = root
    agent._prepare_for_subtask = lambda subtask: (task, 'noise', tools)
    agent.with_reflection = agent.with_rollback = True
    agent.evaluate_tool_result = lambda *args: 'very low'
    def record(path, level):
        name = path.parents[1].name.removeprefix('tool-')
        root['children']['denoising']['tools'][name] = {'img_path': str(path), 'children': {}}
    agent._record_tool_res = record
    agent._dump_summary = Mock()
    agent._render_img_tree = Mock()
    agent._img_nickname = str
    agent.workflow_logger = Mock()
    return agent, task


def test_failed_tool_keeps_diagnostics_and_next_tool_can_succeed(tmp_path):
    agent, task = make_agent(tmp_path, [FakeTool('bad', True), FakeTool('good', False)])
    assert agent.execute_subtask(None) is True
    assert agent.work_mem['n_invocations'] == 2
    assert len(agent.work_mem['tool_errors']) == 1
    assert not (task / 'tool-bad').exists()
    assert (agent.log_dir / 'failed_tools/0001_bad/0-img/partial.txt').read_text() == 'diagnostic'
    assert Path(agent.cur_node['img_path']).read_bytes() == b'completed'


def test_all_tools_failing_is_explicit_and_never_marked_success(tmp_path):
    agent, task = make_agent(tmp_path, [FakeTool('bad', True), FakeTool('bad2', True)])
    with pytest.raises(RuntimeError, match='All candidate tools failed'):
        agent.execute_subtask(None)
    assert len(agent.work_mem['tool_errors']) == 2
