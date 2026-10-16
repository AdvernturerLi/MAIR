"""One isolated AgenticIR sample. Ground truth is never exposed to the agent."""
import argparse
import json
import os
from pathlib import Path
import shutil
import traceback

from pipeline.iragent import IRAgent
from runtime_profile import inventory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--no-retrieval', action='store_true', required=True)
    parser.add_argument('--silent', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['AGENTICIR_SERVICE_LOG_DIR'] = str(args.output / 'service_logs')
    events = [{'step': 'configuration', 'experience': False, 'tool_count': len(inventory()),
               'evaluate_degradation_by': 'depictqa', 'reflect_by': 'depictqa',
               'compare_by': 'depictqa_on_demand', 'with_rollback': True}]
    try:
        agent = IRAgent(input_path=args.input.resolve(), output_dir=args.output / 'agent',
                        llm_config_path=args.config, evaluate_degradation_by='depictqa',
                        with_retrieval=False, schedule_experience_path=None,
                        with_reflection=True, reflect_by='depictqa', with_rollback=True,
                        silent=args.silent)
        assert agent.with_retrieval is False
        agent.run()
        shutil.copy2(agent.res_path, args.output / 'result.png')
        shutil.copy2(agent.work_mem_path, args.output / 'agent_summary.json')
        errors = agent.work_mem.get('tool_errors', [])
        events.extend({'step': 'tool_execution'} for _ in range(agent.work_mem['n_invocations']))
        events.extend({'step': 'tool_error', **e} for e in errors)
        events.append({'step': 'finished', 'execution_path': agent.work_mem['execution_path'],
                       'prompt_tokens': agent.gpt4.prompt_tokens, 'completion_tokens': agent.gpt4.completion_tokens})
    except BaseException as error:
        events.append({'step': 'error', 'error': f'{type(error).__name__}: {error}'})
        traceback.print_exc()
        raise
    finally:
        (args.output / 'trajectory.json').write_text(json.dumps(events, indent=2))


if __name__ == '__main__':
    main()
