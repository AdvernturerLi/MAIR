"""Fixed tool universe and explicit runtime profile for paired experiments."""
import json
from pathlib import Path
from executor import executor

ALGORITHM_VERSION = 'agenticir-noexp-aligned32-bounded-runtime-v1'


def inventory():
    profile = json.loads((Path(__file__).resolve().parent / 'tool_alignment.json').read_text())
    actual = {s: [t.tool_name for t in ts] for s, ts in executor.toolbox_router.items()}
    assert actual == profile['tools'], 'Tool universe changed'
    rows = []
    for subtask, tools in executor.toolbox_router.items():
        for tool in tools:
            if not tool.available():
                raise RuntimeError(f'Missing runtime resource: {subtask}/{tool.tool_name}')
            rows.append({'subtask': subtask, 'tool': tool.tool_name, 'available': True})
    assert len(rows) == 32
    return rows
