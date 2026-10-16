"""Retry whole samples in fresh processes, retaining every attempt and error."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for number in range(1, 4):
        trial = args.output / f'trial_{number:02d}'
        code = subprocess.call([sys.executable, '-u', str(Path(__file__).with_name('run_single.py')),
                                '--output', str(trial), *remaining])
        records.append({'trial': number, 'returncode': code})
        (args.output / 'retry_history.json').write_text(json.dumps(records, indent=2))
        if code == 0 and (trial / 'result.png').exists():
            for name in ('result.png', 'trajectory.json', 'agent_summary.json'):
                shutil.copy2(trial / name, args.output / name)
            return
        if (trial / 'trajectory.json').exists():
            shutil.copy2(trial / 'trajectory.json', args.output / 'trajectory.json')
        if number < 3:
            print(f'Retrying sample in a fresh process: attempt {number + 1}/3', flush=True)
            time.sleep(10 * number)
    raise SystemExit(1)


if __name__ == '__main__':
    main()
