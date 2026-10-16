"""Bounded local service requests; comparison model lives only during comparison.

The original DepictQA comparison checkpoint and prompt are unchanged. Its GPU
process exits before another restoration tool can run.
"""
import ast
from contextlib import contextmanager
import os
from pathlib import Path
import socket
import subprocess
import time

import requests


def parse_response(text):
    import json
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return ast.literal_eval(text)


def post_local(url, payload, attempts=3):
    for attempt in range(attempts):
        try:
            response = requests.post(url, data=payload, timeout=(10, 180))
            response.raise_for_status()
            body = response.json()
            if not isinstance(body.get('answer'), str):
                raise ValueError('Local service returned no text answer')
            return body['answer']
        except (requests.RequestException, ValueError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)


@contextmanager
def comparison_service():
    if os.environ.get('AGENTICIR_COMPARISON_MODE') != 'on_demand_depictqa':
        yield
        return
    executable = Path(os.environ['AGENTICIR_DEPICTQA_PYTHON'])
    app = Path(os.environ['AGENTICIR_COMPARISON_APP'])
    config = Path(os.environ['AGENTICIR_COMPARISON_CONFIG'])
    cwd = Path(os.environ['AGENTICIR_DEPICTQA_ROOT'])
    logdir = Path(os.environ['AGENTICIR_SERVICE_LOG_DIR'])
    logdir.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', 5002)) == 0:
            raise RuntimeError('Unexpected comparison service already occupies port 5002')
    with (logdir / 'depictqa_comparison.log').open('ab') as log:
        process = subprocess.Popen([str(executable), '-u', str(app), '--cfg', str(config)],
                                   cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('DepictQA comparison failed to start; inspect service log')
                with socket.socket() as sock:
                    if sock.connect_ex(('127.0.0.1', 5002)) == 0:
                        break
                time.sleep(1)
            else:
                raise TimeoutError('DepictQA comparison startup exceeded 240 seconds')
            yield
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
