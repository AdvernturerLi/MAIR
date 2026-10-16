"""Frozen-source sequential MAIR batch runner with durable progress and resume.

Restoration runs in an isolated process per image. HQ is used only afterwards
for RGB PSNR/SSIM. Mismatched outputs are bicubic-resized in memory for
evaluation only; saved outputs and planning never use ground truth.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def dataset_path(root, relative):
    result = (root / relative).resolve()
    if root.resolve() not in result.parents:
        raise ValueError(f"Dataset path outside root: {relative}")
    return result


def validate_manifest(manifest, root, expected_count):
    with Path(manifest).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_count or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError(f"Expected {expected_count} unique images, found {len(rows)}")
    for row in rows:
        if row["split"] != "test":
            raise ValueError("This experiment must use the fixed test split")
        if Path(row["sample_id"]).name != row["sample_id"] or row["sample_id"] in (".", ".."):
            raise ValueError("Unsafe sample ID")
        for kind in ("lq", "hq"):
            path = dataset_path(root, row[f"{kind}_path"])
            if digest(path) != row[f"{kind}_sha256"]:
                raise ValueError(f"Hash mismatch: {path}")
    return rows


def freeze_source(source, target):
    target.mkdir()
    selected = []
    for path in source.iterdir():
        if path.is_file() and path.suffix in (".py", ".md"):
            selected.append(path)
    for folder in ("llm", "utils", "pipeline", "executor", "experience"):
        selected.extend(p for p in (source / folder).rglob("*") if p.is_file()
                        and p.suffix in (".py", ".yml", ".json")
                        and "tools" not in p.relative_to(source).parts)
    for path in selected:
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    for tools in (source / "executor").glob("*/tools"):
        if tools.exists():
            (target / tools.relative_to(source)).symlink_to(tools.resolve(), target_is_directory=True)
    if (source / ".venvs").exists():
        (target / ".venvs").symlink_to((source / ".venvs").resolve(), target_is_directory=True)
    hashes = {p.relative_to(source).as_posix(): digest(p) for p in sorted(selected)}
    atomic_json(target.parent / "source_manifest.json", hashes)
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def score_output(output, reference):
    import cv2
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    restored, hq = cv2.imread(str(output)), cv2.imread(str(reference))
    if restored is None or hq is None:
        raise ValueError("Unreadable output/reference")
    shapes = {"actual_size": list(restored.shape[:2]), "expected_size": list(hq.shape[:2])}
    resized = restored.shape != hq.shape
    if resized:
        restored = cv2.resize(restored, (hq.shape[1], hq.shape[0]), interpolation=cv2.INTER_CUBIC)
    restored, hq = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB), cv2.cvtColor(hq, cv2.COLOR_BGR2RGB)
    psnr = float(peak_signal_noise_ratio(hq, restored, data_range=255))
    return {**shapes, "status": "success", "metric_status": "ok_bicubic_to_reference" if resized else "ok",
            "metric_resized": resized,
            "psnr": psnr if math.isfinite(psnr) else "inf",
            "ssim": float(structural_similarity(hq, restored, channel_axis=2, data_range=255))}


def save_results(out, results):
    fields = ["sample_id", "status", "elapsed_s", "returncode", "metric_status", "psnr", "ssim",
                "actual_size", "expected_size", "metric_resized", "tool_calls", "tool_errors", "error", "output"]
    temporary = out / "per_sample_results.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    temporary.replace(out / "per_sample_results.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experience", choices=("none", "paper"), required=True)
    parser.add_argument("--expected-count", type=int, default=160)
    parser.add_argument("--sample-timeout", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pair-with", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    process = None

    def stop(signum, frame):
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise KeyboardInterrupt(f"Stopped by signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    out, data = args.output.resolve(), args.data_root.resolve()
    if out.exists() and not args.resume and any(
        p.name not in {"runner.log", "launch.json"} for p in out.iterdir()
    ):
        raise FileExistsError(f"Use a new experiment directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o700)
    start = time.monotonic()
    progress = {"state": "preflight", "pid": os.getpid(), "total": args.expected_count,
                "completed": 0, "succeeded": 0, "failed": 0, "invalid_output": 0,
                "experience": args.experience, "started_at": now(), "updated_at": now()}
    atomic_json(out / "progress.json", progress)
    (out / "process.pid").write_text(str(os.getpid()))
    try:
        rows = validate_manifest(args.manifest, data, args.expected_count)
        import yaml
        if not args.resume:
            source = args.source_root.resolve()
            sys.path.insert(0, str(source))
            from llm import GPT4
            from paper_resources import ALGORITHM_VERSION, inventory
            client = GPT4(config_path=args.config, silent=True)
            if urlparse(client.base_url).hostname != "api.deepseek.com":
                raise ValueError("Batch image export is authorized only for api.deepseek.com")
            config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
            config.update(OPENAI_API_KEY=client.api_key, OPENAI_BASE_URL=client.base_url, OPENAI_MODEL=client.model)
            frozen_config = out / "runtime_config.yml"
            frozen_config.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
            frozen_config.chmod(0o600)
            public = {k: v for k, v in config.items() if k != "OPENAI_API_KEY"}
            source_hash = freeze_source(source, out / "source_snapshot")
            protocol = {"algorithm_version": ALGORITHM_VERSION, "source_sha256": source_hash,
                        "manifest_sha256": digest(args.manifest), "split": "test", "count": len(rows),
                        "experience": args.experience, "effective_config": public,
                        "input_resize": "none", "output_resize": "none",
                        "metric_resize": "opencv_INTER_CUBIC_to_reference_if_needed_v1",
                        "metrics": ["RGB PSNR", "RGB SSIM"], "tool_inventory": inventory(),
                        "sample_timeout_s": args.sample_timeout, "created_at": now()}
            if args.pair_with:
                prior = json.loads((args.pair_with / "resolved_config.json").read_text())
                for field in ("source_sha256", "manifest_sha256", "effective_config", "input_resize", "output_resize", "metric_resize", "tool_inventory"):
                    if protocol[field] != prior[field]:
                        raise ValueError(f"Paired experiment configuration mismatch: {field}")
            atomic_json(out / "resolved_config.json", protocol)
            shutil.copy(args.manifest, out / "manifest.csv")
        else:
            protocol = json.loads((out / "resolved_config.json").read_text())
            if protocol["manifest_sha256"] != digest(args.manifest) or protocol["experience"] != args.experience:
                raise ValueError("Resume protocol does not match")
        snapshot = out / "source_snapshot"
        frozen_config = out / "runtime_config.yml"
        results = []
        journal = out / "results.jsonl"
        if journal.exists():
            results = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
        completed = {r["sample_id"] for r in results}
        (out / "runs").mkdir(exist_ok=True)
        environment = dict(os.environ)
        for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "PYTHONPATH"):
            environment.pop(key, None)
        maxim_python = snapshot / ".venvs/maxim-jax/bin/python"
        if maxim_python.exists():
            environment["MAIR_MAXIM_PYTHON"] = str(maxim_python.resolve().parent / "python") if not maxim_python.is_symlink() else str(maxim_python)
        for index, row in enumerate(rows, 1):
            if row["sample_id"] in completed:
                continue
            sample_id = row["sample_id"]
            sample = out / "runs" / sample_id
            sample.mkdir(exist_ok=True)
            attempt = sample / f"attempt_{len(list(sample.glob('attempt_*'))) + 1:02d}"
            print(f"[{now()}] START {index}/{len(rows)} {sample_id}", flush=True)
            item_start = time.monotonic()
            command = [sys.executable, "-u", str(snapshot / "main.py"), "--config", str(frozen_config),
                       "--input", str(dataset_path(data, row["lq_path"])), "--output", str(attempt), "--silent"]
            if args.experience == "none":
                command.append("--no-retrieval")
            with (sample / "run.log").open("ab") as log:
                process = subprocess.Popen(command, cwd=snapshot, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                timed_out = False
                while process.poll() is None:
                    elapsed = time.monotonic() - item_start
                    progress.update(state="running", current_index=index, current_sample=sample_id,
                                    current_elapsed_s=round(elapsed, 1), child_pid=process.pid,
                                    completed=len(results), succeeded=sum(r["status"] == "success" for r in results),
                                    invalid_output=sum(r["status"] == "invalid_output" for r in results),
                                    failed=sum(r["status"] not in ("success", "invalid_output") for r in results),
                                    elapsed_s=round(time.monotonic()-start, 1), updated_at=now())
                    atomic_json(out / "progress.json", progress)
                    if elapsed > args.sample_timeout:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        timed_out = True
                        break
                    time.sleep(5)
            record = {"sample_id": sample_id, "returncode": process.returncode,
                      "status": "timeout" if timed_out else "failed", "metric_status": "not_run"}
            trajectory = attempt / "trajectory.json"
            events = json.loads(trajectory.read_text()) if trajectory.exists() else []
            if args.experience == "none" and any(e["step"] == "experience_loaded" for e in events):
                raise RuntimeError("No-experience run unexpectedly loaded experience")
            output = attempt / "result.png"
            if process.returncode == 0 and output.exists():
                try:
                    record.update(score_output(output, dataset_path(data, row["hq_path"])))
                except Exception as error:
                    record.update(status="metric_failed", metric_status="failed", error=str(error))
                record["output"] = str(output)
            else:
                errors = [e for e in events if e["step"] in ("error", "initialization_error", "abort")]
                record["error"] = errors[-1].get("error", errors[-1].get("reason")) if errors else "See sample run.log"
            record.update(elapsed_s=round(time.monotonic()-item_start, 1),
                          tool_calls=sum(e["step"] == "tool_execution" for e in events),
                          tool_errors=sum(e["step"] == "tool_error" for e in events))
            results.append(record)
            with journal.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            save_results(out, results)
            print(f"[{now()}] DONE {len(results)}/{len(rows)} {sample_id} {record['status']} {record['elapsed_s']}s", flush=True)
        progress.update(state="completed", current_sample=None, child_pid=None, current_elapsed_s=0,
                        completed=len(results), succeeded=sum(r["status"] == "success" for r in results),
                        invalid_output=sum(r["status"] == "invalid_output" for r in results),
                        failed=sum(r["status"] not in ("success", "invalid_output") for r in results),
                        elapsed_s=round(time.monotonic()-start, 1), updated_at=now())
        atomic_json(out / "progress.json", progress)
        print(f"[{now()}] FINISHED {json.dumps(progress)}", flush=True)
    except BaseException as error:
        progress.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}", updated_at=now())
        atomic_json(out / "progress.json", progress)
        raise


if __name__ == "__main__":
    main()
