import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from run_manifest import digest, freeze_source, score_output, validate_manifest


def test_source_snapshot_excludes_credentials(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("print('runner')")
    (source / "config.alternative.runtime.yml").write_text("OPENAI_API_KEY: do-not-copy")
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    fingerprint = freeze_source(source, experiment / "source_snapshot")
    assert len(fingerprint) == 64
    assert not (experiment / "source_snapshot/config.alternative.runtime.yml").exists()
    assert "do-not-copy" not in (experiment / "source_manifest.json").read_text()


def test_manifest_detects_changes_and_wrong_split(tmp_path):
    image = tmp_path / "input.png"
    image.write_bytes(b"dataset bytes")
    row = dict(sample_id="019__rain+haze", split="test", hq_path=image.name,
               lq_path=image.name, hq_sha256=digest(image), lq_sha256=digest(image))
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    assert len(validate_manifest(manifest, tmp_path, 1)) == 1
    image.write_bytes(b"modified")
    with pytest.raises(ValueError, match="Hash mismatch"):
        validate_manifest(manifest, tmp_path, 1)


def test_metrics_resize_for_scoring_without_changing_saved_output(tmp_path):
    output, hq = tmp_path / "output.png", tmp_path / "hq.png"
    cv2.imwrite(str(output), np.zeros((16,16,3), dtype=np.uint8))
    cv2.imwrite(str(hq), np.zeros((32,32,3), dtype=np.uint8))
    score = score_output(output, hq)
    assert score["status"] == "success"
    assert score["metric_status"] == "ok_bicubic_to_reference"
    assert score["metric_resized"] is True
    assert score["psnr"] == "inf"
    assert score["ssim"] == 1.0
    assert cv2.imread(str(output)).shape == (16,16,3)
