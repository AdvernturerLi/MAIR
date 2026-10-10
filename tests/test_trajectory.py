import hashlib
import json
from trajectory import Trajectory


def test_trajectory_records_and_saves(tmp_path):
    img = tmp_path / "in.png"
    img.write_bytes(b"hello")
    t = Trajectory(tmp_path / "out", input_path=img)
    t.record("coarse_perception", evaluation=[["rain", "medium"]])
    t.record("plan", plan=["deraining"])
    p = t.save()

    assert p.exists()
    data = json.loads(p.read_text())
    assert data[0]["step"] == "coarse_perception"
    assert data[0]["evaluation"] == [["rain", "medium"]]
    assert data[1]["step"] == "plan"
    assert t.input_hash == hashlib.md5(b"hello").hexdigest()
