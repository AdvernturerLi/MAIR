from pathlib import Path

import pytest

from main import _is_within, _require_empty_output


def test_is_within(tmp_path):
    assert _is_within(tmp_path / "a" / "b", tmp_path)
    assert _is_within(tmp_path, tmp_path)
    assert not _is_within(tmp_path, tmp_path / "a")


def test_force_refuses_project_root(tmp_path):
    import main
    project_root = Path(main.__file__).resolve().parent
    with pytest.raises(SystemExit):
        _require_empty_output(project_root, tmp_path / "in.png", force=True)


def test_force_refuses_ancestor_of_project(tmp_path):
    import main
    project_root = Path(main.__file__).resolve().parent
    with pytest.raises(SystemExit):
        _require_empty_output(project_root.parent, project_root / "in.png", force=True)


def test_force_refuses_input_dir(tmp_path):
    img = tmp_path / "in.png"
    img.write_bytes(b"x")
    with pytest.raises(SystemExit):
        _require_empty_output(tmp_path, img, force=True)


def test_force_deletes_plain_dir(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "x.txt").write_text("x")
    img = tmp_path / "in.png"
    img.write_bytes(b"x")
    _require_empty_output(out, img, force=True)
    assert not out.exists()


def test_require_empty_output_no_force_fails_on_nonempty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "x.txt").write_text("x")
    with pytest.raises(SystemExit):
        _require_empty_output(out, tmp_path / "in.png", force=False)
