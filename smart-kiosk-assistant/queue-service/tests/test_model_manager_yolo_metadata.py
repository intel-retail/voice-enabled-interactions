# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DL Streamer YOLO ``metadata.yaml`` shim.

DL Streamer 2026.3 refuses to initialise *any* inference element for a model it
classifies as YOLO -- including ``gvainference``, which does no post-processing
-- unless ``metadata.yaml`` sits next to the IR. Verified against the stock
images with an otherwise identical pipeline:

    intel/dlstreamer:2026.2.0-ubuntu24      -> prerolls
    intel/dlstreamer:2026.3.0-ubuntu24-rc1  -> "requires .../metadata.yaml"

queue-service runs ``postproc: decoder`` (gvainference + src/yolo_decoder.py),
so nothing in DL Streamer ever consumes this metadata; it only has to satisfy
the gate. These tests pin that the file is written for YOLO models, carries the
labels the model-proc already declares, and is never written for the Intel
person detector.
"""
import sys

import pytest
import yaml

import model_manager


def _write_ir(tmp_path, name):
    """Create a placeholder IR file and return its path."""
    ir = tmp_path / f"{name}.xml"
    ir.write_text("<net/>", encoding="utf-8")
    return ir


def test_metadata_written_for_yolo_ir(tmp_path):
    ir = _write_ir(tmp_path, "yolo26n_static")

    assert model_manager.ensure_yolo_metadata(str(ir)) == str(ir)

    written = tmp_path / "metadata.yaml"
    assert written.is_file()
    meta = yaml.safe_load(written.read_text(encoding="utf-8"))
    # DL Streamer parses exactly these two to pick a version and task.
    assert meta["task"] == "detect"
    assert meta["version"]


def test_no_metadata_for_non_yolo_ir(tmp_path):
    """person-detection-retail-0013 goes through gvadetect and is unaffected."""
    ir = _write_ir(tmp_path, "person-detection-retail-0013")

    model_manager.ensure_yolo_metadata(str(ir))

    assert not (tmp_path / "metadata.yaml").exists()


def test_labels_come_from_model_proc_of_source_ir(tmp_path):
    """A ``*_static.xml`` has no model-proc; fall back to the source IR's.

    Guards the filename derivation -- getting it wrong silently degrades the
    label map to the person-only default instead of failing loudly.
    """
    _write_ir(tmp_path, "yolo26n_static")
    (tmp_path / "yolo26n.json").write_text(
        '{"output_postproc": [{"labels": ["person", "bicycle", "car"]}]}',
        encoding="utf-8",
    )

    model_manager.ensure_yolo_metadata(str(tmp_path / "yolo26n_static.xml"))

    meta = yaml.safe_load((tmp_path / "metadata.yaml").read_text(encoding="utf-8"))
    assert meta["names"] == {0: "person", 1: "bicycle", 2: "car"}


def test_existing_metadata_is_not_overwritten(tmp_path):
    """A hand-authored metadata.yaml must survive re-provisioning."""
    ir = _write_ir(tmp_path, "yolo26n_static")
    existing = tmp_path / "metadata.yaml"
    existing.write_text("task: detect\nnames:\n  0: staff\n", encoding="utf-8")

    model_manager.ensure_yolo_metadata(str(ir))

    assert yaml.safe_load(existing.read_text(encoding="utf-8"))["names"] == {0: "staff"}


def test_write_failure_aborts_startup(tmp_path, monkeypatch):
    """A read-only model dir must fail loudly, not serve a blind pipeline.

    Returning normally here would let main() start the HTTP API while the
    pipeline fails to initialise, so /health would pass while nothing is
    detected -- exactly the blind spot that hid this regression.
    """
    ir = _write_ir(tmp_path, "yolo26n_static")

    def _boom(*args, **kwargs):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(model_manager.tempfile, "mkstemp", _boom)

    with pytest.raises(RuntimeError, match="metadata"):
        model_manager.ensure_yolo_metadata(str(ir))


def test_concurrent_creation_keeps_the_winner(tmp_path, monkeypatch):
    """Two instances racing on a shared model dir: first writer wins.

    The publish must not clobber the other file, and must not raise -- the
    metadata is present, which is all the caller needs.
    """
    ir = _write_ir(tmp_path, "yolo26n_static")
    rival = tmp_path / "metadata.yaml"
    real_mkstemp = model_manager.tempfile.mkstemp

    def _mkstemp_then_race(*args, **kwargs):
        # Appears after the is_file() check, before our publish.
        rival.write_text("task: detect\nnames:\n  0: rival\n", encoding="utf-8")
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(model_manager.tempfile, "mkstemp", _mkstemp_then_race)

    assert model_manager.ensure_yolo_metadata(str(ir)) == str(ir)

    assert yaml.safe_load(rival.read_text(encoding="utf-8"))["names"] == {0: "rival"}


def test_no_staging_files_left_behind(tmp_path):
    """The staged temp file must not linger next to the IR."""
    ir = _write_ir(tmp_path, "yolo26n_static")

    model_manager.ensure_yolo_metadata(str(ir))

    assert not [p.name for p in tmp_path.iterdir() if p.name.startswith(".metadata-")]
