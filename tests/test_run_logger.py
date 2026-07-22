"""Tests for utils/run_logger.py — the on-disk footprint of a run (config
snapshot, metrics.csv, checkpoints, logs). A crash here (or a silent
CSV field mismatch) loses whatever training already happened, since it fires
after each epoch completes.
"""
import csv
import json
import os
import sys

import pytest

from utils.run_logger import RunLogger, _sanitize_run_name, _unique_run_dir


def test_sanitize_run_name_replaces_path_separators():
    assert _sanitize_run_name("foo/bar\\baz") == "foo_bar_baz"


def test_sanitize_run_name_strips_whitespace():
    assert _sanitize_run_name("  my run  ") == "my run"


def test_unique_run_dir_returns_candidate_when_free(tmp_path):
    result = _unique_run_dir(str(tmp_path), "fresh_run")
    assert result == os.path.join(str(tmp_path), "fresh_run")


def test_unique_run_dir_avoids_collision(tmp_path):
    os.makedirs(os.path.join(str(tmp_path), "run1"))
    result = _unique_run_dir(str(tmp_path), "run1")
    assert result == os.path.join(str(tmp_path), "run1_1")


def test_unique_run_dir_avoids_collision_across_multiple_existing(tmp_path):
    os.makedirs(os.path.join(str(tmp_path), "run1"))
    os.makedirs(os.path.join(str(tmp_path), "run1_1"))
    result = _unique_run_dir(str(tmp_path), "run1")
    assert result == os.path.join(str(tmp_path), "run1_2")


def test_run_logger_creates_expected_folder_structure(tmp_path):
    with RunLogger(run_name="test_run", base_dir=str(tmp_path)) as logger:
        for d in (logger.run_dir, logger.plots_dir, logger.vis_dir, logger.attention_dir,
                  logger.checkpoint_dir, logger.testing_dir, logger.testing_vis_dir):
            assert os.path.isdir(d)
    assert os.path.exists(logger.log_path)


def test_run_logger_tees_stdout_to_log_file_and_console(tmp_path, capsys):
    with RunLogger(run_name="tee_test", base_dir=str(tmp_path)) as logger:
        print("hello from test")

    with open(logger.log_path) as f:
        file_content = f.read()
    assert "hello from test" in file_content
    assert "hello from test" in capsys.readouterr().out


def test_run_logger_restores_stdout_and_stderr_after_exit(tmp_path):
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with RunLogger(run_name="restore_test", base_dir=str(tmp_path)):
        assert sys.stdout is not original_stdout
    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr


def test_run_logger_restores_stdout_even_if_exception_raised(tmp_path):
    original_stdout = sys.stdout
    with pytest.raises(ValueError):
        with RunLogger(run_name="exc_test", base_dir=str(tmp_path)):
            raise ValueError("boom")
    assert sys.stdout is original_stdout


def test_save_config_snapshot_writes_json(tmp_path):
    pytest.importorskip("easydict")
    from easydict import EasyDict
    cfg = EasyDict()
    cfg.epoch = 10
    cfg.nested = EasyDict()
    cfg.nested.value = "x"

    with RunLogger(run_name="cfg_test", base_dir=str(tmp_path)) as logger:
        logger.save_config_snapshot(cfg)

    with open(logger.config_snapshot_path) as f:
        data = json.load(f)
    assert data["epoch"] == 10
    assert data["nested"]["value"] == "x"


def test_log_epoch_metrics_writes_rows_in_order(tmp_path):
    with RunLogger(run_name="metrics_test", base_dir=str(tmp_path)) as logger:
        logger.open_metrics_csv(["epoch", "loss"])
        logger.log_epoch_metrics({"epoch": 1, "loss": 0.5})
        logger.log_epoch_metrics({"epoch": 2, "loss": 0.3})

    with open(logger.metrics_csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["epoch"] for r in rows] == ["1", "2"]


def test_log_epoch_metrics_raises_if_row_has_unexpected_key(tmp_path):
    """Regression guard for utils/engine.py's METRICS_CSV_FIELDS: if
    validate()'s returned dict ever grows a new metric key that isn't also
    added to METRICS_CSV_FIELDS, csv.DictWriter raises ValueError — and that
    happens mid-training, after an epoch's worth of GPU time, not at
    startup. See test_engine.py::test_validate_return_keys_are_subset_of_metrics_csv_fields
    for the check that would actually catch this in practice."""
    with RunLogger(run_name="mismatch_test", base_dir=str(tmp_path)) as logger:
        logger.open_metrics_csv(["epoch", "loss"])
        with pytest.raises(ValueError):
            logger.log_epoch_metrics({"epoch": 1, "loss": 0.5, "unexpected_new_metric": 1.0})


def test_log_epoch_metrics_tolerates_missing_keys(tmp_path):
    # e.g. an epoch where cfg.val_interval skipped validation — row only has
    # the base training fields, no val metrics. DictWriter should just
    # write blanks for the missing fieldnames, not raise.
    with RunLogger(run_name="partial_row_test", base_dir=str(tmp_path)) as logger:
        logger.open_metrics_csv(["epoch", "loss", "val_loss"])
        logger.log_epoch_metrics({"epoch": 1, "loss": 0.5})

    with open(logger.metrics_csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["val_loss"] == ""


def test_write_test_metrics_writes_single_row_csv(tmp_path):
    with RunLogger(run_name="test_metrics_test", base_dir=str(tmp_path)) as logger:
        logger.write_test_metrics({"dice": 0.8, "iou": 0.6})

    with open(logger.test_metrics_csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"dice": "0.8", "iou": "0.6"}]


def test_run_logger_falls_back_to_timestamp_name_when_sanitized_empty(tmp_path):
    with RunLogger(run_name="   ", base_dir=str(tmp_path)) as logger:  # strips to "" -> fallback
        assert os.path.basename(logger.run_dir).startswith("run_")
