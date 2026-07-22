"""Tests for utils/progress.py — the ETA estimator train.py's CLI prints
every epoch. No torch/monai involved, so these run everywhere."""
import re

from utils.progress import RunTimeEstimator, format_duration, format_eta_clock


def test_format_duration_seconds_only():
    assert format_duration(45) == "45s"


def test_format_duration_includes_minutes_when_present():
    assert format_duration(125) == "2m 5s"


def test_format_duration_includes_hours_and_minutes():
    assert format_duration(3725) == "1h 2m 5s"


def test_format_duration_includes_days():
    assert format_duration(90000) == "1d 1h 0m 0s"


def test_format_duration_none_means_not_estimated_yet():
    assert format_duration(None) == "estimating..."


def test_format_duration_clamps_negative_to_zero():
    assert format_duration(-5) == "0s"


def test_format_eta_clock_returns_formatted_timestamp():
    result = format_eta_clock(60)
    assert re.match(r"^[A-Z][a-z]{2} \d{2}, \d{2}:\d{2}$", result)


def test_format_eta_clock_none_returns_empty_string():
    assert format_eta_clock(None) == ""


def test_estimator_reports_estimating_before_any_train_step_recorded():
    estimator = RunTimeEstimator(total_train_steps=100, total_val_steps=10)
    assert estimator.remaining_seconds() is None
    assert estimator.eta_string() == "estimating..."


def test_estimator_projects_remaining_time_from_train_steps_only():
    estimator = RunTimeEstimator(total_train_steps=10, total_val_steps=0)
    for _ in range(4):
        estimator.record_train_step(2.0)  # avg = 2.0s/step

    # 6 train steps remain, 0 val steps -> 6 * 2.0 = 12.0s
    assert estimator.remaining_seconds() == 12.0


def test_estimator_uses_train_average_as_val_fallback_before_any_val_step():
    """Before a single validation step has run, we don't know its real
    per-step cost yet — falling back to the train-step average keeps the
    ETA in the right ballpark instead of silently ignoring pending val work."""
    estimator = RunTimeEstimator(total_train_steps=10, total_val_steps=5)
    for _ in range(10):
        estimator.record_train_step(1.0)

    # all train steps done (0 remaining); 5 val steps remaining, no val
    # average recorded yet -> falls back to the 1.0s/step train average
    assert estimator.remaining_seconds() == 5.0


def test_estimator_uses_real_val_average_once_recorded():
    estimator = RunTimeEstimator(total_train_steps=10, total_val_steps=4)
    for _ in range(10):
        estimator.record_train_step(1.0)
    for _ in range(2):
        estimator.record_val_step(3.0)  # avg = 3.0s/step, distinct from train avg

    # 0 train steps remaining; 2 val steps remaining * 3.0s avg = 6.0
    assert estimator.remaining_seconds() == 6.0


def test_estimator_eta_string_mentions_remaining_and_finish_time():
    estimator = RunTimeEstimator(total_train_steps=10, total_val_steps=0)
    estimator.record_train_step(5.0)
    text = estimator.eta_string()
    assert "remaining" in text
    assert "finish" in text
