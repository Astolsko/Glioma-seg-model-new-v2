"""Duration/ETA formatting + a running step-time estimator for the training
CLI. Kept free of torch/monai imports so it's cheap to unit test.
"""
from datetime import datetime, timedelta


def format_duration(seconds):
    if seconds is None:
        return "estimating..."
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if minutes or hours or days:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def format_eta_clock(seconds):
    if seconds is None:
        return ""
    finish = datetime.now() + timedelta(seconds=max(0, seconds))
    return finish.strftime("%b %d, %H:%M")


class RunTimeEstimator:
    """Projects the wall-clock time left for the *whole* run.

    Training and validation steps cost different amounts of time (validation
    runs sliding-window inference), so their per-step average is tracked
    separately. The estimate is available from the first recorded step and
    keeps refining as more steps complete — early on it will be rough, but
    after a few hundred steps the running average settles down. tqdm already
    shows a per-epoch ETA on its own; this covers the total-run estimate that
    tqdm can't see across epoch boundaries.
    """

    def __init__(self, total_train_steps, total_val_steps):
        self.total_train_steps = total_train_steps
        self.total_val_steps = total_val_steps
        self._train_steps_done = 0
        self._train_time_sum = 0.0
        self._val_steps_done = 0
        self._val_time_sum = 0.0

    def record_train_step(self, dt):
        self._train_steps_done += 1
        self._train_time_sum += dt

    def record_val_step(self, dt):
        self._val_steps_done += 1
        self._val_time_sum += dt

    @property
    def avg_train_step_time(self):
        return self._train_time_sum / self._train_steps_done if self._train_steps_done else None

    @property
    def avg_val_step_time(self):
        return self._val_time_sum / self._val_steps_done if self._val_steps_done else None

    def remaining_seconds(self):
        avg_train = self.avg_train_step_time
        if avg_train is None:
            return None

        remaining_train = max(self.total_train_steps - self._train_steps_done, 0)
        eta = remaining_train * avg_train

        remaining_val = max(self.total_val_steps - self._val_steps_done, 0)
        avg_val = self.avg_val_step_time if self.avg_val_step_time is not None else avg_train
        eta += remaining_val * avg_val
        return eta

    def eta_string(self):
        seconds = self.remaining_seconds()
        if seconds is None:
            return "estimating..."
        return f"{format_duration(seconds)} remaining (finish ~{format_eta_clock(seconds)})"
