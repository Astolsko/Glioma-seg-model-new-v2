#!/usr/bin/env bash
#
# Restart supervisor for train.py.
#
#   tools/train_supervisor.sh <run-name> [max-restarts] [-- extra train.py args]
#
# Launches `python train.py --name <run-name> --auto-resume` and relaunches it
# whenever it exits non-zero, up to <max-restarts> times (default 20). Because
# --auto-resume continues from logs/<run>/checkpoints/last.pth, each restart
# picks up at the epoch after the last completed one — the same command works
# for the first launch and every restart after it.
#
# In-process OOM recovery (utils/engine.py) already absorbs the occasional
# unlucky batch. This layer is for what a process cannot recover from in-flight:
# a hard OOM past the skip limit, a CUDA context that has gone unusable, a
# killed process, a driver hiccup.
#
# Exit code 0 (training finished, or it stopped early on patience) ends the
# loop. So does SIGINT — a Ctrl-C means you wanted it to stop, and a supervisor
# that restarts through that is a supervisor you have to fight.
#
# Run under nohup/tmux for a real 45h run:
#   nohup tools/train_supervisor.sh v2-run3 > logs/v2-run3-supervisor.log 2>&1 &
#
set -uo pipefail

BACKOFF_SECONDS="${BACKOFF_SECONDS:-60}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_NAME="${1:-}"
if [[ -z "$RUN_NAME" || "$RUN_NAME" == "--" ]]; then
    echo "usage: $0 <run-name> [max-restarts] [-- extra train.py args]" >&2
    exit 2
fi
shift

# max-restarts is optional and positional, so only consume $1 when it actually
# looks like a count — otherwise `supervisor.sh run -- --flag` silently takes
# "--" as the restart budget and every arithmetic comparison after it fails.
MAX_RESTARTS=20
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    MAX_RESTARTS="$1"
    shift
fi
[[ "${1:-}" == "--" ]] && shift
EXTRA_ARGS=("$@")

cd "$(dirname "$0")/.." || exit 1

attempt=0
while true; do
    attempt=$((attempt + 1))
    echo "=============================================================="
    echo "[supervisor] attempt $attempt/$((MAX_RESTARTS + 1)) for run '$RUN_NAME' at $(date -Is)"
    echo "=============================================================="

    "$PYTHON_BIN" train.py --name "$RUN_NAME" --auto-resume "${EXTRA_ARGS[@]}"
    status=$?

    if [[ $status -eq 0 ]]; then
        echo "[supervisor] train.py finished cleanly at $(date -Is)"
        exit 0
    fi

    # 130 = SIGINT (Ctrl-C), 143 = SIGTERM. Both mean a human or the OS asked
    # for this to stop; restarting through them defeats the point.
    if [[ $status -eq 130 || $status -eq 143 ]]; then
        echo "[supervisor] interrupted (exit $status) — not restarting"
        exit $status
    fi

    if [[ $attempt -gt $MAX_RESTARTS ]]; then
        echo "[supervisor] exit $status and restart budget ($MAX_RESTARTS) is spent — giving up." >&2
        echo "[supervisor] see logs/$RUN_NAME/log.txt for the traceback." >&2
        exit $status
    fi

    echo "[supervisor] train.py exited $status — restarting in ${BACKOFF_SECONDS}s."
    echo "[supervisor] the GPU needs a moment to release the dead process's memory."
    sleep "$BACKOFF_SECONDS"
done
