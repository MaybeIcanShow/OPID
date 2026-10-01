#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/opid/bin/python}"
TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-$PROJECT_ROOT/outputs/tensorboard/runs}"
TENSORBOARD_HOST="${TENSORBOARD_HOST:-127.0.0.1}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
export CUDA_VISIBLE_DEVICES=""
exec "$PYTHON_BIN" -m tensorboard.main \
  --logdir "$TENSORBOARD_LOGDIR" \
  --host "$TENSORBOARD_HOST" \
  --port "$TENSORBOARD_PORT" \
  --reload_interval 5 \
  --reload_multifile=true \
  --reload_multifile_inactive_secs=86400 \
  --load_fast=false \
  --window_title "OPID ToolBench GRPO" \
  "$@"
