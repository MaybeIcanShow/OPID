#!/usr/bin/env bash
# Always start a separate run from the original Qwen3 weights.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export MODEL_PATH="${MODEL_PATH:-$HOME/model/Qwen3-1.7B}"
export OUTPUT_DIR="${OUTPUT_DIR:-$HOME/model/ckpt/grpo_qwen3_1.7b_toolbench_2048_fresh_$RUN_TAG}"
export TENSORBOARD_DIR="$OUTPUT_DIR/tensorboard"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Fresh training requires a new output directory: $OUTPUT_DIR" >&2
  exit 1
fi
exec bash examples/grpo_trainer/run_toolbench_qwen3_4gpu.sh \
  "trainer.experiment_name=grpo_qwen3_1.7b_toolbench_2048_fresh_$RUN_TAG" \
  "$@" \
  trainer.resume_mode=disable \
  trainer.resume_from_path=null
