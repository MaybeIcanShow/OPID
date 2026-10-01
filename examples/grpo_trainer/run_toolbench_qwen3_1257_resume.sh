#!/usr/bin/env bash
# Resume the original-weight experiment on GPUs 1,2,5,7 in a separate run directory.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,5,7}"
export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export MODEL_PATH="${MODEL_PATH:-$HOME/model/Qwen3-1.7B}"
export OUTPUT_DIR="${OUTPUT_DIR:-$HOME/model/ckpt/grpo_qwen3_1.7b_toolbench_2048_gpu1257_$RUN_TAG}"
export TENSORBOARD_DIR="$OUTPUT_DIR/tensorboard"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-$HOME/model/ckpt/grpo_qwen3_1.7b_toolbench_2048_fresh_20261001_022739/global_step_50}"
if [[ ! -f "$RESUME_CHECKPOINT/data.pt" || ! -d "$RESUME_CHECKPOINT/actor" ]]; then
  echo "Incomplete resume checkpoint: $RESUME_CHECKPOINT" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "GPU migration requires a new output directory: $OUTPUT_DIR" >&2
  exit 1
fi
exec bash examples/grpo_trainer/run_toolbench_qwen3_4gpu.sh \
  "trainer.experiment_name=grpo_qwen3_1.7b_toolbench_2048_gpu1257_$RUN_TAG" \
  "$@" \
  trainer.resume_mode=resume_path \
  "trainer.resume_from_path=$RESUME_CHECKPOINT" \
  trainer.del_local_ckpt_after_load=False
