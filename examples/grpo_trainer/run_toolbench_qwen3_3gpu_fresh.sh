#!/usr/bin/env bash
# Fresh StableToolBench run on GPUs 1,2,5; batch sizes are divisible by three.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,5}"
export N_GPUS_PER_NODE=3
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-12}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-12}"
export GROUP_SIZE="${GROUP_SIZE:-8}"
export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export OUTPUT_DIR="${OUTPUT_DIR:-$HOME/model/ckpt/grpo_qwen3_1.7b_stabletoolbench_gpu125_fresh_$RUN_TAG}"
export TENSORBOARD_DIR="$OUTPUT_DIR/tensorboard"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.25}"
export ACTOR_OPTIMIZER_OFFLOAD=True
export TEST_FREQ="${TEST_FREQ:-10}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export PYTORCH_ALLOC_CONF=expandable_segments:False
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Fresh training requires a new output directory: $OUTPUT_DIR" >&2
  exit 1
fi
exec bash examples/grpo_trainer/run_toolbench_qwen3.sh \
  "trainer.experiment_name=grpo_qwen3_1.7b_stabletoolbench_gpu125_fresh_$RUN_TAG" \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=5120 \
  actor_rollout_ref.actor.use_torch_compile=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=5120 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=5120 \
  actor_rollout_ref.rollout.max_model_len=5120 \
  actor_rollout_ref.rollout.max_num_batched_tokens=5120 \
  actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  '+actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32]' \
  ray_init.num_cpus=24 \
  '+ray_init.runtime_env.env_vars.VLLM_ATTENTION_BACKEND=FLASH_ATTN' \
  '+ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL=INFO' \
  "$@" \
  trainer.val_before_train=True \
  trainer.resume_mode=disable \
  trainer.resume_from_path=null
