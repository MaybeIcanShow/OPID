#!/usr/bin/env bash
# Four-GPU shared-memory profile; preserves the original batch/group sizes for resume.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,5,7}"
export N_GPUS_PER_NODE=4
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_LOGGING_LEVEL=INFO
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# vLLM sleep mode is incompatible with expandable_segments=True.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export PYTORCH_ALLOC_CONF=expandable_segments:False
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.25}"
export ACTOR_OPTIMIZER_OFFLOAD=True
# Input width is 4096 prompt + 1024 response; dynamic budget must be >= 5120.
DYNAMIC_TOKEN_BUDGET="${DYNAMIC_TOKEN_BUDGET:-5120}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
# Fixed actor microbatch 1 lowers update peaks on the shared GPU 7.
ACTOR_USE_DYNAMIC_BSZ="${ACTOR_USE_DYNAMIC_BSZ:-False}"
export TEST_FREQ="${TEST_FREQ:-10}"
export SAVE_FREQ="${SAVE_FREQ:-10}"

exec bash examples/grpo_trainer/run_toolbench_qwen3.sh \
  "actor_rollout_ref.actor.use_dynamic_bsz=$ACTOR_USE_DYNAMIC_BSZ" \
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$DYNAMIC_TOKEN_BUDGET" \
  actor_rollout_ref.actor.use_torch_compile=True \
  "actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$DYNAMIC_TOKEN_BUDGET" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$DYNAMIC_TOKEN_BUDGET" \
  actor_rollout_ref.rollout.max_model_len=5120 \
  actor_rollout_ref.rollout.max_num_batched_tokens=5120 \
  actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  '+actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32]' \
  trainer.val_before_train=False \
  ray_init.num_cpus=32 \
  '+ray_init.runtime_env.env_vars.VLLM_ATTENTION_BACKEND=FLASH_ATTN' \
  '+ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL=INFO' \
  "$@"
