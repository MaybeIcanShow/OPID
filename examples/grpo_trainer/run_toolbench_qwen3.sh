#!/usr/bin/env bash
set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export TOKENIZERS_PARALLELISM=false
ulimit -u 65536

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/opid/bin/python}"
MODEL_PATH="${MODEL_PATH:-$HOME/model/Qwen3-1.7B}"
DATA_DIR="${DATA_DIR:-$PWD/data/toolbench_processed}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/model/ckpt/grpo_qwen3_1.7b_toolbench_2048}"
TRAIN_SIZE="${TRAIN_SIZE:-2048}"
VAL_SIZE="${VAL_SIZE:-128}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
GROUP_SIZE="${GROUP_SIZE:-8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
ENABLE_THINKING="${ENABLE_THINKING:-True}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-1}"
ENABLE_TENSORBOARD="${ENABLE_TENSORBOARD:-True}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-$OUTPUT_DIR/tensorboard}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

if [[ "${ENABLE_TENSORBOARD,,}" == "true" ]]; then
  LOGGER="['console','tensorboard']"
else
  LOGGER="['console']"
fi
export TENSORBOARD_DIR

"$PYTHON_BIN" -m examples.data_preprocess.preprocess_toolbench \
  --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE" --output-dir "$DATA_DIR"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.val_batch_size="$VAL_SIZE" \
  data.max_prompt_length=4096 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.filter_overlong_prompts_workers=8 \
  data.truncation=left \
  data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking="$ENABLE_THINKING" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.max_num_seqs=128 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.use_kl_in_reward=False \
  env.env_name=toolbench \
  env.max_steps=12 \
  env.history_length=0 \
  env.rollout.n="$GROUP_SIZE" \
  env.toolbench.cache_root="$PWD/data/StableToolBench/server/tool_response_cache" \
  env.toolbench.service_url="${STABLETOOLBENCH_SERVICE_URL:-http://127.0.0.1:12001/virtual}" \
  env.toolbench.request_timeout=8 \
  trainer.critic_warmup=0 \
  trainer.logger="$LOGGER" \
  trainer.project_name=agentic_toolbench \
  trainer.experiment_name=grpo_qwen3_1.7b_toolbench_2048 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.val_before_train=True \
  trainer.test_freq="$TEST_FREQ" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.resume_mode=auto \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.rollout_data_dir="$OUTPUT_DIR/rollouts" \
  ray_init.num_cpus=16 \
  "$@"
