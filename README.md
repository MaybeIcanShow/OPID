<h1 align="center">
OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning
</h1>

  <div align="center">
  <p>
      <a href="https://arxiv.org/abs/2606.26790">
        <img src="https://img.shields.io/badge/Paper-arxiv%3A2606.26790-blue" alt="Paper"/>
      </a>
      <a href="https://huggingface.co/papers/2606.26790">
        <img src="https://img.shields.io/badge/Daily%20Paper-huggingface-yellow" alt="HF Paper"/>
      </a>
      <a href="https://huggingface.co/Jinyang23/OPID-ALFWorld-1.7B">
        <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-OPID--ALFWorld--1.7B-yellow" alt="Model Checkpoint"/>
      </a>
    </p>
  </div>

## News

- **2026-06-25**: We have released our paper and code.

If you have any questions ❓ or are interested in collaboration 🤝, please feel free to contact me at 
wu-jy23@mails.tsinghua.edu.cn.


## Overview

We introduce **OPID**, an **On-Policy Skill Distillation** framework that turns completed
agent trajectories into hierarchical hindsight skills. OPID routes episode-level and step-level
skills during training to provide dense token-level supervision, while requiring no analyzer,
skill retrieval, or privileged context at inference time.

<div align="center">
  <img src="figs/pipeline.png" alt="OPID pipeline" style="width:100%;">
  <br>
  <em>Figure 1: Overview of OPID.</em>
</div>

OPID achieves strong performance across ALFWorld, Search-based QA, and WebShop, improving over
outcome-only RL and competitive skill-distillation baselines.

<div align="center">
  <img src="figs/results.png" alt="OPID results" style="width:100%;">
  <br>
  <em>Figure 2: Main results.</em>
</div>

## Installation

### Python Environment

```bash
conda create -n opid python==3.12 -y
conda activate opid

pip3 install vllm==0.11.0
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=89 MAX_JOBS=8 \
  pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
```

Log in to Weights & Biases if you use WandB logging. Many example scripts use
`trainer.logger=['console','wandb']`.

```bash
export WANDB_API_KEY=your_key_here
```

OPID uses an LLM analyzer to extract episode-level and step-level hindsight skills during training.
Configure an OpenAI-compatible endpoint before running OPID scripts:

```bash
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
export OPENAI_MODEL=your_analyzer_model
export OPENAI_API_RETRIES=5
export OPENAI_API_RETRY_DELAY=1.0
```

Set the model root used by the training scripts:

```bash
export MODELS_ROOT=/path/to/models-and-checkpoints
```

### Install Supported Environments

#### 1. ALFWorld

```bash
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip3 install alfworld
```

Download PDDL and game files plus the pre-trained MaskRCNN detector:

```bash
alfworld-download -f
```

#### 2. WebShop

WebShop requires Python <=3.10, so begin by creating a separate environment:

```bash
conda create -n verl-webshop python==3.10 -y
conda activate verl-webshop
```

Install WebShop:

```bash
cd ./agent_system/environments/env_package/webshop/webshop
./setup.sh -d all
```

After WebShop is installed, return to the repo root and install the training package:

```bash
cd repo_root/
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.2
```

Some WebShop dependencies may report `typer` compatibility warnings. They can be safely ignored.

#### 3. Search-Based QA

```bash
cd ./agent_system/environments/env_package/search/third_party
pip install -e .
pip install gym==0.26.2
```

Prepare the Search-R1 style dataset:

```bash
cd repo_root/
python examples/data_preprocess/preprocess_search_r1_dataset.py
```

The processed data is saved under `~/data/searchR1_processed_direct` by default.

Build a separate retrieval environment for the local search server:

```bash
conda create -n retriever python=3.10 -y
conda activate retriever

conda install numpy==1.26.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets pyserini huggingface_hub
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y
pip install uvicorn fastapi
```

Download the index:

```bash
conda activate retriever

local_dir=~/data/searchR1
python examples/search/searchr1_download.py --local_dir $local_dir
cat $local_dir/part_* > $local_dir/e5_Flat.index
gzip -d $local_dir/wiki-18.jsonl.gz
```

Start the local flat e5 retrieval server:

```bash
conda activate retriever

bash examples/search/retriever/retrieval_launch.sh > retrieval_server.log
```

#### 4. ToolBench

ToolBench GRPO now executes tools through StableToolBench and validates on a
versioned subset of its official solvable queries. By default:

| Role | Endpoint | Model |
| --- | --- | --- |
| Tool responses after a local disk-cache miss | `http://10.8.176.56:8001/v1` | `MirrorAPI-Cache` |
| Final-answer scoring | `http://10.8.176.56:8000/v1` | `MirrorAPI` |

The judge uses the upstream FAC prompt with the user-selected MirrorAPI model.
These are MirrorAPI judge results, not results from the dedicated upstream
`stabletoolbench/Evaluator` model. An answer submission alone is never a success.
Judge transport or parsing failures are recorded separately, and an incomplete
judge run has no full-dataset success-rate metric.

Expected data layout:

```text
data/
|- ToolBench/toolllama_G123_dfs_train.json
|- StableToolBench/
|  |- tools/
|  |- tool_response_cache/
|  `- solvable_queries/
`- toolbench_stable_processed/
```

The new preprocessing path removes duplicate training queries and excludes all
official evaluation queries from training. It selects validation tasks across
all six official groups and records the source revision, hashes, exclusions,
and group counts in `metadata.json`. It uses the full task and tool definitions
when checking prompt length. Existing `data/toolbench_processed` is retained
as historical data.

Local response-cache lookup follows the upstream directory convention and
matches the actual JSON arguments. A miss calls the selected remote simulator;
demonstration responses are no longer replayed by tool name alone. The cache
client does not load a local GPU model. Set
`TOOLBENCH_BACKEND=mirrorapi` to use the other simulator explicitly.

Override the deployment without editing code:

```bash
MIRRORAPI_CACHE_URL=http://10.8.176.56:8001/v1 \
MIRRORAPI_CACHE_MODEL=MirrorAPI-Cache \
TOOLBENCH_JUDGE_URL=http://10.8.176.56:8000/v1 \
TOOLBENCH_JUDGE_MODEL=MirrorAPI \
bash examples/grpo_trainer/run_toolbench_qwen3_3gpu_fresh.sh
```

For authenticated servers, set `STABLETOOLBENCH_API_KEY` and
`STABLETOOLBENCH_JUDGE_API_KEY`; keys are read from the environment rather than
put into command-line overrides. Private model endpoints bypass environment
HTTP proxies by default.

Each turn preserves the original task and tool definitions. If history exceeds
the prompt budget, oldest turns are dropped first; a shortened latest turn is
marked explicitly. The initial task is never silently left-truncated.

Validation saves final answers, judge statuses/reasons, tool-response sources,
and interaction histories to `$OUTPUT_DIR/validation/<step>.jsonl`. Metrics use
the `val/stabletoolbench/` prefix, including `judge_coverage`,
`judge_success_rate`, per-group results, and cache/simulator response counts.
An additional `episode_reward_mean` is trajectory-weighted; the older
`test_score` remains step-weighted for compatibility.

Use a fresh output directory when switching from the old data and reward
protocol. Old checkpoints' optimizer/data-loader progress does not describe
the new deduplicated dataset.

## Training

All OPID scripts live under `examples/opid_trainer/` and assume the repo root as the working directory.

```bash
bash examples/opid_trainer/run_alfworld_opid_guide.sh
bash examples/opid_trainer/run_webshop_opid_guide.sh
bash examples/opid_trainer/run_search_opid_guide.sh
```

Additional scripts are provided for Qwen3:

```bash
bash examples/opid_trainer/run_alfworld_opid_guide_qwen3.sh
bash examples/opid_trainer/run_webshop_opid_guide_qwen3.sh
bash examples/opid_trainer/run_search_opid_guide_qwen3.sh
```

### ToolBench GRPO

The fresh-run launcher prepares the deduplicated training data and official
StableToolBench validation subset, then starts Qwen3 GRPO on GPUs **1, 2, 5**:

```bash
export MODEL_PATH=$HOME/model/Qwen3-1.7B
export PYTHON_BIN=$HOME/miniconda3/envs/opid/bin/python
export CUDA_VISIBLE_DEVICES=1,2,5

bash examples/grpo_trainer/run_toolbench_qwen3_3gpu_fresh.sh
```

The launcher defaults to 2,048 training tasks, 128 validation tasks across six
official groups, a training/minibatch size of 12, and eight rollouts per training
task. It uses `data/toolbench_stable_processed/{train,test}.parquet` and writes
checkpoints to a new
`$HOME/model/ckpt/grpo_qwen3_1.7b_stabletoolbench_gpu125_fresh_<timestamp>`
directory. Initial validation runs before training; subsequent validation and
checkpoint saves occur every 10 steps. The fresh launcher disables resume and
requires an output directory that does not already exist.

Tool calls first check the local StableToolBench response cache by actual
arguments, then use **MirrorAPI-Cache** at `http://10.8.176.56:8001/v1` on a miss.
**MirrorAPI** at `http://10.8.176.56:8000/v1` scores final answers using the upstream
FAC prompt. Report these scores with the configured MirrorAPI judge identified;
the official dedicated FAC evaluator is a separate model. These remote services
must already be running before the launcher starts.

Override dataset and output paths, sizes, or deployed endpoints with environment
variables, keeping three-GPU batch sizes divisible by three:

```bash
TRAIN_SIZE=2048 \
VAL_SIZE=128 \
TRAIN_BATCH_SIZE=12 \
PPO_MINI_BATCH_SIZE=12 \
DATA_DIR=$PWD/data/toolbench_stable_processed \
OUTPUT_DIR=$HOME/model/ckpt/toolbench-grpo-$(date +%Y%m%d_%H%M%S) \
MIRRORAPI_CACHE_URL=http://10.8.176.56:8001/v1 \
TOOLBENCH_JUDGE_URL=http://10.8.176.56:8000/v1 \
bash examples/grpo_trainer/run_toolbench_qwen3_3gpu_fresh.sh
```

To prepare data separately, including downloading missing official evaluation
queries at the pinned revision, run the following after setting `MODEL_PATH`
and `PYTHON_BIN` as above:

```bash
"$PYTHON_BIN" -m examples.data_preprocess.preprocess_toolbench \
  --source data/ToolBench/toolllama_G123_dfs_train.json \
  --tool-root data/StableToolBench/tools \
  --eval-query-dir data/StableToolBench/solvable_queries \
  --download-eval-queries \
  --output-dir data/toolbench_stable_processed \
  --tokenizer "$MODEL_PATH" \
  --max-initial-prompt-tokens 3584 \
  --train-size 2048 \
  --val-size 128
```

The preparation step excludes all official evaluation queries from training and
records duplicate removals, prompt filtering, source hashes, and group sizes in
`data/toolbench_stable_processed/metadata.json`. The launcher rechecks the same
sources before each run. See the ToolBench setup section above for the expected
tool/cache directory layout, endpoint credentials, and validation output fields.

Useful OPID parameters:

- `OPID_ANALYSIS_MAX_STEP_SKILLS_PER_TRAJ`: maximum number of critical step skills per trajectory.
- `OPID_EPISODE_SKILL_TEACHER_ADV_W`: weight for episode-level skill teacher advantage.
- `OPID_STEP_SKILL_TEACHER_ADV_W`: weight for step-level skill teacher advantage.


## Merge Checkpoints

See `scripts/model_merger.py` for FSDP/Megatron merge examples using paths under
`./checkpoints/...`.

## ⭐ Citation

If you find this project useful, welcome to cite us.

```bibtex
@article{yang2026opid,
  title={OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning},
  author={Yang, Shuo and Wu, Jinyang and Lu, Zhengxi and Shen, Yuhao and Zhang, Fan and Feng, Lang and Zhang, Shuai and Luo, Haoran and Lian, Zheng and Wen, Zhengqi and others},
  journal={arXiv preprint arXiv:2606.26790},
  year={2026}
}
```

## Acknowledgement

This project builds on [verl-agent](https://github.com/langfengQ/verl-agent),
[veRL](https://github.com/volcengine/verl),
[SkillRL](https://github.com/aiming-lab/SkillRL). We thank the authors of those projects.
