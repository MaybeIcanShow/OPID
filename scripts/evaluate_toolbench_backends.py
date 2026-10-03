"""Run fresh, paired ToolBench trajectories against the two MirrorAPI backends.

Each model invocation reuses one vLLM engine for both backends. Completed batches
are saved immediately and can be resumed without replaying completed tasks.
"""

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np
from omegaconf import OmegaConf
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from agent_system.environments.env_manager import ToolBenchEnvironmentManager
from agent_system.environments.env_package.toolbench.context import fit_messages
from agent_system.environments.env_package.toolbench.envs import ToolBenchMultiProcessEnv
from agent_system.environments.env_package.toolbench.metrics import summarize_evaluations
from agent_system.environments.env_package.toolbench.projection import toolbench_projection
from agent_system.environments.env_package.toolbench.protocol import ACTION_STOPS


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def log(event, **fields):
    print(json.dumps({"time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                      "event": event, **fields}, ensure_ascii=False), flush=True)


def summarize(records):
    metrics = summarize_evaluations(records)
    metrics["judge_success_tasks"] = sum(r["evaluation"]["status"] == "success" for r in records)
    metrics["judge_failure_tasks"] = sum(r["evaluation"]["status"] == "failure" for r in records)
    metrics["submitted_answers"] = sum(bool(r["answer_submitted"]) for r in records)
    metrics["termination_counts"] = dict(Counter(r["termination_reason"] for r in records))
    metrics["episode_reward_mean"] = float(np.mean([r["episode_reward"] for r in records])) if records else None
    events = [e for r in records for e in r["step_events"]]
    tool_events = [e for e in events if e["tool_calling"]]
    metrics["tool_error_calls"] = sum(e["reward"] < 0 for e in tool_events)
    metrics["tool_error_rate"] = metrics["tool_error_calls"] / len(tool_events) if tool_events else 0.0
    metrics["tool_error_codes"] = dict(Counter(e["response_error_code"] for e in tool_events
                                               if e.get("response_error_code")))
    metrics["invalid_actions"] = sum(not e["is_action_valid"] for e in events)
    return metrics


def evaluate_backend(args, records, tokenizer, model, sampling_class, backend, base_metadata):
    output = Path(args.output) / args.label / backend
    output.mkdir(parents=True, exist_ok=True)
    metadata = {**base_metadata, "backend": backend}
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError(f"Resume configuration differs: {metadata_path}")
    else:
        write_json(metadata_path, metadata)
    results_path = output / "results.jsonl"
    completed = [json.loads(line) for line in results_path.read_text().splitlines()] if results_path.exists() else []
    completed_ids = {str(r["record_id"]) for r in completed}
    wanted_ids = {str(r["record_id"]) for r in records}
    if len(completed_ids) != len(completed) or not completed_ids <= wanted_ids:
        raise ValueError("Result IDs are duplicated or outside the dataset")
    pending = [r for r in records if str(r["record_id"]) not in completed_ids]
    log("backend_start", label=args.label, backend=backend, completed=len(completed), remaining=len(pending))
    started = time.monotonic()
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml").env
    config.max_steps = args.max_steps
    config.toolbench.backend = backend
    config.toolbench.cache_enabled = args.disk_cache
    config.toolbench.mirrorapi_url = "http://10.8.176.56:8000/v1"
    config.toolbench.mirrorapi_model = "MirrorAPI"
    config.toolbench.mirrorapi_cache_url = "http://10.8.176.56:8001/v1"
    config.toolbench.mirrorapi_cache_model = "MirrorAPI-Cache"
    config.toolbench.max_concurrency = args.concurrency
    config.toolbench.evaluator.api_base = "http://10.8.176.56:8000/v1"
    config.toolbench.evaluator.model = "MirrorAPI"
    config.toolbench.evaluator.mode = "fac_evidence"
    config.toolbench.evaluator.enabled = True
    config.toolbench.evaluator.max_concurrency = args.concurrency
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset:offset + args.batch_size]
        env = ToolBenchMultiProcessEnv(42, len(batch), 1, False, config)
        manager = ToolBenchEnvironmentManager(env, toolbench_projection,
                                             OmegaConf.create({"env": {"history_length": args.max_steps}}))
        observations, _ = manager.reset(batch)
        done = np.zeros(len(batch), dtype=bool)
        rewards = np.zeros(len(batch), dtype=float)
        events = [[] for _ in batch]
        prompt_hashes = [None] * len(batch)
        try:
            for step in range(args.max_steps):
                active = np.flatnonzero(~done).tolist()
                prompts = []
                parameters = []
                for index in active:
                    context = observations["toolbench_context"][index]
                    messages = fit_messages(context["initial"], context["history"], tokenizer,
                                            4096, {"enable_thinking": False})
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                           add_generation_prompt=True, enable_thinking=False)
                    prompts.append(prompt)
                    if step == 0:
                        prompt_hashes[index] = hashlib.sha256(prompt.encode()).hexdigest()
                    # Each record/round gets the same seed across all four runs.
                    seed = int(hashlib.sha256(f"42:{batch[index]['record_id']}:{step}".encode()).hexdigest()[:8], 16)
                    parameters.append(sampling_class(temperature=args.temperature, top_p=1.0, top_k=-1,
                                                     max_tokens=1024, stop=ACTION_STOPS, seed=seed))
                generation_started = time.monotonic()
                completions = model.generate(prompts, parameters, use_tqdm=False)
                generation_seconds = time.monotonic() - generation_started
                actions = [""] * len(batch)
                generation = [{} for _ in batch]
                for index, completion in zip(active, completions):
                    sample = completion.outputs[0]
                    actions[index] = sample.text
                    generation[index] = {"finish_reason": sample.finish_reason, "stop_reason": sample.stop_reason,
                                         "token_count": len(sample.token_ids),
                                         "prompt_token_count": len(completion.prompt_token_ids),
                                         "raw_text": tokenizer.decode(sample.token_ids, skip_special_tokens=True)}
                environment_started = time.monotonic()
                observations, step_rewards, next_done, infos = manager.step(actions, generation_metadata=generation)
                environment_seconds = time.monotonic() - environment_started
                rewards += step_rewards
                done |= next_done
                for index in active:
                    info = infos[index]
                    events[index].append({"round": step + 1, "reward": float(step_rewards[index]),
                                          "is_action_valid": bool(info.get("is_action_valid")),
                                          "tool_calling": bool(info.get("tool_calling")),
                                          "response_source": info.get("response_source"),
                                          "response_error_code": info.get("response_error_code"),
                                          "termination_reason": info.get("termination_reason")})
                progress = {"label": args.label, "backend": backend, "batch_offset": offset,
                            "round": step + 1, "active": len(active), "batch_done": int(done.sum()),
                            "saved_tasks": len(completed), "total_tasks": len(records),
                            "generation_seconds": round(generation_seconds, 2),
                            "environment_seconds": round(environment_seconds, 2)}
                write_json(output / "progress.json", progress)
                log("round", **progress)
                if done.all():
                    break
            batch_results = manager.get_evaluation_records()
            for index, result in enumerate(batch_results):
                result.update(weight_label=args.label, backend=backend, episode_reward=float(rewards[index]),
                              step_events=events[index], first_prompt_sha256=prompt_hashes[index])
            with results_path.open("a") as stream:
                for result in batch_results:
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            completed.extend(batch_results)
            summary = {"complete": len(completed) == len(records), "metrics": summarize(completed),
                       "elapsed_seconds_this_invocation": time.monotonic() - started, "metadata": metadata}
            write_json(output / "summary.json", summary)
            log("batch_saved", label=args.label, backend=backend, saved_tasks=len(completed),
                successes=summary["metrics"]["judge_success_tasks"],
                errors=summary["metrics"]["judge_error_tasks"])
        finally:
            env.close()
    log("backend_complete", label=args.label, backend=backend, saved_tasks=len(completed), metrics=summarize(completed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True, choices=["initial", "step30"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", default="data/toolbench_stable_processed_v2/test.parquet")
    parser.add_argument("--backends", nargs="+", choices=["mirrorapi_cache", "mirrorapi"],
                        default=["mirrorapi_cache", "mirrorapi"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--disk-cache", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    label_directory = Path(args.output) / args.label
    label_directory.mkdir(parents=True, exist_ok=True)
    # A supervisor may resume a model while an earlier invocation is still live.
    # Serialize invocations for this output label before loading another engine.
    lock = (label_directory / ".evaluation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    data_path = Path(args.data).resolve()
    rows = pq.read_table(data_path).to_pylist()[:args.sample_size]
    records = [row["env_kwargs"] for row in rows]
    if not records or len({str(r["record_id"]) for r in records}) != len(records):
        raise ValueError("Expected a nonempty dataset with unique record IDs")
    model_path = Path(args.model).resolve()
    metadata = {"weight_label": args.label, "model": str(model_path), "data": str(data_path),
                "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "record_ids": [r["record_id"] for r in records], "disk_cache": args.disk_cache,
                "generation": {"temperature": args.temperature, "top_p": 1.0, "top_k": -1,
                               "seed_scheme": "sha256(42:record_id:zero_based_round)[:8]",
                               "thinking": False, "prompt_tokens": 4096, "response_tokens": 1024,
                               "max_steps": args.max_steps, "stop": ACTION_STOPS},
                "tool_urls": {"mirrorapi": "http://10.8.176.56:8000/v1",
                              "mirrorapi_cache": "http://10.8.176.56:8001/v1"},
                "tool_models": {"mirrorapi": "MirrorAPI", "mirrorapi_cache": "MirrorAPI-Cache"},
                "tool_temperature": 0, "tool_max_tokens": 2048, "tool_seed": 42,
                "judge": {"url": "http://10.8.176.56:8000/v1", "model": "MirrorAPI",
                          "mode": "fac_evidence", "max_tokens": 1024},
                "batch_size": args.batch_size, "concurrency": args.concurrency}
    all_complete = True
    for backend in args.backends:
        directory = label_directory / backend
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            all_complete = False
            continue
        summary = json.loads(summary_path.read_text())
        expected_metadata = {**metadata, "backend": backend}
        if summary["metadata"] != expected_metadata:
            raise ValueError(f"Resume configuration differs: {summary_path}")
        all_complete &= bool(summary["complete"])
    if all_complete:
        log("already_complete", label=args.label, backends=args.backends)
        return
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    from vllm import LLM, SamplingParams
    log("model_loading", label=args.label, model=str(model_path), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    model = LLM(model=str(model_path), dtype="bfloat16", tensor_parallel_size=1, seed=42,
                gpu_memory_utilization=0.25, max_model_len=5120, max_num_batched_tokens=5120,
                max_num_seqs=args.batch_size, enforce_eager=True, enable_prefix_caching=True)
    for backend in args.backends:
        evaluate_backend(args, records, tokenizer, model, SamplingParams, backend, metadata)


if __name__ == "__main__":
    main()
