"""Compare first-turn protocols and run real multi-turn tool/judge smoke tasks.

Run with CUDA_VISIBLE_DEVICES set to an available GPU. This does not train weights.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from agent_system.environments.env_manager import ToolBenchEnvironmentManager
from agent_system.environments.env_package.toolbench.context import fit_messages
from agent_system.environments.env_package.toolbench.envs import ToolBenchMultiProcessEnv
from agent_system.environments.env_package.toolbench.metrics import summarize_evaluations
from agent_system.environments.env_package.toolbench.projection import parse_tool_action, toolbench_projection
from agent_system.environments.env_package.toolbench.protocol import ACTION_STOPS
from agent_system.environments.env_package.toolbench.stabletoolbench import parse_arguments, StableToolBenchError


def valid_action(text, record):
    name, raw = parse_tool_action(text)
    try:
        arguments = parse_arguments(raw)
    except StableToolBenchError:
        return False
    if name.lower() == "finish":
        return arguments.get("return_type") == "give_up_and_restart" or (
            arguments.get("return_type") == "give_answer" and bool(arguments.get("final_answer"))
        )
    return isinstance(record["function_map"].get(name), dict)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/zhengbaowei/model/Qwen3-1.7B")
    parser.add_argument("--data", default="data/toolbench_stable_processed/test.parquet")
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compare-first-turn", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = pq.read_table(args.data).slice(0, args.sample_size).to_pylist()
    records = [row["env_kwargs"] for row in rows]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml").env
    config.max_steps = args.max_steps
    env = ToolBenchMultiProcessEnv(0, len(records), 1, False, config)
    manager = ToolBenchEnvironmentManager(env, toolbench_projection, OmegaConf.create({"env": {"history_length": 12}}))
    obs, _ = manager.reset(records)

    from vllm import LLM, SamplingParams
    model = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=1,
                gpu_memory_utilization=0.25, max_model_len=5120, max_num_batched_tokens=5120,
                max_num_seqs=32, enforce_eager=True, enable_prefix_caching=True, seed=42)
    summary = {"sample_size": len(records), "record_ids": [r["record_id"] for r in records],
               "model": args.model, "thinking": False, "stops": ACTION_STOPS}
    if args.compare_first_turn:
        old_prompts = [tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=True)
                       for row in rows]
        completions = model.generate(old_prompts, SamplingParams(temperature=0.6, top_p=0.95, top_k=-1, max_tokens=1024), use_tqdm=False)
        old = [{"id": r["record_id"], "text": c.outputs[0].text, "finish_reason": c.outputs[0].finish_reason,
                "tokens": len(c.outputs[0].token_ids), "valid": valid_action(c.outputs[0].text, r)}
               for r, c in zip(records, completions)]
        (output / "original_first_turn.json").write_text(json.dumps(old, ensure_ascii=False, indent=2))
        summary["original_first_turn_valid"] = sum(r["valid"] for r in old)
        summary["original_first_turn_length_stops"] = sum(r["finish_reason"] == "length" for r in old)
        print("ORIGINAL", json.dumps({k:v for k,v in summary.items() if k.startswith("original")}), flush=True)

    sampling = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=1024, stop=ACTION_STOPS)
    done = np.zeros(len(records), dtype=bool)
    rounds = []
    started = time.monotonic()
    for step in range(args.max_steps):
        active = np.flatnonzero(~done).tolist()
        prompts = []
        for index in active:
            ctx = obs["toolbench_context"][index]
            messages = fit_messages(ctx["initial"], ctx["history"], tokenizer, 4096, {"enable_thinking": False})
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        completions = model.generate(prompts, sampling, use_tqdm=False)
        actions, generation = [""] * len(records), [{} for _ in records]
        for index, completion in zip(active, completions):
            sample = completion.outputs[0]
            actions[index] = sample.text
            generation[index] = {"finish_reason": sample.finish_reason, "stop_reason": sample.stop_reason,
                                 "token_count": len(sample.token_ids),
                                 "raw_text": tokenizer.decode(sample.token_ids, skip_special_tokens=True)}
        if step == 0:
            summary["fixed_first_turn_valid"] = sum(valid_action(actions[i], records[i]) for i in active)
            summary["fixed_first_turn_length_stops"] = sum(generation[i]["finish_reason"] == "length" for i in active)
        obs, _, next_done, infos = manager.step(actions, generation_metadata=generation)
        done |= next_done
        counts = {"round": step + 1, "active": len(active),
                  "valid": sum(int(infos[i]["is_action_valid"]) for i in active),
                  "length_stops": sum(generation[i]["finish_reason"] == "length" for i in active),
                  "done": int(done.sum())}
        rounds.append(counts)
        print("ROUND", json.dumps(counts), flush=True)
        if done.all():
            break
    results = manager.get_evaluation_records()
    summary.update(metrics=summarize_evaluations(results), rounds=rounds, elapsed_seconds=time.monotonic()-started)
    (output / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    env.close()
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
