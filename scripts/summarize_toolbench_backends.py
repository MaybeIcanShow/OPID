"""Check and summarize four saved ToolBench evaluation combinations."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def load_records(directory):
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    by_id = {str(row["record_id"]): row for row in rows}
    if len(rows) != len(by_id):
        raise ValueError(f"Duplicate records: {directory}")
    for row in rows:
        if len(row["generations"]) != len(row["step_events"]):
            raise ValueError(f"Missing generation or event: {row['record_id']}")
        total = sum(e["reward"] for e in row["step_events"])
        if abs(total - row["episode_reward"]) > 1e-6:
            raise ValueError(f"Reward sum differs: {row['record_id']}")
    return by_id


def compare(left, right, left_name, right_name):
    counts = Counter()
    changed = []
    identical_answer_conflicts = []
    for record_id in left:
        before, after = left[record_id], right[record_id]
        first, second = before["evaluation"]["status"], after["evaluation"]["status"]
        if first not in {"success", "failure"} or second not in {"success", "failure"}:
            counts["excluded_due_to_judge_error_or_unscored"] += 1
            continue
        counts["paired_scored_tasks"] += 1
        if first == second:
            counts["both_pass" if first == "success" else "both_fail"] += 1
        else:
            kind = "improved" if second == "success" else "declined"
            counts[kind] += 1
            changed.append({"record_id": record_id, "change": kind, "query": before["query"],
                            "left_status": first, "right_status": second,
                            "left_answer": before["final_answer"], "right_answer": after["final_answer"],
                            "left_tool_calls": before["tool_calls"], "right_tool_calls": after["tool_calls"],
                            "left_reason": before["evaluation"]["reason"],
                            "right_reason": after["evaluation"]["reason"]})
            if before["final_answer"] == after["final_answer"]:
                identical_answer_conflicts.append(record_id)
        if before["final_answer"] == after["final_answer"]:
            counts["identical_final_answers"] += 1
    paired = counts["paired_scored_tasks"]
    for key in ("both_pass", "both_fail", "improved", "declined",
                "excluded_due_to_judge_error_or_unscored", "identical_final_answers"):
        counts.setdefault(key, 0)
    return {"left": left_name, "right": right_name, **dict(counts),
            "paired_success_delta_percentage_points": 100 * (counts["improved"] - counts["declined"]) / paired if paired else None,
            "same_answer_different_judge_verdict_ids": identical_answer_conflicts}, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    args = parser.parse_args()
    root = Path(args.run_dir)
    sets, summaries, flat = {}, {}, []
    for label in ("initial", "step30"):
        for backend in ("mirrorapi_cache", "mirrorapi"):
            name = f"{label}/{backend}"
            directory = root / label / backend
            summary = json.loads((directory / "summary.json").read_text())
            if not summary["complete"]:
                raise ValueError(f"Evaluation incomplete: {name}")
            records = load_records(directory)
            expected = {str(record_id) for record_id in summary["metadata"]["record_ids"]}
            if set(records) != expected:
                raise ValueError(f"Record set differs: {name}")
            sets[name] = records
            summaries[name] = summary
            metrics = summary["metrics"]
            scored = metrics["judge_scored_tasks"]
            successes = metrics["judge_success_tasks"]
            flat.append({"weight": label, "backend": backend, "tasks": len(records),
                         "judge_pass": successes, "judge_scored": scored,
                         "judge_pass_rate_scored": successes / scored if scored else None,
                         "judge_errors": metrics["judge_error_tasks"],
                         "answers_submitted": metrics["submitted_answers"],
                         "give_up": metrics["termination_counts"].get("give_up", 0),
                         "max_steps": metrics["termination_counts"].get("max_steps", 0),
                         "tool_calls": metrics["tool_calls"],
                         "disk_cache_calls": metrics.get("tool_responses/disk_cache_count", 0),
                         "simulator_calls": metrics.get(f"tool_responses/{backend}_count", 0),
                         "unavailable_calls": metrics.get("tool_responses/unavailable_count", 0),
                         "tool_error_calls": metrics["tool_error_calls"],
                         "tool_error_rate": metrics["tool_error_rate"],
                         "invalid_actions": metrics["invalid_actions"],
                         "action_valid_rate": metrics["action_valid_rate"],
                         "episode_reward_mean": metrics["episode_reward_mean"]})
    first_name = next(iter(sets))
    first = sets[first_name]
    for name, records in sets.items():
        if set(first) != set(records):
            raise ValueError(f"Queries differ across combinations: {name}")
        for record_id in first:
            if records[record_id]["query"] != first[record_id]["query"]:
                raise ValueError(f"Query text differs: {record_id}")
            if records[record_id]["first_prompt_sha256"] != first[record_id]["first_prompt_sha256"]:
                raise ValueError(f"Initial prompt differs: {name}/{record_id}")
    comparisons = {}
    pairs = [(f"initial/{b}", f"step30/{b}", f"weights_on_{b}") for b in ("mirrorapi_cache", "mirrorapi")]
    pairs += [(f"{w}/mirrorapi_cache", f"{w}/mirrorapi", f"backends_with_{w}") for w in ("initial", "step30")]
    for left, right, name in pairs:
        comparison, changes = compare(sets[left], sets[right], left, right)
        comparisons[name] = comparison
        (root / f"{name}_changes.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in changes))
    result = {"complete": True, "tasks_per_combination": len(first),
              "trajectories": sum(len(rows) for rows in sets.values()),
              "all_queries_and_initial_prompts_identical": True, "table": flat,
              "paired_comparisons": comparisons, "full_summaries": summaries,
              "metric_note": "MirrorAPI score using the pinned official FAC prompt; not verified factual accuracy or the dedicated StableToolBench Evaluator score.",
              "cache_note": "Both backends read the same official disk cache first. Cache misses use the selected simulator. Disk cache is read-only.",
              "parser_note": "Same strict JSON adapter as training. Malformed simulator JSON and truncated completions are rejected. Upstream StableToolBench also attempts regex field recovery; these scores therefore measure the current local pipeline, not that upstream behavior.",
              "parser_diagnostic": "mirrorapi_format_probe.json",
              "generation_note": "Greedy decoding in all four combinations. Earlier training validation used sampling and is not directly comparable."}
    (root / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (root / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({"table": flat, "paired_comparisons": comparisons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
