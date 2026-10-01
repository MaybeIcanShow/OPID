"""Trajectory-level metrics; unscored requests never become zero task scores."""

from collections import Counter


def summarize_evaluations(records):
    if not records:
        return {}
    total = len(records)
    outcomes = Counter(record["evaluation"]["status"] for record in records)
    scored = outcomes["success"] + outcomes["failure"]
    result = {
        "tasks": total,
        "judge_scored_tasks": scored,
        "judge_coverage": scored / total,
        "judge_error_tasks": outcomes["error"],
        "unscored_tasks": outcomes["unscored"],
        "answer_submission_rate": sum(bool(r.get("answer_submitted")) for r in records) / total,
    }
    if scored == total:
        result["judge_success_rate"] = outcomes["success"] / total
    elif scored:
        result["judge_success_rate_scored_subset"] = outcomes["success"] / scored
    sources = Counter()
    for record in records:
        sources.update(record.get("response_sources", {}))
    calls = sum(sources.values())
    result["tool_calls"] = calls
    for source, count in sorted(sources.items()):
        result[f"tool_responses/{source}_count"] = count
        result[f"tool_responses/{source}_fraction"] = count / calls if calls else 0.0
    for group in sorted({r.get("eval_group", "") for r in records} - {""}):
        subset = [r for r in records if r.get("eval_group") == group]
        group_scored = [r for r in subset if r["evaluation"]["status"] in ("success", "failure")]
        result[f"groups/{group}/tasks"] = len(subset)
        result[f"groups/{group}/judge_coverage"] = len(group_scored) / len(subset)
        if len(group_scored) == len(subset):
            result[f"groups/{group}/judge_success_rate"] = sum(
                r["evaluation"]["status"] == "success" for r in group_scored
            ) / len(subset)
    return result
