"""Audit saved ToolBench trajectories with GLM-5.3 via the supplied Ark endpoint.

Reuse the exact DeepSeek rubric and evidence extraction. Read credentials from a
hidden terminal prompt; save only inputs' hashes, model verdicts, and API metadata.
Run as ``python -m scripts.evaluate_toolbench_glm``.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

from scripts.evaluate_toolbench_deepseek import SYSTEM_PROMPT, check_verdict, make_payload

MODEL = "GLM-5.3"
API_URL = "https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions"
REQUEST_SETTINGS = {
    "model": MODEL,
    "thinking": {"type": "enabled"},
    "reasoning_effort": "high",
    "temperature": 0,
    "response_format": {"type": "json_object"},
    "max_tokens": 8192,
    "stream": False,
}


def audit_one(step: int, row: dict, key: str, cap: int) -> dict:
    base = {
        "step": step,
        "record_id": str(row["record_id"]),
        "original_status": row["evaluation"]["status"],
        "original_score": row["evaluation"]["score"],
    }
    if not row["final_answer"].strip():
        return {
            **base, "source": "no_answer_rule", "evidence_truncated": False,
            "verdict": {"complete": "no", "grounding": "uncertain", "missing": ["final answer"],
                        "reason": "未提交最终答案。", "confidence": "high"},
            "usage": {},
        }
    payload, truncated = make_payload(row, cap)
    payload.update(REQUEST_SETTINGS)
    last_error = ""
    attempts = []
    for attempt in range(4):
        started = time.monotonic()
        try:
            response = requests.post(
                API_URL, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload, timeout=(15, 180),
            )
            if response.status_code >= 400:
                detail = response.text.replace(key, "[redacted]")[:500]
                last_error = f"HTTP {response.status_code}: {detail}"
                if response.status_code in {400, 401, 402, 403, 404}:
                    raise RuntimeError(last_error)
                attempts.append({"http_status": response.status_code, "error": last_error})
                time.sleep(min(2 ** attempt, 12))
                continue
            body = response.json()
            if str(body.get("model", "")).lower() != "glm-5.3":
                raise RuntimeError(f"Unexpected returned model: {body.get('model')!r}; expected GLM-5.3")
            choice = body["choices"][0]
            usage = body.get("usage") or {}
            attempts.append({"finish_reason": choice.get("finish_reason"), "usage": usage,
                             "elapsed_seconds": round(time.monotonic() - started, 3)})
            if choice.get("finish_reason") != "stop":
                if choice.get("finish_reason") == "length":
                    payload["max_tokens"] = min(payload["max_tokens"] * 2, 32768)
                raise ValueError(f"finish_reason={choice.get('finish_reason')}")
            content = choice["message"]["content"].strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
            raw_verdict = json.loads(content)
            value = dict(raw_verdict) if isinstance(raw_verdict, dict) else raw_verdict
            # GLM sometimes omits or nulls this descriptive field on complete answers.
            # Preserve its raw value; never infer or change either scoring label.
            if isinstance(value, dict) and value.get("complete") == "yes" and value.get("missing") is None:
                value["missing"] = []
            if isinstance(value, dict) and isinstance(value.get("missing"), str):
                value["missing"] = [value["missing"]] if value["missing"].strip() else []
            verdict = check_verdict(value)
            return {
                **base, "source": "glm_api", "verdict": verdict,
                "raw_verdict": raw_verdict,
                "evidence_truncated": truncated, "response_model": body.get("model"),
                "usage": usage, "attempts": attempts,
                "effective_max_tokens": payload["max_tokens"],
            }
        except RuntimeError:
            raise
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            last_error = (type(exc).__name__ + ": " + str(exc)).replace(key, "[redacted]")[:300]
            time.sleep(min(2 ** attempt, 12))
    return {**base, "source": "error", "error": last_error, "evidence_truncated": truncated,
            "usage": {}, "attempts": attempts}


def load_results(path: Path) -> dict:
    result = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            result[(row["step"], str(row["record_id"]))] = row
    return result


def run_batch(rows, path, key, workers, cap):
    saved = load_results(path)
    pending = [(step, row) for step, row in rows
               if saved.get((step, str(row["record_id"])), {}).get("source", "error") == "error"]
    print(f"{path.name}: selected={len(rows)}, pending={len(pending)}", flush=True)
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {}
    try:
        with path.open("a") as handle:
            futures = {pool.submit(audit_one, step, row, key, cap): (step, row["record_id"])
                       for step, row in pending}
            for count, future in enumerate(as_completed(futures), 1):
                result = future.result()
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                if count % 25 == 0 or count == len(pending):
                    print(f"{path.name}: completed {count}/{len(pending)}", flush=True)
    finally:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def summarize(output_dir, rows, deepseek_dir):
    saved = load_results(output_dir / "results.jsonl")
    original = {(step, str(row["record_id"])): row for step, row in rows}
    deepseek = load_results(deepseek_dir / "results.jsonl")
    saved = {key: result for key, result in saved.items() if key in original}
    controls = list(load_results(output_dir / "calibration_results.jsonl").values())
    controls_correct = sum(x.get("verdict", {}).get("complete") ==
                           ("yes" if x["original_status"] == "success" else "no")
                           for x in controls if x["source"] != "error")
    summary = {
        "model_requested": MODEL,
        "model_returned": sorted({x["response_model"] for x in saved.values() if x.get("response_model")}),
        "expected_record_count": len(rows), "record_count": len(saved),
        "api_call_count": sum(x["source"] == "glm_api" for x in saved.values()),
        "no_answer_rule_count": sum(x["source"] == "no_answer_rule" for x in saved.values()),
        "api_error_count": sum(x["source"] == "error" for x in saved.values()),
        "shortened_evidence_count": sum(x["evidence_truncated"] for x in saved.values()),
        "controls": {"total": len(controls), "correct": controls_correct,
                     "errors": sum(x["source"] == "error" for x in controls)},
        "steps": {}, "paired_vs_0": {},
    }
    # Providers may include nested token details; count the common totals explicitly.
    all_logged = [json.loads(line) for line in (output_dir / "results.jsonl").read_text().splitlines()]
    summary["usage"] = {name: sum(attempt.get("usage", {}).get(name, 0) or 0
                                for x in all_logged for attempt in x.get("attempts", []))
                        for name in ["prompt_tokens", "completion_tokens", "total_tokens"]}
    disagreements = []
    for step in sorted({step for step, _ in original}):
        part = [x for (s, _), x in saved.items() if s == step]
        complete = Counter(x.get("verdict", {}).get("complete", "error") for x in part)
        support = sum(x.get("verdict", {}).get("complete") == "yes" and
                      x["verdict"]["grounding"] == "supported" for x in part)
        dpart = [deepseek[(step, str(x["record_id"]))] for x in part if (step, str(x["record_id"])) in deepseek]
        confusion_original = Counter(f"{x['original_status']}->{x.get('verdict', {}).get('complete', 'error')}" for x in part)
        confusion_deepseek = Counter(f"{deepseek[(step, str(x['record_id']))]['verdict']['complete']}->{x.get('verdict', {}).get('complete', 'error')}"
                                     for x in part if (step, str(x["record_id"])) in deepseek)
        summary["steps"][str(step)] = {
            "tasks": len(part), "glm": {"complete_yes": complete["yes"], "complete_no": complete["no"],
            "complete_uncertain": complete["uncertain"], "errors": complete["error"],
            "complete_rate": complete["yes"] / len(part) if part else None,
            "complete_and_grounded_supported": support},
            "deepseek_complete_yes": sum(x["verdict"]["complete"] == "yes" for x in dpart),
            "original_success": sum(x["original_status"] == "success" for x in part),
            "original_vs_glm": dict(confusion_original), "deepseek_vs_glm": dict(confusion_deepseek),
        }
    for key, result in saved.items():
        d = deepseek.get(key, {})
        glm = result.get("verdict", {}).get("complete")
        if glm != d.get("verdict", {}).get("complete") or result["original_status"] == "error" or (
            (result["original_status"] == "success") != (glm == "yes")
        ):
            source = original[key]
            disagreements.append({"step": key[0], "record_id": key[1], "query": source["query"],
                                  "final_answer": source["final_answer"], "original": source["evaluation"],
                                  "deepseek": d.get("verdict"), "glm": result.get("verdict")})
    for step in sorted({s for s, _ in saved} - {0}):
        paired = Counter()
        for (s, rid), result in saved.items():
            if s != step or (0, rid) not in saved:
                continue
            before = saved[(0, rid)].get("verdict", {}).get("complete") == "yes"
            after = result.get("verdict", {}).get("complete") == "yes"
            paired["both_yes" if before and after else "lost" if before else "gained" if after else "both_no"] += 1
        summary["paired_vs_0"][str(step)] = dict(paired)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with (output_dir / "disagreements.jsonl").open("w") as handle:
        for row in sorted(disagreements, key=lambda x: (x["step"], x["record_id"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deepseek-dir", type=Path, default=Path("outputs/deepseek_v4_flash_audit_20261001"))
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 10, 20, 30])
    parser.add_argument("--only", help="Comma-separated step:record_id pairs for a pilot")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-evidence-chars", type=int, default=20000)
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, hashes = [], {}
    requested = set(args.only.split(",")) if args.only else None
    for step in args.steps:
        path = args.validation_dir / f"{step}.jsonl"
        hashes[str(step)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.open():
            row = json.loads(line)
            if requested is None or f"{step}:{row['record_id']}" in requested:
                rows.append((step, row))
    metadata = {
        "model_requested": MODEL, "api_url": API_URL, "request_settings": REQUEST_SETTINGS,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "input_sha256": hashes, "max_evidence_chars": args.max_evidence_chars,
        "rubric_shared_with": "scripts/evaluate_toolbench_deepseek.py",
        "parser_note": "Missing/null missing-items fields on complete=yes are normalized to []; string missing-items fields to a one-item list. Raw verdicts retained; completeness/grounding labels are unchanged.",
        "evaluation_scope": "Saved answers' completeness and consistency with logged tool observations; no new agent generations or live tool calls.",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = args.output_dir / "metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        for field in ["model_requested", "api_url", "request_settings", "prompt_sha256", "input_sha256", "max_evidence_chars"]:
            if previous[field] != metadata[field]:
                raise SystemExit(f"Cannot mix changed {field} with saved results; use a new output directory")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    key = getpass.getpass("Ark API key (hidden): ")
    if not key:
        raise SystemExit("No API key provided")
    if args.calibrate:
        cases = json.loads(Path("tests/fixtures/toolbench_judge_calibration.json").read_text())
        controls = [(0, {"record_id": str(i), "query": c["query"], "final_answer": c["answer"], "history": [],
                         "evaluation": {"status": c["expected"], "score": float(c["expected"] == "success")}})
                    for i, c in enumerate(cases)]
        run_batch(controls, args.output_dir / "calibration_results.jsonl", key, args.workers, args.max_evidence_chars)
    run_batch(rows, args.output_dir / "results.jsonl", key, args.workers, args.max_evidence_chars)
    summarize(args.output_dir, rows, args.deepseek_dir)


if __name__ == "__main__":
    main()
