"""Check the DeepSeek audit prompt against the existing labeled completeness controls."""

from __future__ import annotations

import argparse
import getpass
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.evaluate_toolbench_deepseek import audit_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("tests/fixtures/toolbench_judge_calibration.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    key = getpass.getpass("DeepSeek API key (hidden): ")
    if not key:
        raise SystemExit("No API key provided")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {}
        for index, case in enumerate(cases):
            row = {
                "record_id": str(index),
                "query": case["query"],
                "final_answer": case["answer"],
                "history": [],
                "evaluation": {"status": case["expected"], "score": 1.0 if case["expected"] == "success" else 0.0},
            }
            futures[pool.submit(audit_one, 0, row, key, 20000)] = (index, case)
        for future in as_completed(futures):
            index, case = futures[future]
            result = future.result()
            results.append({"case_index": index, "split": case["split"], "query": case["query"],
                            "answer": case["answer"], "expected": case["expected"], **result})
    results.sort(key=lambda r: r["case_index"])
    with args.output.open("w") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    verdicts = Counter((r["expected"], r.get("verdict", {}).get("complete")) for r in results)
    print(json.dumps({"total": len(results), "confusion": {str(k): v for k, v in verdicts.items()},
                      "errors": sum(r["source"] == "error" for r in results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
