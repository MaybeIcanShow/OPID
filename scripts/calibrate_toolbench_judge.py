"""Run explicit complete/incomplete answer controls against a configured judge.

Run as ``python -m scripts.calibrate_toolbench_judge --output PATH``. Expected
labels are used only for reporting and are never included in judge requests.
This small control set checks completeness; it does not certify benchmark accuracy.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from omegaconf import OmegaConf

from agent_system.environments.env_package.toolbench.evaluation import StableToolBenchEvaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="tests/fixtures/toolbench_judge_calibration.json")
    parser.add_argument("--mode", choices=("fac_prompt", "fac_evidence"), default="fac_evidence")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cases = json.loads(Path(args.cases).read_text())
    config = OmegaConf.to_container(OmegaConf.load("verl/trainer/config/ppo_trainer.yaml").env.toolbench.evaluator)
    config["mode"] = args.mode
    evaluator = StableToolBenchEvaluator(config)

    def evaluate(case):
        return {**case, "evaluation": asdict(evaluator.evaluate(case["query"], case["answer"]))}

    with ThreadPoolExecutor(max_workers=config["max_concurrency"]) as pool:
        results = list(pool.map(evaluate, cases))
    summary = {"judge_model": evaluator.model, "judge_mode": evaluator.mode}
    for split in sorted({r["split"] for r in results}):
        rows = [r for r in results if r["split"] == split]
        summary[split] = {"total": len(rows), "correct": sum(r["evaluation"]["status"] == r["expected"] for r in rows),
                          "errors": sum(r["evaluation"]["status"] == "error" for r in rows)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as file:
        json.dump({"summary": summary, "results": results}, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
