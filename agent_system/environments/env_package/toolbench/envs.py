from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from omegaconf import DictConfig

from .evaluation import StableToolBenchEvaluator
from .projection import parse_tool_action
from .stabletoolbench import StableToolBenchClient, StableToolBenchError, parse_arguments

logger = logging.getLogger(__name__)


class ToolBenchMultiProcessEnv:
    """StableToolBench tool execution with an independent final-answer evaluator."""

    def __init__(self, seed: int, env_num: int, group_n: int, is_train: bool, env_config: DictConfig):
        self.batch_size = env_num * group_n
        self.max_steps = int(env_config.max_steps)
        self.is_train = is_train
        self.client = StableToolBenchClient(env_config.toolbench)
        self.evaluator = StableToolBenchEvaluator(env_config.toolbench.get("evaluator", {}))
        self.envs: list[dict[str, Any]] = []
        self._executor = ThreadPoolExecutor(max_workers=min(self.batch_size, 64))
        if not self.evaluator.configured:
            logger.warning(
                "No StableToolBench answer evaluator configured: task outcomes will be "
                "unscored, Finish has no success bonus, and task success metrics are omitted."
            )

    @staticmethod
    def _initial_observation(record: dict[str, Any]) -> str:
        system = str(record.get("system_prompt") or "").strip()
        query = str(record.get("query") or "").strip()
        return f"{system}\n\nUser query:\n{query}\nBegin!" if system else f"Task: {query}\nBegin!"

    def _finish(self, record, final_answer, reason, valid=True):
        state = record["_state"]
        state["done"] = True
        state["final_answer"] = final_answer
        state["termination_reason"] = reason
        result = self.evaluator.evaluate(str(record.get("query", "")), final_answer)
        state["evaluation"] = {
            "status": result.status, "score": result.score, "reason": result.reason,
            "judge_model": self.evaluator.model,
        }
        # No submitted answer within the budget is a protocol failure.
        # Missing judge configuration remains explicitly unscored.
        if not final_answer and self.evaluator.configured:
            state["evaluation"] = {"status": "failure", "score": 0.0, "reason": "no_final_answer"}
        evaluation = state["evaluation"]
        info = {
            "won": evaluation["status"] == "success",
            "tool_calling": False,
            "is_action_valid": int(valid),
            "data_source": record.get("benchmark", "toolbench"),
            "evaluation_status": evaluation["status"],
            "termination_reason": reason,
        }
        # An unscored submission is not a successful task.
        reward = float(evaluation["score"]) if evaluation["score"] is not None else 0.0
        return "", reward, True, info

    def _step_one(self, index: int, action: str):
        record = self.envs[index]
        state = record["_state"]
        if state["done"]:
            return "", 0.0, True, {
                "won": state.get("evaluation", {}).get("status") == "success",
                "tool_calling": False, "is_action_valid": 0,
                "data_source": record.get("benchmark", "toolbench"),
                "evaluation_status": state.get("evaluation", {}).get("status", "unscored"),
            }
        state["step"] += 1
        name, action_input = parse_tool_action(action)
        try:
            arguments = parse_arguments(action_input)
        except StableToolBenchError:
            name, arguments = "", {}

        if name.lower() == "finish":
            return_type = arguments.get("return_type")
            answer = arguments.get("final_answer", "")
            if return_type == "give_answer" and isinstance(answer, str) and answer.strip():
                state["answer_submitted"] = True
                return self._finish(record, answer.strip(), "give_answer")
            if return_type == "give_up_and_restart":
                return self._finish(record, "", "give_up")
            name = ""  # A malformed Finish must not earn a success reward.

        mapping = (record.get("function_map") or {}).get(name) if name else None
        if not isinstance(mapping, dict) or not mapping:
            observation = (
                "Invalid action. Use one available tool name with a JSON object as Action Input, "
                "or Finish with return_type and a nonempty final_answer."
            )
            reward = -0.05
            info = {"tool_calling": False, "is_action_valid": 0, "response_source": "invalid_action"}
        else:
            state["tool_calls"] += 1
            try:
                response, source = self.client.execute(mapping, arguments)
                info = {"tool_calling": True, "is_action_valid": 1, "response_source": source}
                reward = -0.05 if response.get("error") else 0.0
            except StableToolBenchError as exc:
                source = "unavailable"
                response = {"error": str(exc), "response": ""}
                info = {
                    "tool_calling": True, "is_action_valid": 1,
                    "response_source": source, "response_error_code": exc.code,
                }
                reward = -0.05
            state["response_sources"][source] = state["response_sources"].get(source, 0) + 1
            observation = (
                f"Tool response ({source}):\n{json.dumps(response, ensure_ascii=False)}"
                "\n\nContinue with one Thought/Action/Action Input step."
            )
        state["history"].append({"action": action, "observation": observation})
        info.update({"won": False, "data_source": record.get("benchmark", "toolbench")})
        done = state["step"] >= self.max_steps
        if done:
            _, final_reward, _, terminal_info = self._finish(record, "", "max_steps")
            reward += final_reward
            info.update({k: v for k, v in terminal_info.items()
                         if k not in {"tool_calling", "is_action_valid"}})
        return observation, reward, done, info

    def reset(self, kwargs: list[dict[str, Any]] | None):
        kwargs = [] if kwargs is None else list(kwargs)
        if len(kwargs) > self.batch_size:
            raise ValueError(f"received {len(kwargs)} records, expected at most {self.batch_size}")
        self.envs = [dict(item) for item in kwargs]
        for record in self.envs:
            record["_state"] = {
                "step": 0, "done": False, "history": [], "tool_calls": 0,
                "response_sources": {}, "final_answer": "", "answer_submitted": False,
                "termination_reason": "", "evaluation": {"status": "unscored", "score": None, "reason": "not_finished"},
            }
        return [self._initial_observation(record) for record in self.envs], [
            {"data_source": record.get("benchmark", "toolbench"), "record_id": record.get("record_id", "")}
            for record in self.envs
        ]

    def step(self, actions: list[str]):
        if len(actions) != len(self.envs):
            raise ValueError(f"received {len(actions)} actions for {len(self.envs)} tasks")
        results = list(self._executor.map(lambda pair: self._step_one(*pair), enumerate(actions)))
        observations, rewards, dones, infos = zip(*results)
        return list(observations), list(rewards), list(dones), list(infos)

    def get_evaluation_records(self):
        return [
            {
                "record_id": record.get("record_id", ""),
                "eval_group": record.get("eval_group", ""),
                "query": record.get("query", ""),
                "final_answer": record["_state"]["final_answer"],
                "answer_submitted": record["_state"]["answer_submitted"],
                "termination_reason": record["_state"]["termination_reason"],
                "tool_calls": record["_state"]["tool_calls"],
                "response_sources": dict(record["_state"]["response_sources"]),
                "evaluation": dict(record["_state"]["evaluation"]),
                "history": list(record["_state"]["history"]),
            }
            for record in self.envs
        ]

    def close(self):
        self._executor.shutdown(wait=True)
        self.client.close()


def build_toolbench_envs(seed: int, env_num: int, group_n: int, is_train: bool, env_config: DictConfig):
    return ToolBenchMultiProcessEnv(seed, env_num, group_n, is_train, env_config)
