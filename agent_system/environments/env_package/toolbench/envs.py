from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import requests
from omegaconf import DictConfig

from .projection import parse_tool_action


def _standardize(value: str) -> str:
    value = re.sub(r"[^\u4e00-\u9fa5^a-z^A-Z^0-9^_]", "_", value)
    return re.sub(r"_+", "_", value).strip("_").lower()


class ToolBenchMultiProcessEnv:
    """Threaded ToolBench environment with offline, cache, and service fallbacks."""

    def __init__(self, seed: int, env_num: int, group_n: int, is_train: bool, env_config: DictConfig):
        super().__init__()
        self.batch_size = env_num * group_n
        self.max_steps = int(env_config.max_steps)
        self.cache_root = Path(str(env_config.toolbench.cache_root)).expanduser()
        self.service_url = str(env_config.toolbench.service_url)
        self.request_timeout = float(env_config.toolbench.request_timeout)
        self.envs: list[dict[str, Any] | None] = [None] * self.batch_size
        self._executor = ThreadPoolExecutor(max_workers=min(self.batch_size, 64))

    @staticmethod
    def _initial_observation(record: dict[str, Any]) -> str:
        system = record.get("system_prompt", "").strip()
        query = record.get("query", "").strip()
        return f"{system}\n\nUser query:\n{query}\nBegin!" if system else f"Task: {query}\nBegin!"

    def _cache_response(self, record: dict[str, Any], action_name: str, action_input: str) -> str | None:
        mapping = record.get("function_map", {}).get(action_name)
        if not mapping:
            return None
        path = self.cache_root / _standardize(mapping["category"]) / _standardize(mapping["tool_name"]) / f"{_standardize(mapping['api_name'])}.json"
        if not path.exists():
            return None
        try:
            cache = json.loads(path.read_text())
            parsed_input = json.loads(action_input)
        except (OSError, json.JSONDecodeError):
            return None
        for key in (str(parsed_input), action_input, json.dumps(parsed_input, ensure_ascii=False, sort_keys=True)):
            if key in cache:
                return json.dumps(cache[key], ensure_ascii=False)
        return None

    def _request_service(self, record: dict[str, Any], action_name: str, action_input: str) -> str | None:
        mapping = record.get("function_map", {}).get(action_name)
        if not mapping or not self.service_url:
            return None
        payload = {"category": mapping["category"], "tool_name": mapping["tool_name"], "api_name": mapping["api_name"], "tool_input": action_input, "strip": "truncate", "toolbench_key": ""}
        try:
            response = requests.post(self.service_url, json=payload, timeout=self.request_timeout)
            if response.ok:
                return response.text
        except requests.RequestException:
            pass
        return None

    def _step_one(self, index: int, action: str):
        record = self.envs[index]
        if record is None:
            return "", 0.0, True, {"won": False, "tool_calling": False, "data_source": "toolbench"}
        name, action_input = parse_tool_action(action)
        state = record.setdefault("_state", {"step": 0, "used": {}})
        state["step"] += 1
        if name.lower() == "finish" or name.lower().startswith("finish->"):
            try:
                success = json.loads(action_input).get("return_type") == "give_answer"
            except json.JSONDecodeError:
                success = "give_answer" in action_input
            return "", float(success), True, {"won": bool(success), "tool_calling": False, "is_action_valid": 1, "data_source": "toolbench"}
        if not name:
            done = state["step"] >= self.max_steps
            return "Invalid action format. Use Action and Action Input.", -0.05, done, {"won": False, "tool_calling": False, "is_action_valid": 0, "data_source": "toolbench"}

        responses = record.get("offline_tool_responses", {}).get(name, [])
        used = int(state["used"].get(name, 0))
        response = responses[used] if used < len(responses) else None
        state["used"][name] = used + 1
        source = "offline"
        if response is None:
            response = self._cache_response(record, name, action_input)
            source = "cache"
        if response is None:
            response = self._request_service(record, name, action_input)
            source = "service"
        if response is None:
            response = json.dumps({"error": "tool response unavailable", "response": ""})
            source = "missing"
        done = state["step"] >= self.max_steps
        reward = 0.0 if source != "missing" else -0.05
        return f"Tool response ({source}):\n{response}\n\nContinue with one Thought/Action/Action Input step.", reward, done, {"won": False, "tool_calling": True, "is_action_valid": 1, "data_source": "toolbench", "response_source": source}

    def reset(self, kwargs: list[dict[str, Any]] | None):
        kwargs = [] if kwargs is None else list(kwargs)
        if len(kwargs) > self.batch_size:
            raise ValueError(f"received {len(kwargs)} records, expected {self.batch_size}")
        kwargs += [{"query": "", "system_prompt": ""}] * (self.batch_size - len(kwargs))
        self.envs = [dict(item) for item in kwargs]
        for record in self.envs:
            record["_state"] = {"step": 0, "used": {}}
        return [self._initial_observation(record) for record in self.envs], [{"data_source": "toolbench", "record_id": record.get("record_id", "")} for record in self.envs]

    def step(self, actions: list[str]):
        results = [future.result() for future in [self._executor.submit(self._step_one, i, action) for i, action in enumerate(actions)]]
        observations, rewards, dones, infos = zip(*results)
        return list(observations), list(rewards), list(dones), list(infos)

    def close(self):
        self._executor.shutdown(wait=True)


def build_toolbench_envs(seed: int, env_num: int, group_n: int, is_train: bool, env_config: DictConfig):
    return ToolBenchMultiProcessEnv(seed, env_num, group_n, is_train, env_config)
