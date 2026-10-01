"""StableToolBench disk cache and remote MirrorAPI tool execution.

This is an independently implemented client for the public protocol described in
https://github.com/THUNLP-MT/StableToolBench/tree/master/server . It follows the
cache naming and API-document fields of ``main_mirrorapi_cache.py``. The simulator
prompt preserves the upstream SFT_SYSTEM used by the deployed MirrorAPI models.
No training-trajectory responses are replayed and the upstream cache is read-only.
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import threading
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests


# StableToolBench SFT_SYSTEM, Apache-2.0; upstream commit:
# aa4ed9f4737ad98bd706663f01d63623c3427812/server/system_prompts.py
# Integration modified here; license reproduced in STABLETOOLBENCH_LICENSE.
SIMULATOR_SYSTEM_PROMPT = '''Imagine you are an API Server operating within a specialized tool, which contains a collection of distinct APIs. Your role is to deeply understand the function of each API based on their descriptions in the API documentation. As you receive specific inputs for individual API calls within this tool, analyze these inputs to determine their intended purpose. Your task is to craft a JSON formatted response that aligns with the expected output of the API. The JSON scheme is:
{
    "error": "",
    "response": ""
}

The error field should remain empty, indicating no errors in processing. The response field should contain the content you formulate based on the API's functionality and the input provided. Ensure that your responses are meaningful, directly addressing the API's intended functionality.\x20
The key is to maintain the JSON format's integrity while ensuring that your response is an accurate reflection of the API's intended output within the tool.
Please note that your answer should not contain anything other than a json format object, which should be parsable directly to json.
Note that:
- your response should contain rich information given the api input parameters.
- your response must be effective and have practical content.

API calls may fail for various reasons, such as invalid input parameters, authentication issues, or server errors. Your goal is to generate a response that accurately reflects the API's intended functionality, even if the input parameters are incorrect. Your response should be informative and relevant to the API's purpose, providing a clear and concise explanation of the expected output based on the input provided.
Here is an example:
API doc:
{
    "api_name": "List Languages",
    "api_description": "Get a list of currently supported languages. We are constantly adding more every few weeks.",
    "required_parameters": [],
    "optional_parameters": [],
    "tool_description": "Introducing our cutting-edge text to speech service, designed to provide you with the most realistic human-sounding voices at an affordable price. Our service is fast and reliable, delivering high-quality audio output in a matter of seconds. Additionally, we offer a wide range of languages and a variety of voice choices, so you can find the perfect fit for your project. Whether you need a voiceover for a video, an audiobook, or any other project, our text to speech service has you covered. Ex...",
    "tool_name": "TTSKraken",
    "tool_category": "Artificial_Intelligence_Machine_Learning"
}
Request:
    data = {
        "category": "Artificial_Intelligence_Machine_Learning",
        "tool_name": "TTSKraken",
        "api_name": "List Languages",
        "tool_input": "{}",
        "strip": "filter",
        }
Response:
    {
        "error": "",
        "response": "{"status":0,"msg":"Success","languages":["en","fr-fr","pt-br"]}"
    }
'''


class StableToolBenchError(RuntimeError):
    """An input, resource, transport or simulator-protocol failure, never a win."""

    def __init__(self, code: str, message: str, source: str = "stabletoolbench"):
        super().__init__(message)
        self.code = code
        self.source = source


def standardize(value: str) -> str:
    """Use StableToolBench's executable-name convention (including digit prefix)."""
    value = re.sub(r"[^\u4e00-\u9fa5^a-z^A-Z^0-9^_]", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    return f"get_{value}" if value and value[0].isdigit() else value


def change_name(value: str) -> str:
    return f"is_{value}" if value in {"from", "class", "return", "false", "true", "id", "and"} else value


def normalize_mapping(mapping: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return category (case preserved), bare tool name and executable API name."""
    if not isinstance(mapping, Mapping) or any(
        not isinstance(mapping.get(key), str) or not mapping[key].strip()
        for key in ("category", "tool_name", "api_name")
    ):
        raise StableToolBenchError("invalid_mapping", "Tool mapping requires category, tool_name and api_name.")
    # Category directory names retain capitalization, unlike tool/API names.
    category = re.sub(r"[ ,/]+", "_", mapping["category"])
    if category in {".", ".."} or "\\" in category or "\x00" in category:
        raise StableToolBenchError("invalid_mapping", "Invalid tool category.")
    suffix = f"_for_{category}"
    tool = mapping["tool_name"]
    if tool.lower().endswith(suffix.lower()):
        tool = tool[:-len(suffix)]
    tool = standardize(tool)
    api = change_name(standardize(mapping["api_name"]))
    # Some datasets store the exposed function name (API_for_tool) as api_name.
    for api_suffix in (f"_for_{standardize(mapping['tool_name'])}", f"_for_{tool}"):
        if api.endswith(api_suffix):
            api = api[:-len(api_suffix)]
            break
    api = change_name(api)
    if not tool or not api:
        raise StableToolBenchError("invalid_mapping", "Empty normalized tool or API name.")
    return category, tool, api


def _strict_json(value: Any) -> str:
    """Canonical JSON without Python equality's True == 1 or key-order hazards."""
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        for child in value.values():
            _strict_json(child)
    elif isinstance(value, list):
        for child in value:
            _strict_json(child)
    elif value is not None and type(value) not in (str, bool, int, float):
        raise ValueError("Value is not JSON compatible")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite values are not JSON compatible")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_loads(value: str):
    return json.loads(value, object_pairs_hook=_unique_pairs)


def parse_arguments(arguments: str | Mapping[str, Any]) -> dict[str, Any]:
    try:
        if not isinstance(arguments, (str, Mapping)):
            raise ValueError("Tool arguments must be a JSON object")
        parsed = _json_loads(arguments) if isinstance(arguments, str) else dict(arguments)
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments must be a JSON object")
        _strict_json(parsed)
    except (ValueError, TypeError) as exc:
        raise StableToolBenchError("invalid_arguments", f"Invalid tool arguments: {exc}") from exc
    return parsed


def validate_response(response: Any, source: str) -> dict[str, Any]:
    if isinstance(response, str):
        try:
            response = _json_loads(response)
        except (ValueError, TypeError) as exc:
            raise StableToolBenchError("invalid_response", "Tool response is not valid JSON.", source) from exc
    if not isinstance(response, dict) or not isinstance(response.get("error"), str) or "response" not in response:
        raise StableToolBenchError("invalid_response", "Tool response must contain error (string) and response.", source)
    try:
        _strict_json(response)
    except (ValueError, TypeError) as exc:
        raise StableToolBenchError("invalid_response", "Tool response contains non-JSON values.", source) from exc
    return {"error": response["error"], "response": response["response"]}


_AMBIGUOUS_CACHE_ENTRY = object()


class StableToolBenchClient:
    """Read exact disk-cache hits, otherwise query the configured MirrorAPI model.

    ``config`` is the ``env.toolbench`` mapping. No model is loaded locally.
    The semaphore bounds requests within one environment/client process.
    """

    def __init__(self, config: Mapping[str, Any]):
        self.backend = str(config.get("backend", "mirrorapi_cache")).lower().replace("-", "_")
        if self.backend not in {"mirrorapi", "mirrorapi_cache"}:
            raise ValueError("toolbench.backend must be mirrorapi_cache or mirrorapi")
        self.cache_enabled = bool(config.get("cache_enabled", True))
        self.cache_root = Path(str(config.get("cache_root", "data/StableToolBench/tool_response_cache"))).expanduser()
        self.tools_root = Path(str(config.get("tools_root", "data/StableToolBench/tools"))).expanduser()
        default_url = "http://10.8.176.56:8001/v1" if self.backend == "mirrorapi_cache" else "http://10.8.176.56:8000/v1"
        default_model = "MirrorAPI-Cache" if self.backend == "mirrorapi_cache" else "MirrorAPI"
        self.base_url = str(config.get(f"{self.backend}_url", default_url)).rstrip("/")
        self.model = str(config.get(f"{self.backend}_model", default_model))
        if not self.base_url.startswith(("http://", "https://")) or not self.model:
            raise ValueError("MirrorAPI requires an HTTP(S) base URL and a model name")
        self.request_timeout = float(config.get("request_timeout", 60))
        self.max_tokens = int(config.get("max_tokens", 2048))
        self.temperature = float(config.get("temperature", 0))
        self.max_concurrency = int(config.get("max_concurrency", 8))
        if self.request_timeout <= 0 or self.max_tokens <= 0 or self.max_concurrency <= 0:
            raise ValueError("MirrorAPI timeout, max_tokens and max_concurrency must be positive")
        self._semaphore = threading.BoundedSemaphore(self.max_concurrency)
        self._trust_env = bool(config.get("trust_env", False))
        self._api_key = str(config.get("api_key") or os.environ.get(str(config.get("api_key_env", "STABLETOOLBENCH_API_KEY"))) or "EMPTY")
        self._sessions: list[requests.Session] = []
        self._sessions_lock = threading.Lock()
        self._local = threading.local()
        # Bounded, lazy CPU caches: never preload the complete 467 MiB corpus.
        self._cache_index = lru_cache(maxsize=16)(self._read_cache_index)
        self._document = lru_cache(maxsize=128)(self._read_document)

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.trust_env = self._trust_env
            self._local.session = session
            with self._sessions_lock:
                self._sessions.append(session)
        return session

    @staticmethod
    def _read_json_file(path: Path, source: str):
        try:
            return _json_loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StableToolBenchError("invalid_resource", f"Cannot read JSON resource: {path}", source) from exc

    def _read_cache_index(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        cache = self._read_json_file(path, "disk_cache")
        if not isinstance(cache, dict):
            raise StableToolBenchError("invalid_resource", f"Cache is not an object: {path}", "disk_cache")
        index = {}
        for raw_key, response in cache.items():
            try:
                try:
                    arguments = _json_loads(raw_key)
                except ValueError:
                    # Official files use str(dict), including single quotes/True/None.
                    arguments = ast.literal_eval(raw_key)
                if not isinstance(arguments, dict):
                    continue
                key = _strict_json(arguments)
            except (ValueError, TypeError, SyntaxError):
                continue
            if key in index and index[key] != response:
                index[key] = _AMBIGUOUS_CACHE_ENTRY
            elif key not in index:
                index[key] = response
        return index

    def _read_document(self, category: str, tool: str, api: str) -> dict[str, Any]:
        path = self.tools_root / category / f"{tool}.json"
        document = self._read_json_file(path, self.backend)
        if not isinstance(document, dict) or not isinstance(document.get("api_list"), list):
            raise StableToolBenchError("invalid_resource", f"Tool document lacks api_list: {path}", self.backend)
        matches = [item for item in document["api_list"] if isinstance(item, dict)
                   and change_name(standardize(str(item.get("name", "")))) == api]
        if len(matches) != 1:
            raise StableToolBenchError("unknown_api", f"Expected one API document for {category}/{tool}/{api}; found {len(matches)}.", self.backend)
        return self._api_document(category, tool, api, matches[0], document.get("tool_description") or "")

    def _api_document(self, category: str, tool: str, api: str, info: Any, description: str) -> dict[str, Any]:
        if not isinstance(info, dict) or change_name(standardize(str(info.get("name") or info.get("api_name") or ""))) != api:
            raise StableToolBenchError("invalid_resource", f"API schema does not match {category}/{tool}/{api}.", self.backend)
        required = info.get("required_parameters") or []
        optional = info.get("optional_parameters") or []
        if not isinstance(required, list) or not isinstance(optional, list):
            raise StableToolBenchError("invalid_resource", f"API parameters are not lists: {category}/{tool}/{api}", self.backend)
        return {
            "api_name": api,
            "api_description": info.get("description") or info.get("api_description") or "",
            "required_parameters": required,
            "optional_parameters": optional,
            "tool_description": description,
            "tool_name": tool,
            "tool_category": category,
        }

    def execute(self, mapping: Mapping[str, Any], arguments: str | Mapping[str, Any]) -> tuple[dict[str, Any], str]:
        category, tool, api = normalize_mapping(mapping)
        parsed = parse_arguments(arguments)
        if self.cache_enabled:
            path = self.cache_root / category / f"{tool}_for_{category}" / f"{api}.json"
            cache = self._cache_index(path)
            key = _strict_json(parsed)
            if key in cache:
                if cache[key] is _AMBIGUOUS_CACHE_ENTRY:
                    raise StableToolBenchError("ambiguous_cache", f"Conflicting responses for the requested cache key: {path}", "disk_cache")
                return validate_response(cache[key], "disk_cache"), "disk_cache"
        if mapping.get("api_schema_json"):
            try:
                schema = _json_loads(mapping["api_schema_json"])
            except (ValueError, TypeError) as exc:
                raise StableToolBenchError("invalid_resource", "Embedded API schema is not valid JSON.", self.backend) from exc
            api_doc = self._api_document(category, tool, api, schema, mapping.get("tool_description") or "")
        else:
            api_doc = self._document(category, tool, api)
        request = {key: value for key, value in parsed.items() if key != "toolbench_key"}
        # The field set and API doc / Request structure match the official server.
        # Python representations also match its training-time formatting.
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SIMULATOR_SYSTEM_PROMPT},
                {"role": "user", "content": f"API doc:\n{api_doc}\n\nRequest:\n{request}"},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": 42,
        }
        if not self._semaphore.acquire(timeout=self.request_timeout):
            raise StableToolBenchError("queue_timeout", "Timed out waiting for a MirrorAPI request slot.", self.backend)
        try:
            try:
                result = self._session().post(
                    f"{self.base_url}/chat/completions", json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"}, timeout=self.request_timeout,
                )
                result.raise_for_status()
                completion = result.json()
            except requests.Timeout as exc:
                raise StableToolBenchError("request_timeout", "MirrorAPI request timed out.", self.backend) from exc
            except requests.RequestException as exc:
                # Do not put headers, input arguments or auth credentials in errors.
                status = getattr(getattr(exc, "response", None), "status_code", None)
                detail = f" (HTTP {status})" if status is not None else ""
                raise StableToolBenchError("request_failed", f"MirrorAPI request failed{detail}.", self.backend) from exc
            except ValueError as exc:
                raise StableToolBenchError("invalid_completion", "MirrorAPI HTTP response is not JSON.", self.backend) from exc
        finally:
            self._semaphore.release()
        try:
            choice = completion["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            if not isinstance(content, str) or finish_reason in {"length", "content_filter"}:
                raise ValueError("Missing or truncated completion")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise StableToolBenchError("invalid_completion", "MirrorAPI returned a missing or truncated completion.", self.backend) from exc
        return validate_response(content, self.backend), self.backend

    def close(self):
        with self._sessions_lock:
            for session in self._sessions:
                session.close()
            self._sessions.clear()
        self._cache_index.cache_clear()
        self._document.cache_clear()
