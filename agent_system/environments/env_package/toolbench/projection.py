from __future__ import annotations

import json
import re


def _json_tool_call(value: str) -> tuple[str, str]:
    try:
        payload = json.loads(value)
        if isinstance(payload, dict):
            name = payload.get("name", "")
            arguments = payload.get("arguments", payload.get("parameters", {}))
            return str(name), json.dumps(arguments, ensure_ascii=False)
    except json.JSONDecodeError:
        pass
    return "", "{}"


def parse_tool_action(action: str) -> tuple[str, str]:
    text = str(action).strip()
    match = re.search(r"Action\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if match:
        input_match = re.search(r"Action\s*Input\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
        return match.group(1).strip(), input_match.group(1).strip() if input_match else "{}"
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return _json_tool_call(match.group(1))
    match = re.search(r"<function=([^>]+)>\s*(.*?)\s*</function>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip() or "{}"
    return "", "{}"


def toolbench_projection(actions: list[str]) -> tuple[list[str], list[int]]:
    parsed, valids = [], []
    for action in actions:
        name, action_input = parse_tool_action(action)
        parsed.append(action)
        valids.append(int(bool(name)))
    return parsed, valids
