from __future__ import annotations

import json
import re


def _json_tool_call(value: str) -> tuple[str, str]:
    try:
        payload = json.loads(value)
        if isinstance(payload, dict):
            name = payload.get("name", "")
            arguments = payload.get("arguments", payload.get("parameters", {}))
            if isinstance(name, str) and name.strip() and isinstance(arguments, dict):
                return name.strip(), json.dumps(arguments, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        pass
    return "", "{}"


def parse_tool_action(action: str) -> tuple[str, str]:
    text = str(action).strip()
    # An action quoted in reasoning is not an executed tool call.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    elif "<think>" in text:
        return "", "{}"
    action_count = len(re.findall(r"^\s*Action\s*:", text, flags=re.IGNORECASE | re.MULTILINE))
    xml_count = len(re.findall(r"<tool_call>|<function=", text, flags=re.IGNORECASE))
    if action_count + xml_count != 1:
        return "", "{}"
    matches = list(re.finditer(r"^\s*Action\s*:\s*([^\n]+)", text, flags=re.IGNORECASE | re.MULTILINE))
    if matches:
        if len(matches) != 1:
            return "", "{}"
        match = matches[0]
        input_match = re.search(r"^\s*Action\s*Input\s*:\s*(.*)", text[match.end():], flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if input_match:
            return match.group(1).strip(), input_match.group(1).strip()
        return "", "{}"
    calls = list(re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.IGNORECASE | re.DOTALL))
    if len(calls) == 1:
        return _json_tool_call(calls[0].group(1))
    calls = list(re.finditer(r"<function=([^>]+)>\s*(.*?)\s*</function>", text, flags=re.IGNORECASE | re.DOTALL))
    if len(calls) == 1:
        return calls[0].group(1).strip(), calls[0].group(2).strip() or "{}"
    return "", "{}"


def toolbench_projection(actions: list[str]) -> tuple[list[str], list[int]]:
    parsed, valids = [], []
    for action in actions:
        name, action_input = parse_tool_action(action)
        parsed.append(action)
        try:
            valid = bool(name) and isinstance(json.loads(action_input), dict)
        except (ValueError, TypeError):
            valid = False
        valids.append(int(valid))
    return parsed, valids
