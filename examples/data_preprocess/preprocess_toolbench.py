"""Prepare a reproducible ToolBench split for agentic GRPO."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from datasets import Dataset

ACTION_RE = re.compile(
    r"Action:\s*(?P<name>[^\n]+?)\s*\nAction Input:\s*(?P<input>.*?)(?=\n(?:Thought:|Action:|$)|$)",
    flags=re.IGNORECASE | re.DOTALL,
)


def standardize(value: str) -> str:
    value = re.sub(r"[^\u4e00-\u9fa5^a-z^A-Z^0-9^_]", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    if value and value[0].isdigit():
        value = "get_" + value
    return value


def change_name(value: str) -> str:
    return "is_" + value if value in {"from", "class", "return", "false", "true", "id", "and"} else value


def load_tool_index(tool_root: Path) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for category_dir in sorted(tool_root.iterdir()):
        if not category_dir.is_dir():
            continue
        for tool_file in category_dir.glob("*.json"):
            try:
                tool = json.loads(tool_file.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            tool_name = tool.get("tool_name", tool_file.stem)
            for api in tool.get("api_list", []):
                api_name = api.get("name", "")
                if api_name:
                    function_name = f"{change_name(standardize(api_name))}_for_{standardize(tool_name)}"
                    index[function_name] = {
                        "category": category_dir.name,
                        "tool_name": tool_name,
                        "api_name": api_name,
                    }
    return index


def convert_record(record: dict[str, Any], tool_index: dict[str, dict[str, str]], idx: int) -> dict[str, Any]:
    conversations = record.get("conversations", [])
    system_prompt = next((item.get("value", "") for item in conversations if item.get("from") == "system"), "")
    user_prompt = next((item.get("value", "") for item in conversations if item.get("from") == "user"), "")
    user_prompt = user_prompt or str(record.get("id", ""))
    actions: list[dict[str, str]] = []
    function_responses: dict[str, list[str]] = {}
    for position, item in enumerate(conversations):
        if item.get("from") != "assistant":
            continue
        matches = list(ACTION_RE.finditer(item.get("value", "")))
        for match in matches:
            name = match.group("name").strip()
            action_input = match.group("input").strip()
            response = ""
            if position + 1 < len(conversations) and conversations[position + 1].get("from") == "function":
                response = str(conversations[position + 1].get("value", ""))[:12000]
            actions.append({"name": name, "input": action_input})
            if response:
                function_responses.setdefault(name, []).append(response)

    function_map: dict[str, dict[str, str]] = {}
    for name in re.findall(r"['\"]name['\"]:\s*['\"]([^'\"]+)['\"]", system_prompt):
        if name in tool_index:
            function_map[name] = tool_index[name]
    for action in actions:
        if action["name"] in tool_index:
            function_map[action["name"]] = tool_index[action["name"]]

    return {
        "data_source": "toolbench",
        "prompt": [{"role": "user", "content": user_prompt.strip()}],
        "ability": "agent",
        "env_kwargs": {
            "query": user_prompt.strip(),
            "system_prompt": system_prompt,
            "function_map": function_map,
            "offline_tool_responses": function_responses,
            "record_id": str(record.get("id", idx)),
        },
        "extra_info": {"split": "train", "index": idx},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/ToolBench/toolllama_G123_dfs_train.json")
    parser.add_argument("--tool-root", default="data/StableToolBench/server/tools")
    parser.add_argument("--output-dir", default="data/toolbench_processed")
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--val-size", type=int, default=128)
    args = parser.parse_args()
    source = Path(args.source).expanduser()
    with source.open() as handle:
        raw_records = json.load(handle)
    needed = args.train_size + args.val_size
    if len(raw_records) < needed:
        raise ValueError(f"source has {len(raw_records)} rows, need {needed}")
    tool_index = load_tool_index(Path(args.tool_root).expanduser())
    print(f"indexed {len(tool_index)} StableToolBench APIs")
    converted = [convert_record(record, tool_index, idx) for idx, record in enumerate(raw_records[:needed])]
    for row in converted[args.train_size:]:
        row["extra_info"]["split"] = "test"
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(converted[: args.train_size]).to_parquet(str(output_dir / "train.parquet"))
    Dataset.from_list(converted[args.train_size :]).to_parquet(str(output_dir / "test.parquet"))
    (output_dir / "metadata.json").write_text(json.dumps({"source": str(source), "train_size": args.train_size, "val_size": args.val_size, "tool_index_size": len(tool_index)}, indent=2))
    print(f"wrote {args.train_size} train and {args.val_size} validation rows to {output_dir}")


if __name__ == "__main__":
    main()
