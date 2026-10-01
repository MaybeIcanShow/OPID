"""Prepare deduplicated ToolBench training and pinned StableToolBench validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import urllib.request
from collections import Counter
from functools import lru_cache
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable

from datasets import Dataset

OFFICIAL_REPOSITORY = "https://github.com/THUNLP-MT/StableToolBench"
OFFICIAL_REVISION = "aa4ed9f4737ad98bd706663f01d63623c3427812"
EVAL_GROUPS = ("G1_instruction", "G1_category", "G1_tool", "G2_instruction", "G2_category", "G3_instruction")
OFFICIAL_FILE_SHA256 = {
    "G1_instruction": "500aa0105ed451538448f6ea04f10661536b013f3e1a293840fe401688160e7c",
    "G1_category": "123cb9073c8b6e6473df91c8a51170277087e1c263eba2eb6077170edbe849c3",
    "G1_tool": "05d0c180bbec9d060fd267cc79399bda05beddd0a6d0de68b5488926a9029a1a",
    "G2_instruction": "82d569cd0fc8fc742a15a3011fb5cd6a8b11e0a5b344c884602347439f43cad7",
    "G2_category": "71e18112f7940934f159d623b1c2b4740b1b4dc58a0b596c63b8151abbf8267e",
    "G3_instruction": "f923d1a9452646bd1a415b266117d90675d0f8fc04cf014db58d3cd78c25634f",
}
ACTION_RE = re.compile(
    r"Action:\s*(?P<name>[^\n]+?)\s*\nAction Input:\s*(?P<input>.*?)(?=\n(?:Thought:|Action:|$)|$)",
    flags=re.IGNORECASE | re.DOTALL,
)
FUNCTION_RE = re.compile(r"['\"]name['\"]:\s*['\"]([^'\"]+)['\"]")


def standardize(value: str) -> str:
    value = re.sub(r"[^\u4e00-\u9fa5^a-z^A-Z^0-9^_]", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    return "get_" + value if value and value[0].isdigit() else value


def change_name(value: str) -> str:
    return "is_" + value if value in {"from", "class", "return", "false", "true", "id", "and"} else value


def clean_query(value: str) -> str:
    # ToolLLaMA's conversation wrapper is not part of the benchmark question.
    return re.sub(r"\s*\nBegin!\s*$", "", value).strip()


def normalize_query(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", clean_query(value)).casefold().split())


def record_query(record: dict[str, Any]) -> str:
    user = next((item.get("value", "") for item in record.get("conversations", []) if item.get("from") == "user"), "")
    return clean_query(user or record.get("query", "") or str(record.get("id", "")))


def initial_observation(system_prompt: str, query: str) -> str:
    # Keep identical to ToolBenchMultiProcessEnv._initial_observation so prompt
    # length filtering measures the real initial context, including all tools.
    from agent_system.environments.env_package.toolbench.protocol import initial_observation as build_initial
    return build_initial(system_prompt, query)


def _identity(category: str, tool_name: str, api_name: str) -> str:
    return f"{category}/{standardize(tool_name)}/{change_name(standardize(api_name))}"


def load_tool_index(tool_root: Path) -> dict[str, dict[str, str]]:
    if not tool_root.is_dir():
        raise FileNotFoundError(f"tool documentation directory does not exist: {tool_root}")
    index: dict[str, dict[str, str]] = {}
    for tool_file in sorted(tool_root.glob("*/*.json")):
        tool = json.loads(tool_file.read_text())
        tool_name = tool.get("tool_name", tool_file.stem)
        for api in tool.get("api_list", []):
            api_name = api.get("name", "")
            if not api_name:
                continue
            schema = {
                "category_name": tool_file.parent.name, "tool_name": tool_name,
                "api_name": api_name, "api_description": api.get("description", ""),
                "required_parameters": api.get("required_parameters", []),
                "optional_parameters": api.get("optional_parameters", []),
                "method": api.get("method", "GET"),
            }
            mapping = {
                "category": tool_file.parent.name, "tool_name": tool_name, "api_name": api_name,
                "tool_description": str(tool.get("tool_description", "")),
                "api_schema_json": json.dumps(schema, ensure_ascii=False),
                "tool_file_name": tool_file.stem,
            }
            # The official function name uses the tool filename and keeps the
            # last 64 characters. Retain legacy aliases for training records.
            for alias in {tool_name, tool_file.stem}:
                name = f"{change_name(standardize(api_name))}_for_{standardize(alias)}"
                index[name] = mapping
                index[name[-64:]] = mapping
                index[_identity(tool_file.parent.name, alias, api_name)] = mapping
    return index


def convert_record(record: dict[str, Any], tool_index: dict[str, dict[str, str]], idx: int) -> dict[str, Any]:
    """Convert legacy ToolLLaMA training records; retain the public call shape."""
    conversations = record.get("conversations", [])
    system_prompt = next((item.get("value", "") for item in conversations if item.get("from") == "system"), "")
    query = record_query(record)
    function_responses: dict[str, list[str]] = {}
    names = set(FUNCTION_RE.findall(system_prompt))
    for position, item in enumerate(conversations):
        if item.get("from") != "assistant":
            continue
        for match in ACTION_RE.finditer(item.get("value", "")):
            name = match.group("name").strip()
            names.add(name)
            if position + 1 < len(conversations) and conversations[position + 1].get("from") == "function":
                response = str(conversations[position + 1].get("value", ""))[:12000]
                if response:
                    function_responses.setdefault(name, []).append(response)
    function_map = {name: tool_index[name] for name in sorted(names) if name in tool_index}
    return {
        "data_source": "toolbench", "prompt": [{"role": "user", "content": initial_observation(system_prompt, query)}],
        "ability": "agent",
        "env_kwargs": {
            "query": query, "system_prompt": system_prompt, "function_map": function_map,
            "offline_tool_responses": function_responses, "record_id": str(record.get("id", idx)),
            "benchmark": "toolbench", "split": "train",
        },
        "extra_info": {"split": "train", "index": idx},
    }


def _function_definition(api: dict[str, Any], name: str) -> dict[str, Any]:
    required_keys = {"api_name", "api_description", "required_parameters", "optional_parameters"}
    if missing := required_keys - api.keys():
        raise ValueError(f"incomplete official API documentation for {name}: missing {sorted(missing)}")
    properties, required, optional, original_names = {}, [], [], {}
    types = {"NUMBER": "integer", "INTEGER": "integer", "STRING": "string", "BOOLEAN": "boolean", "OBJECT": "object", "ARRAY": "array"}
    for group, names in (("required_parameters", required), ("optional_parameters", optional)):
        for parameter in api[group]:
            parameter_name = change_name(standardize(parameter["name"]))
            if parameter_name in properties:
                if original_names[parameter_name] != parameter["name"]:
                    raise ValueError(f"ambiguous normalized parameter {parameter_name} in {name}")
                # A few official schemas repeat the same parameter with different
                # examples. Keep one property and preserve each example.
                previous = properties[parameter_name]
                description = str(parameter.get("description", ""))
                if description and description not in previous["description"]:
                    previous["description"] += "\n" + description
                example = parameter.get("default")
                if example not in (None, "", previous.get("example_value")):
                    previous.setdefault("examples", [previous["example_value"]] if "example_value" in previous else []).append(example)
                continue
            original_names[parameter_name] = parameter["name"]
            properties[parameter_name] = {
                "type": types.get(str(parameter.get("type", "STRING")).upper(), "string"),
                "description": str(parameter.get("description", "")),
            }
            if parameter.get("default") not in (None, ""):
                properties[parameter_name]["example_value"] = parameter["default"]
            names.append(parameter_name)
    return {"name": name, "description": api["api_description"], "parameters": {"type": "object", "properties": properties, "required": required, "optional": optional}}


def convert_eval_record(record: dict[str, Any], tool_index: dict[str, dict[str, str]], group: str, idx: int) -> dict[str, Any]:
    query = clean_query(record["query"])
    if not query or not record.get("api_list") or "query_id" not in record:
        raise ValueError(f"invalid StableToolBench record in {group} at index {idx}")
    functions, function_map, tool_descriptions = [], {}, {}
    missing_local_docs = 0
    for api in record["api_list"]:
        local = tool_index.get(_identity(api["category_name"], api["tool_name"], api["api_name"]))
        missing_local_docs += int(local is None)
        tool_file_name = local["tool_file_name"] if local else standardize(api["tool_name"])
        name = f"{change_name(standardize(api['api_name']))}_for_{tool_file_name}"
        if name in function_map:
            raise ValueError(f"duplicate function name {name} in official query {record['query_id']}")
        functions.append(_function_definition(api, name))
        function_map[name] = {
            "category": api["category_name"], "tool_name": api["tool_name"], "api_name": api["api_name"],
            "tool_description": local.get("tool_description", "") if local else "",
            # Retain official schemas in metadata, never demonstration responses.
            "api_schema_json": json.dumps(api, ensure_ascii=False), "tool_file_name": tool_file_name,
        }
        if local and local.get("tool_description"):
            tool_descriptions[tool_file_name] = local["tool_description"]
    functions.append({
        "name": "Finish", "description": "Finish with a complete final answer, or give up if the task cannot be completed.",
        "parameters": {"type": "object", "properties": {
            "return_type": {"type": "string", "enum": ["give_answer", "give_up_and_restart"]},
            "final_answer": {"type": "string"}}, "required": ["return_type"]},
    })
    system_prompt = (
        "Use the available tools to complete the user's task. At each step output:\n"
        "Action: one function name\nAction Input: a JSON object\n"
        "Wait for the tool result before choosing the next action. Always end with Finish. "
        "Use return_type give_answer and a complete final_answer only when the task is completed; "
        "otherwise use give_up_and_restart.\n"
        "Tools:\n" + json.dumps(tool_descriptions, ensure_ascii=False) +
        "\nAvailable functions:\n" + json.dumps(functions, ensure_ascii=False)
    )
    return {
        "data_source": "stabletoolbench", "prompt": [{"role": "user", "content": initial_observation(system_prompt, query)}],
        "ability": "agent",
        "env_kwargs": {
            "query": query, "system_prompt": system_prompt, "function_map": function_map,
            "record_id": str(record["query_id"]), "benchmark": "stabletoolbench", "split": "test", "eval_group": group,
        },
        "extra_info": {"split": "stabletoolbench_official_subset", "index": idx, "eval_group": group,
                       "query_id": str(record["query_id"]), "missing_local_api_docs": missing_local_docs},
    }


def load_eval_queries(query_dir: Path, download: bool = False) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Load and hash-verify all six official groups, even for a one-group run."""
    groups, files = {}, {}
    for group in EVAL_GROUPS:
        relative = f"test_instruction/{group}.json"
        path = query_dir / relative
        url = f"https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/{OFFICIAL_REVISION}/solvable_queries/{relative}"
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"missing official queries {path}; pass --download-eval-queries to fetch the pinned public files")
            with urllib.request.urlopen(url, timeout=90) as response:
                content = response.read()
            digest = hashlib.sha256(content).hexdigest()
            if digest != OFFICIAL_FILE_SHA256[group]:
                raise ValueError(f"download hash mismatch for {group}: {digest}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != OFFICIAL_FILE_SHA256[group]:
            raise ValueError(f"official query hash mismatch for {path}; refusing to label modified data as StableToolBench")
        groups[group] = json.loads(content)
        files[relative] = {"path": str(path), "url": url, "sha256": digest, "rows": len(groups[group])}
    return groups, {"repository": OFFICIAL_REPOSITORY, "revision": OFFICIAL_REVISION, "files": files}


def select_eval_records(groups: dict[str, list[dict[str, Any]]], selected_groups: list[str], size: int) -> list[tuple[str, dict[str, Any]]]:
    if not selected_groups or len(set(selected_groups)) != len(selected_groups) or set(selected_groups) - groups.keys():
        raise ValueError(f"--eval-groups must contain distinct groups from {','.join(EVAL_GROUPS)}")
    selected, seen = [], set()
    for batch in zip_longest(*(groups[group] for group in selected_groups)):
        for group, record in zip(selected_groups, batch):
            if record is None:
                continue
            key = normalize_query(record["query"])
            if not key or key in seen:
                continue
            seen.add(key)
            selected.append((group, record))
            if len(selected) == size:
                return selected
    raise ValueError(f"requested {size} validation rows, only {len(selected)} distinct official queries available")


def select_train_records(raw_records: list[dict[str, Any]], excluded_queries: set[str], size: int,
                         tool_index: dict[str, dict[str, str]] | None = None,
                         prompt_tokens: Callable[[str, str], int] | None = None,
                         max_prompt_tokens: int = 3584) -> tuple[list[tuple[int, dict[str, Any]]], dict[str, int]]:
    selected, seen = [], set()
    stats = Counter(source_rows=len(raw_records), empty_query_rows=0,
                    rows_overlapping_any_official_eval_query=0, rows_with_missing_tool_docs=0,
                    duplicate_training_query_rows=0, prompt_length_checked_rows=0,
                    rows_excluded_by_prompt_length=0)
    for idx, record in enumerate(raw_records):
        key = normalize_query(record_query(record))
        if not key:
            stats["empty_query_rows"] += 1
            continue
        if key in excluded_queries:
            stats["rows_overlapping_any_official_eval_query"] += 1
            continue
        if tool_index is not None:
            system = next((item.get("value", "") for item in record.get("conversations", []) if item.get("from") == "system"), "")
            names = {name for name in FUNCTION_RE.findall(system) if name.lower() != "finish"}
            if not names or any(name not in tool_index for name in names):
                stats["rows_with_missing_tool_docs"] += 1
                continue
        if key in seen:
            stats["duplicate_training_query_rows"] += 1
            continue
        seen.add(key)
        if len(selected) < size:
            if prompt_tokens is not None:
                system = next((item.get("value", "") for item in record.get("conversations", []) if item.get("from") == "system"), "")
                stats["prompt_length_checked_rows"] += 1
                if prompt_tokens(system, record_query(record)) > max_prompt_tokens:
                    stats["rows_excluded_by_prompt_length"] += 1
                    continue
            selected.append((idx, record))
    if len(selected) < size:
        raise ValueError(f"only {len(selected)} eligible unique training rows, need {size}")
    stats["unique_train_queries_after_query_and_doc_filters"] = len(seen)
    stats["selected_train_rows"] = len(selected)
    stats["train_eval_query_overlap"] = 0
    return selected, dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/ToolBench/toolllama_G123_dfs_train.json")
    parser.add_argument("--tool-root", default="data/StableToolBench/tools")
    parser.add_argument("--output-dir", default="data/toolbench_stable_processed_v2")
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--eval-query-dir", default="data/StableToolBench/solvable_queries")
    parser.add_argument("--eval-groups", default=",".join(EVAL_GROUPS))
    parser.add_argument("--eval-revision", choices=[OFFICIAL_REVISION], default=OFFICIAL_REVISION)
    parser.add_argument("--download-eval-queries", action="store_true", help="download missing official query files at the pinned revision")
    parser.add_argument("--tokenizer", help="local tokenizer path for filtering full initial prompts before split sampling")
    parser.add_argument("--max-initial-prompt-tokens", type=int, default=3584)
    parser.add_argument("--enable-thinking", choices=("true", "false"), default="false")
    args = parser.parse_args()
    if args.train_size < 1 or args.val_size < 1:
        parser.error("train-size and val-size must both be positive")
    groups, provenance = load_eval_queries(Path(args.eval_query_dir).expanduser(), args.download_eval_queries)
    selected_groups = [group.strip() for group in args.eval_groups.split(",")]
    # Validate requested groups before doing tool-index or tokenizer work.
    select_eval_records(groups, selected_groups, args.val_size)
    tool_index = load_tool_index(Path(args.tool_root).expanduser())
    prompt_tokens = None
    if args.tokenizer:
        if args.max_initial_prompt_tokens < 1:
            parser.error("max-initial-prompt-tokens must be positive")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(Path(args.tokenizer).expanduser()), local_files_only=True)

        @lru_cache(maxsize=8192)
        def prompt_tokens(system: str, query: str) -> int:
            messages = [{"role": "user", "content": initial_observation(system, query)}]
            return len(tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=args.enable_thinking == "true"))

    eligible_groups, eval_rows_by_id, eval_length_stats = {}, {}, {}
    for group in selected_groups:
        eligible_groups[group] = []
        stats = Counter(official_candidates=len(groups[group]), excluded_by_prompt_length=0, excluded_invalid_api_docs=0)
        invalid_docs = []
        for idx, record in enumerate(groups[group]):
            try:
                row = convert_eval_record(record, tool_index, group, idx)
            except (KeyError, TypeError, ValueError) as error:
                stats["excluded_invalid_api_docs"] += 1
                invalid_docs.append({"query_id": str(record.get("query_id", idx)), "reason": str(error)})
                continue
            if prompt_tokens is not None:
                count = prompt_tokens(row["env_kwargs"]["system_prompt"], row["env_kwargs"]["query"])
                row["extra_info"]["initial_prompt_tokens"] = count
                if count > args.max_initial_prompt_tokens:
                    stats["excluded_by_prompt_length"] += 1
                    continue
            eligible_groups[group].append(record)
            eval_rows_by_id[(group, str(record["query_id"]))] = row
        stats["eligible_candidates"] = len(eligible_groups[group])
        eval_length_stats[group] = dict(stats, invalid_api_docs=invalid_docs)
    selected_eval = select_eval_records(eligible_groups, selected_groups, args.val_size)
    excluded_queries = {normalize_query(record["query"]) for rows in groups.values() for record in rows}
    source = Path(args.source).expanduser()
    with source.open() as handle:
        raw_records = json.load(handle)
    selected_train, split_stats = select_train_records(raw_records, excluded_queries, args.train_size, tool_index, prompt_tokens, args.max_initial_prompt_tokens)
    train_rows = [convert_record(record, tool_index, idx) for idx, record in selected_train]
    val_rows = [eval_rows_by_id[(group, str(record["query_id"]))] for group, record in selected_eval]
    for idx, row in enumerate(val_rows):
        row["extra_info"]["index"] = idx
    if prompt_tokens is not None:
        for row in train_rows:
            row["extra_info"]["initial_prompt_tokens"] = prompt_tokens(row["env_kwargs"]["system_prompt"], row["env_kwargs"]["query"])
    missing_train = sorted({name for row in train_rows for name in FUNCTION_RE.findall(row["env_kwargs"]["system_prompt"])
                            if name.lower() != "finish" and name not in row["env_kwargs"]["function_map"]})
    if missing_train:
        raise ValueError(f"missing tool documentation for {len(missing_train)} training functions: {missing_train[:10]}")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train_rows).to_parquet(str(output_dir / "train.parquet"))
    Dataset.from_list(val_rows).to_parquet(str(output_dir / "test.parquet"))
    # Stream the large training-file digest instead of making a second copy.
    with source.open("rb") as handle:
        source_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    metadata = {
        "source": str(source), "source_sha256": source_sha256, "train_size": len(train_rows), "val_size": len(val_rows),
        "validation_split": "stabletoolbench_official_subset", "validation_sampling": "deterministic_round_robin_unique_query",
        "validation_source": provenance, "validation_groups": dict(Counter(group for group, _ in selected_eval)),
        "validation_query_ids": [{"group": group, "query_id": str(record["query_id"])} for group, record in selected_eval],
        "validation_has_demonstration_responses": False, "all_official_eval_unique_queries": len(excluded_queries),
        "missing_local_eval_api_docs": sum(row["extra_info"]["missing_local_api_docs"] for row in val_rows),
        "eval_api_doc_source": "complete api_list schemas in the official query files; local tool descriptions supplement them",
        "tool_root": str(Path(args.tool_root).expanduser()), "tool_index_size": len(tool_index), "split_statistics": split_stats,
        "query_normalization": "NFKC, remove trailing ToolLLaMA Begin! line, casefold, collapse whitespace",
        "prompt_includes_tool_definitions": True,
        "initial_prompt_filter": {
            "enabled": bool(args.tokenizer), "tokenizer": args.tokenizer,
            "max_tokens": args.max_initial_prompt_tokens if args.tokenizer else None,
            "chat_template_kwargs": {"add_generation_prompt": True, "enable_thinking": args.enable_thinking == "true"},
            "validation_group_statistics": eval_length_stats,
            "max_selected_train_tokens": max(row["extra_info"]["initial_prompt_tokens"] for row in train_rows) if args.tokenizer else None,
            "max_selected_val_tokens": max(row["extra_info"]["initial_prompt_tokens"] for row in val_rows) if args.tokenizer else None,
        },
        "eval_function_name_policy": "full normalized api_for_tool; text actions do not require lossy 64-character truncation",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "train_size": len(train_rows), "val_size": len(val_rows),
                      "validation_groups": metadata["validation_groups"], "split_statistics": split_stats}, indent=2))


if __name__ == "__main__":
    main()
