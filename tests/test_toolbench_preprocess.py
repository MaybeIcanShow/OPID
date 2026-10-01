"""Regression coverage for the official validation split and leakage prevention."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq
from datasets import Dataset

from examples.data_preprocess import preprocess_toolbench as preprocess


def training_record(query, record_id="different id"):
    return {
        "id": record_id,
        "conversations": [
            {"from": "system", "value": 'Tools: [{"name": "search_for_demo"}]'},
            {"from": "user", "value": query},
            {"from": "assistant", "value": 'Action: search_for_demo\nAction Input: {"q":"a"}'},
            {"from": "function", "value": "training demonstration only"},
        ],
    }


def official_record(query="find a place", query_id=1, tool="Demo"):
    return {
        "query_id": query_id, "query": query,
        "api_list": [{
            "category_name": "Search", "tool_name": tool, "api_name": "Search",
            "api_description": "Search for a place", "method": "GET",
            "required_parameters": [{"name": "q", "type": "STRING", "description": "Place name", "default": "Paris"}],
            "optional_parameters": [{"name": "limit", "type": "NUMBER", "description": "Result count", "default": 3}],
        }],
        # A stray demonstration must never enter validation inputs.
        "conversations": [{"from": "function", "value": "leaked answer"}],
    }


class ToolBenchPreprocessTests(unittest.TestCase):
    def test_query_deduplication_ignores_ids_and_training_wrapper(self):
        rows = [
            training_record("  Official\tQUESTION\nBegin!\n", "not_the_eval_id"),
            training_record("Ａnother   Query\nBegin!", "a"),
            training_record("another query", "b"),
            training_record("new question", "a"),
        ]
        selected, stats = preprocess.select_train_records(rows, {preprocess.normalize_query("official question")}, 2)
        self.assertEqual([idx for idx, _ in selected], [1, 3])
        self.assertEqual(stats["rows_overlapping_any_official_eval_query"], 1)
        self.assertEqual(stats["duplicate_training_query_rows"], 1)
        self.assertEqual(stats["train_eval_query_overlap"], 0)

    def test_round_robin_validation_groups_are_balanced_and_unique(self):
        groups = {group: [official_record(f"{group} question {i}", i) for i in range(30)] for group in preprocess.EVAL_GROUPS}
        selected = preprocess.select_eval_records(groups, list(preprocess.EVAL_GROUPS), 128)
        counts = {group: sum(selected_group == group for selected_group, _ in selected) for group in groups}
        self.assertEqual(list(counts.values()), [22, 22, 21, 21, 21, 21])
        groups[preprocess.EVAL_GROUPS[1]][0]["query"] = groups[preprocess.EVAL_GROUPS[0]][0]["query"]
        selected = preprocess.select_eval_records(groups, list(preprocess.EVAL_GROUPS), 128)
        self.assertEqual(len({preprocess.normalize_query(row["query"]) for _, row in selected}), 128)

    def test_complete_validation_schema_without_training_responses(self):
        row = preprocess.convert_eval_record(official_record(), {}, "G1_instruction", 0)
        env = row["env_kwargs"]
        self.assertEqual(row["data_source"], "stabletoolbench")
        self.assertEqual(row["extra_info"]["split"], "stabletoolbench_official_subset")
        self.assertNotIn("offline_tool_responses", env)
        self.assertNotIn("leaked answer", json.dumps(row))
        self.assertIn('"required": ["q"]', env["system_prompt"])
        self.assertIn('"optional": ["limit"]', env["system_prompt"])
        self.assertIn("Place name", row["prompt"][0]["content"])
        self.assertEqual(row["prompt"][0]["content"], preprocess.initial_observation(env["system_prompt"], env["query"]))
        self.assertEqual(json.loads(env["function_map"]["search_for_demo"]["api_schema_json"])["required_parameters"][0]["name"], "q")
        self.assertEqual(row["extra_info"]["missing_local_api_docs"], 1)

    def test_incomplete_official_documentation_fails(self):
        record = official_record()
        del record["api_list"][0]["required_parameters"]
        with self.assertRaisesRegex(ValueError, "incomplete official API documentation"):
            preprocess.convert_eval_record(record, {}, "G1_instruction", 0)

    def test_parquet_validation_never_creates_offline_response_nulls(self):
        rows = [preprocess.convert_eval_record(official_record(tool=tool, query_id=idx), {}, "G1_instruction", idx)
                for idx, tool in enumerate(["First", "Second"])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.parquet"
            Dataset.from_list(rows).to_parquet(str(path))
            restored = pq.read_table(path).to_pylist()
        for row in restored:
            self.assertNotIn("offline_tool_responses", row["env_kwargs"])
            mappings = [mapping for mapping in row["env_kwargs"]["function_map"].values() if mapping is not None]
            self.assertEqual(len(mappings), 1)
            self.assertTrue(all(isinstance(mapping["api_schema_json"], str) for mapping in mappings))

    def test_official_source_metadata_has_revision_hash_and_refuses_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_instruction").mkdir()
            hashes = {}
            for group in preprocess.EVAL_GROUPS:
                data = json.dumps([official_record(query=group)]).encode()
                (root / "test_instruction" / f"{group}.json").write_bytes(data)
                hashes[group] = hashlib.sha256(data).hexdigest()
            with patch.object(preprocess, "OFFICIAL_FILE_SHA256", hashes):
                groups, metadata = preprocess.load_eval_queries(root)
                self.assertEqual(metadata["revision"], preprocess.OFFICIAL_REVISION)
                self.assertEqual(metadata["repository"], preprocess.OFFICIAL_REPOSITORY)
                self.assertEqual(set(groups), set(preprocess.EVAL_GROUPS))
                for group in preprocess.EVAL_GROUPS:
                    self.assertEqual(metadata["files"][f"test_instruction/{group}.json"]["sha256"], hashes[group])
                (root / "test_instruction" / "G1_instruction.json").write_text("[]")
                with self.assertRaisesRegex(ValueError, "refusing to label modified data"):
                    preprocess.load_eval_queries(root)

    def test_prompt_filtering_happens_before_training_size_is_filled(self):
        rows = [training_record("too long"), training_record("fits")]
        selected, stats = preprocess.select_train_records(
            rows, set(), 1, {"search_for_demo": {}},
            prompt_tokens=lambda system, query: 100 if query == "too long" else 10,
            max_prompt_tokens=20,
        )
        self.assertEqual([idx for idx, _ in selected], [1])
        self.assertEqual(stats["rows_excluded_by_prompt_length"], 1)
        self.assertEqual(stats["prompt_length_checked_rows"], 2)

    def test_repeated_official_parameter_preserves_both_examples(self):
        record = official_record()
        record["api_list"][0]["optional_parameters"].append(
            {"name": "limit", "type": "NUMBER", "description": "Result count", "default": 5})
        row = preprocess.convert_eval_record(record, {}, "G1_instruction", 0)
        self.assertIn('"examples": [3, 5]', row["env_kwargs"]["system_prompt"])
        self.assertIn('"optional": ["limit"]', row["env_kwargs"]["system_prompt"])

    def test_optional_parameters_remain_when_no_required_parameters(self):
        record = official_record()
        record["api_list"][0]["required_parameters"] = []
        row = preprocess.convert_eval_record(record, {}, "G1_instruction", 0)
        self.assertIn('"optional": ["limit"]', row["env_kwargs"]["system_prompt"])

    def test_training_skips_missing_tool_documentation_explicitly(self):
        bad = training_record("missing tool")
        bad["conversations"][0]["value"] = 'Tools: [{"name": "unavailable_function"}]'
        good = training_record("supported query")
        selected, stats = preprocess.select_train_records([bad, good], set(), 1, {"search_for_demo": {}})
        self.assertEqual([idx for idx, _ in selected], [1])
        self.assertEqual(stats["rows_with_missing_tool_docs"], 1)

    def test_long_tool_names_keep_distinct_apis(self):
        record = official_record(tool="a_very_long_tool_name_" * 4)
        second = dict(record["api_list"][0], api_name="Another Search")
        record["api_list"].append(second)
        row = preprocess.convert_eval_record(record, {}, "G1_instruction", 0)
        self.assertEqual(len(row["env_kwargs"]["function_map"]), 2)

    def test_legacy_training_conversion_keeps_api_and_full_prompt(self):
        mapping = {"category": "Search", "tool_name": "Demo", "api_name": "Search"}
        row = preprocess.convert_record(training_record("question\nBegin!"), {"search_for_demo": mapping}, 7)
        self.assertEqual(row["env_kwargs"]["query"], "question")
        self.assertIn("Tools:", row["prompt"][0]["content"])
        self.assertEqual(row["env_kwargs"]["offline_tool_responses"]["search_for_demo"], ["training demonstration only"])
        self.assertEqual(row["extra_info"], {"split": "train", "index": 7})


if __name__ == "__main__":
    unittest.main()
