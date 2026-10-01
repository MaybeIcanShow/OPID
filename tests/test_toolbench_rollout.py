"""CPU integration coverage for multi-turn ToolBench validation collection.

Uses the production collector, DataProto, context budgeting and manager, with a
character tokenizer and a scripted three-worker actor. No weights, GPU or remote
endpoint are loaded. Two real tasks exercise worker padding and different episode
lengths, including a failed tool request and a terminal judge error.
"""
import json
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from omegaconf import OmegaConf

from agent_system.environments.env_manager import ToolBenchEnvironmentManager
from agent_system.environments.env_package.toolbench.envs import ToolBenchMultiProcessEnv
from agent_system.environments.env_package.toolbench.evaluation import EvaluationResult
from agent_system.environments.env_package.toolbench.metrics import summarize_evaluations
from agent_system.environments.env_package.toolbench.projection import toolbench_projection
from agent_system.environments.env_package.toolbench.stabletoolbench import StableToolBenchError
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


class CpuCharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(character) + 2 for character in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(int(token) - 2) for token in token_ids if int(token) > 1)

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return [self.decode(row, skip_special_tokens) for row in token_ids]

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        assert return_tensors == "pt"
        tokens = torch.tensor([self.encode(text)], dtype=torch.long)
        return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kwargs):
        assert add_generation_prompt and not tokenize
        return "<user>\n" + messages[0]["content"] + "\n</user>\n<assistant>\n"


def action(name, arguments):
    return "Action: " + name + "\nAction Input: " + json.dumps(arguments)


class ScriptedCpuActor:
    world_size = 3

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.observed_prompts = []
        self.batch_sizes = []
        self.started = 0
        self.stopped = 0

    def start_rollout_generation_session(self):
        self.started += 1

    def stop_rollout_generation_session(self):
        self.stopped += 1

    def generate_sequences(self, batch):
        assert len(batch) == self.world_size, "Two validation tasks must be padded to three workers"
        assert batch.batch["input_ids"].device.type == "cpu"
        step = len(self.observed_prompts)
        prompts = self.tokenizer.batch_decode(batch.batch["input_ids"])
        self.observed_prompts.append(prompts)
        self.batch_sizes.append(len(batch))
        texts = []
        for prompt in prompts:
            case = "alpha" if "TASK_ALPHA" in prompt else "beta"
            if step == 0 or (step == 1 and case == "beta"):
                texts.append(action("lookup", {"case": case}))
            else:
                texts.append(action("Finish", {"return_type": "give_answer", "final_answer": f"{case.upper()}_FINAL_ANSWER"}))
        responses = torch.zeros((len(batch), 160), dtype=torch.long)
        for index, text in enumerate(texts):
            tokens = self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
            assert len(tokens) <= responses.shape[1]
            responses[index, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        input_ids = torch.cat([batch.batch["input_ids"], responses], dim=-1)
        attention_mask = torch.cat([batch.batch["attention_mask"], responses.ne(0).long()], dim=-1)
        return DataProto.from_dict(tensors={
            "prompts": batch.batch["input_ids"].clone(),
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask),
        })


class RecordingToolBenchManager(ToolBenchEnvironmentManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_history = []

    def step(self, actions):
        result = super().step(actions)
        self.metric_history.append(summarize_evaluations(self.get_evaluation_records()))
        return result


class ToolBenchRolloutIntegrationTest(unittest.TestCase):
    def run_rollout(self, beta_status):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        config = OmegaConf.create({
            "env": {
                "max_steps": 3, "rollout": {"n": 1},
                "toolbench": {"cache_root": directory.name, "tools_root": directory.name,
                              "request_timeout": 1, "evaluator": {"enabled": False}},
            },
            "data": {"max_prompt_length": 700, "truncation": "left", "return_raw_chat": True,
                     "apply_chat_template_kwargs": {"enable_thinking": False}},
            "algorithm": {"filter_groups": {"enable": False}},
        })
        with patch("agent_system.environments.env_package.toolbench.envs.logger.warning"):
            # Capacity is three, but the last validation batch contains two tasks.
            env = ToolBenchMultiProcessEnv(0, 3, 1, False, config.env)
        self.addCleanup(env.close)
        backend_counts = {"alpha": 0, "beta": 0}
        lock = threading.Lock()

        def backend(mapping, arguments):
            case = arguments["case"]
            with lock:
                backend_counts[case] += 1
                count = backend_counts[case]
            if case == "beta" and count == 1:
                raise StableToolBenchError("request_timeout", "BETA_REQUEST_TIMEOUT")
            # Long responses force production fit_context to trim observations,
            # while retaining the initial user task and complete tool definition.
            response = f"{case.upper()}_TOOL_RESULT_HEAD " + "details " * 500 + f" {case.upper()}_TOOL_RESULT_TAIL"
            return {"error": "", "response": response}, "disk_cache" if case == "alpha" else "mirrorapi_cache"

        def judge(query, answer):
            if "TASK_ALPHA" in query:
                return EvaluationResult("success", 1.0, "Complete answer")
            return EvaluationResult(beta_status, 0.0 if beta_status == "failure" else None, "Beta verdict")

        env.client.execute = MagicMock(side_effect=backend)
        env.evaluator = SimpleNamespace(configured=True, model="MirrorAPI", evaluate=MagicMock(side_effect=judge))
        records = [
            {"record_id": case, "query": f"TASK_{case.upper()}: Report the answer from lookup.",
             "system_prompt": "TOOL_SCHEMA_PROTECTED: lookup(case) requires a string case and returns a report.",
             "benchmark": "stabletoolbench", "eval_group": group,
             "function_map": {"lookup": {"category": "Reports", "tool_name": "reports", "api_name": "lookup"}}}
            for case, group in (("alpha", "G1_instruction"), ("beta", "G2_category"))
        ]
        gen_batch = DataProto.from_dict(
            tensors={"input_ids": torch.zeros((2, 1), dtype=torch.long)},
            non_tensors={
                "raw_prompt": np.array([[{"role": "user", "content": item["query"]}] for item in records], dtype=object),
                "data_source": np.array(["stabletoolbench"] * 2, dtype=object),
                "env_kwargs": np.array(records, dtype=object),
            },
            meta_info={"eos_token_id": 1, "pad_token_id": 0, "validate": True},
        )
        tokenizer = CpuCharacterTokenizer()
        actor = ScriptedCpuActor(tokenizer)
        manager = RecordingToolBenchManager(env, toolbench_projection, config)
        collector = TrajectoryCollector(config, tokenizer)
        with patch("torch.cuda.init", side_effect=AssertionError("This integration test must remain CPU only")):
            output = collector.multi_turn_loop(gen_batch, actor, manager, is_train=False)
        return output, actor, manager, env, tokenizer, records, backend_counts

    def assert_collection_integrity(self, output, actor, manager, env, tokenizer, records, backend_counts, beta_status):
        self.assertIsInstance(output, DataProto)
        # Alpha ends after two turns, beta after three: padded actor rows and
        # alpha's inactive third turn must not become training/validation rows.
        self.assertEqual(len(output), 5)
        self.assertEqual(actor.batch_sizes, [3, 3, 3])
        self.assertEqual((actor.started, actor.stopped), (1, 1))
        for key, values in output.non_tensor_batch.items():
            self.assertEqual(len(values), 5, f"Non-tensor field {key} lost rows")
        for key, values in output.batch.items():
            self.assertEqual(values.shape[0], 5, f"Tensor field {key} lost rows")
            self.assertEqual(values.device.type, "cpu")
        np.testing.assert_array_equal(output.non_tensor_batch["index"], [0, 0, 1, 1, 1])
        np.testing.assert_array_equal(output.non_tensor_batch["step_num"], [0, 1, 0, 1, 2])
        np.testing.assert_array_equal(output.non_tensor_batch["episode_lengths"], [2, 2, 3, 3, 3])
        self.assertEqual(len(set(output.non_tensor_batch["traj_uid"])), 2)
        self.assertEqual(len(manager.get_evaluation_records()), 2)
        self.assertEqual(backend_counts, {"alpha": 1, "beta": 2})
        self.assertEqual(env.evaluator.evaluate.call_count, 2)
        expected_metadata = {
            "response_source": ["disk_cache", "", "unavailable", "mirrorapi_cache", ""],
            "response_error_code": ["", "", "request_timeout", "", ""],
            "evaluation_status": ["", "success", "", "", beta_status],
            "termination_reason": ["", "give_answer", "", "", "give_answer"],
        }
        for key, expected in expected_metadata.items():
            self.assertEqual(output.non_tensor_batch[key].tolist(), expected)
        for prompts in actor.observed_prompts:
            self.assertEqual(prompts[0], prompts[2], "Padding must duplicate only at the actor boundary")
            for index, prompt in enumerate(prompts):
                record = records[0 if index == 2 else index]
                self.assertIn(record["query"], prompt)
                self.assertIn(record["system_prompt"], prompt)
                self.assertLessEqual(len(tokenizer.encode(prompt)), 700)
        alpha_second = actor.observed_prompts[1][0]
        self.assertIn(action("lookup", {"case": "alpha"}), alpha_second)
        self.assertIn("ALPHA_TOOL_RESULT_HEAD", alpha_second)
        self.assertIn("ALPHA_TOOL_RESULT_TAIL", alpha_second)
        self.assertIn("[truncated to fit the prompt budget]", alpha_second)
        self.assertIn("BETA_REQUEST_TIMEOUT", actor.observed_prompts[1][1])
        beta_third = actor.observed_prompts[2][1]
        self.assertIn("BETA_TOOL_RESULT_HEAD", beta_third)
        self.assertIn("BETA_TOOL_RESULT_TAIL", beta_third)
        self.assertNotIn("BETA_REQUEST_TIMEOUT", beta_third, "Old complete turn should be trimmed first")
        self.assertEqual([m["judge_coverage"] for m in manager.metric_history[:2]], [0.0, 0.5])
        self.assertNotIn("judge_success_rate", manager.metric_history[1])
        self.assertEqual(manager.metric_history[1]["judge_success_rate_scored_subset"], 1.0)

    def test_validation_multiturn_padding_metadata_and_judge_error(self):
        objects = self.run_rollout("error")
        self.assert_collection_integrity(*objects, beta_status="error")
        output, _, manager, _, _, _, _ = objects
        metrics = manager.metric_history[-1]
        self.assertEqual(metrics["judge_coverage"], 0.5)
        self.assertEqual(metrics["judge_error_tasks"], 1)
        self.assertNotIn("judge_success_rate", metrics)
        self.assertEqual(metrics["judge_success_rate_scored_subset"], 1.0)
        np.testing.assert_array_equal(output.non_tensor_batch["toolbench_judge_scored_fraction"], [0.5] * 5)
        np.testing.assert_array_equal(output.non_tensor_batch["toolbench_judge_error_fraction"], [0.5] * 5)
        self.assertAlmostEqual(float(output.non_tensor_batch["episode_rewards"][-1]), -0.05, places=6)

    def test_validation_full_judge_coverage_reports_both_task_outcomes(self):
        objects = self.run_rollout("failure")
        self.assert_collection_integrity(*objects, beta_status="failure")
        output, _, manager, _, _, _, _ = objects
        metrics = manager.metric_history[-1]
        self.assertEqual(metrics["judge_coverage"], 1.0)
        self.assertEqual(metrics["judge_success_rate"], 0.5)
        self.assertEqual(metrics["groups/G1_instruction/judge_success_rate"], 1.0)
        self.assertEqual(metrics["groups/G2_category/judge_success_rate"], 0.0)
        np.testing.assert_array_equal(output.non_tensor_batch["toolbench_judge_scored_fraction"], [1.0] * 5)
        np.testing.assert_array_equal(output.non_tensor_batch["toolbench_judge_error_fraction"], [0.0] * 5)
        np.testing.assert_array_equal(output.non_tensor_batch["toolbench_judge_success_fraction"], [0.5] * 5)


if __name__ == "__main__":
    unittest.main()
