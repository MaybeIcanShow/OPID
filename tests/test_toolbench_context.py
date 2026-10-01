"""Conversation retention, prompt budgeting, and honest validation denominators."""
import copy
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from agent_system.environments.env_manager import ToolBenchEnvironmentManager
from agent_system.environments.env_package.toolbench.context import fit_context, format_context
from agent_system.environments.env_package.toolbench.envs import ToolBenchMultiProcessEnv
from agent_system.environments.env_package.toolbench.evaluation import EvaluationResult
from agent_system.environments.env_package.toolbench.metrics import summarize_evaluations
from agent_system.environments.env_package.toolbench.projection import toolbench_projection


class CharacterTokenizer:
    """Predictable token budget while preserving chat-template overhead."""
    def __init__(self):
        self.template_options = []

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kwargs):
        self.template_options.append(kwargs)
        assert add_generation_prompt and not tokenize
        return "<user>" + messages[0]["content"] + "</user><assistant>"

    def encode(self, text, add_special_tokens=False):
        return list(text)

    def size(self, text):
        return len(self.apply_chat_template([{"role": "user", "content": text}], True, False))


class ToolBenchContextTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.initial = "TOOLS: forecast(city)\nQUERY: Beijing weather"

    def test_initial_context_is_unchanged_and_template_overhead_counts(self):
        limit = self.tokenizer.size(self.initial)
        self.assertEqual(fit_context(self.initial, [], self.tokenizer, limit), self.initial)
        with self.assertRaisesRegex(ValueError, "tool definitions exceed"):
            fit_context(self.initial, [], self.tokenizer, limit - 1)

    def test_old_complete_turns_are_dropped_before_latest_is_shortened(self):
        history = [
            {"action": "old_action", "observation": "old_response"},
            {"action": "latest_action", "observation": "latest_response"},
        ]
        original = copy.deepcopy(history)
        latest_context = format_context(self.initial, [history[-1]])
        limit = self.tokenizer.size(latest_context)
        result = fit_context(self.initial, history, self.tokenizer, limit)
        self.assertEqual(result, latest_context)
        self.assertIn(self.initial, result)
        self.assertNotIn("old_action", result)
        self.assertNotIn("truncated", result)
        self.assertEqual(history, original)

    def test_oversized_latest_turn_retains_task_and_marks_truncation(self):
        history = [{"action": "START_A" + "a" * 500 + "END_A", "observation": "START_O" + "b" * 900 + "END_O"}]
        original = copy.deepcopy(history)
        compact = format_context(self.initial, [{"action": "a" * 16, "observation": "b" * 16}])
        limit = self.tokenizer.size(compact) + 120
        result = fit_context(self.initial, history, self.tokenizer, limit)
        self.assertTrue(result.startswith(self.initial))
        self.assertLessEqual(self.tokenizer.size(result), limit)
        for fragment in ("START_A", "END_A", "START_O", "END_O", "[truncated to fit the prompt budget]"):
            self.assertIn(fragment, result)
        self.assertEqual(history, original)

    def test_tool_definitions_are_never_silently_truncated(self):
        initial = "TOOLS: " + "required tool schema " * 100
        with self.assertRaisesRegex(ValueError, "increase the prompt budget"):
            fit_context(initial, [{"action": "x", "observation": "y"}], self.tokenizer, 100)

    def test_no_room_for_latest_observation_raises_instead_of_discarding_it(self):
        with self.assertRaisesRegex(ValueError, "no room"):
            fit_context(self.initial, [{"action": "request", "observation": "response"}], self.tokenizer, self.tokenizer.size(self.initial))

    def test_chat_template_options_are_used_for_budget_measurement(self):
        fit_context(self.initial, [], self.tokenizer, 200, {"enable_thinking": False})
        self.assertTrue(self.tokenizer.template_options)
        self.assertTrue(all(options == {"enable_thinking": False} for options in self.tokenizer.template_options))


class ToolBenchManagerContextTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        config = OmegaConf.create({
            "max_steps": 12, "toolbench": {"cache_root": directory.name, "tools_root": directory.name,
                                              "evaluator": {"enabled": False}},
        })
        with patch("agent_system.environments.env_package.toolbench.envs.logger.warning"):
            self.env = ToolBenchMultiProcessEnv(0, 1, 1, False, config)
        self.addCleanup(self.env.close)
        self.env.client.execute = MagicMock(side_effect=[
            ({"error": "", "response": "FIRST_OBSERVATION"}, "mirrorapi_cache"),
            ({"error": "", "response": "SECOND_OBSERVATION"}, "disk_cache"),
        ])
        self.record = {
            "query": "Find the forecast for Beijing.", "system_prompt": "forecast(city) provides weather; keep the city argument.",
            "function_map": {"forecast": {"category": "Weather", "tool_name": "weather", "api_name": "forecast"}},
        }
        self.manager = ToolBenchEnvironmentManager(self.env, toolbench_projection, OmegaConf.create({"env": {"history_length": 1}}))

    def test_second_and_later_turns_keep_task_tools_actions_and_observations(self):
        initial, _ = self.manager.reset([self.record])
        first_action = 'Action: forecast\nAction Input: {"city": "Beijing"}'
        observations, _, _, _ = self.manager.step([first_action])
        text = observations["text"][0]
        for fragment in (self.record["query"], self.record["system_prompt"], first_action, "FIRST_OBSERVATION"):
            self.assertIn(fragment, text)
        self.assertEqual(observations["text_base"], initial["text"])
        first_snapshot = observations["toolbench_context"][0]
        next_action = 'Action: forecast\nAction Input: {"city": "Beijing", "days": 2}'
        later, _, _, _ = self.manager.step([next_action])
        for fragment in (first_action, "FIRST_OBSERVATION", next_action, "SECOND_OBSERVATION", self.record["query"]):
            self.assertIn(fragment, later["text"][0])
        self.assertEqual(len(later["toolbench_context"][0]["history"]), 2)
        self.assertEqual(len(first_snapshot["history"]), 1)

    def test_terminal_batch_steps_do_not_add_fake_history(self):
        self.manager.reset([self.record])
        self.env.evaluator = SimpleNamespace(configured=True, model="MirrorAPI", evaluate=MagicMock(return_value=EvaluationResult("success", 1.0, "done")))
        finish = 'Action: Finish\nAction Input: {"return_type": "give_answer", "final_answer": "The forecast is sunny."}'
        observations, _, _, _ = self.manager.step([finish])
        self.assertEqual(len(observations["toolbench_context"][0]["history"]), 1)
        after, rewards, dones, _ = self.manager.step(['Action: forecast\nAction Input: {}'])
        self.assertEqual(len(after["toolbench_context"][0]["history"]), 1)
        self.assertEqual(rewards.tolist(), [0.0])
        self.assertEqual(dones.tolist(), [True])
        self.env.client.execute.assert_not_called()
        self.env.evaluator.evaluate.assert_called_once()

    def test_reset_clears_previous_episode_context(self):
        self.manager.reset([self.record])
        self.manager.step(['Action: forecast\nAction Input: {}'])
        new_record = {**self.record, "query": "Find weather for Chengdu."}
        observations, _ = self.manager.reset([new_record])
        self.assertIn(new_record["query"], observations["text"][0])
        self.assertNotIn("FIRST_OBSERVATION", observations["text"][0])
        self.assertNotIn(self.record["query"], observations["text"][0])
        self.assertEqual(observations["toolbench_context"][0]["history"], [])

    def test_environment_rejection_overrides_syntactically_valid_projection(self):
        self.manager.reset([self.record])
        _, _, _, infos = self.manager.step(['Action: unknown\nAction Input: {}'])
        self.assertEqual(int(infos[0]["is_action_valid"]), 0)
        self.env.client.execute.assert_not_called()


class ToolBenchEvaluationMetricsTest(unittest.TestCase):
    @staticmethod
    def record(status, group="G1_instruction", submitted=True, sources=None):
        return {
            "evaluation": {"status": status, "score": 1.0 if status == "success" else 0.0 if status == "failure" else None},
            "eval_group": group, "answer_submitted": submitted, "response_sources": sources or {},
        }

    def test_full_coverage_has_trajectory_success_rate_and_groups(self):
        records = [self.record("success", sources={"disk_cache": 2}), self.record("failure", submitted=False, sources={"mirrorapi_cache": 1}), self.record("success", "G2_category")]
        metrics = summarize_evaluations(records)
        self.assertEqual(metrics["tasks"], 3)
        self.assertEqual(metrics["judge_coverage"], 1)
        self.assertAlmostEqual(metrics["judge_success_rate"], 2 / 3)
        self.assertAlmostEqual(metrics["answer_submission_rate"], 2 / 3)
        self.assertEqual(metrics["groups/G1_instruction/judge_success_rate"], 0.5)
        self.assertEqual(metrics["groups/G2_category/judge_success_rate"], 1.0)
        self.assertEqual(metrics["tool_calls"], 3)
        self.assertAlmostEqual(metrics["tool_responses/disk_cache_fraction"], 2 / 3)

    def test_judge_errors_or_unscored_tasks_do_not_become_full_success_rate(self):
        records = [self.record("success"), self.record("error"), self.record("unscored", "G2_category"), self.record("failure", "G2_category")]
        metrics = summarize_evaluations(records)
        self.assertEqual(metrics["judge_coverage"], 0.5)
        self.assertEqual(metrics["judge_error_tasks"], 1)
        self.assertEqual(metrics["unscored_tasks"], 1)
        self.assertNotIn("judge_success_rate", metrics)
        self.assertEqual(metrics["judge_success_rate_scored_subset"], 0.5)
        self.assertNotIn("groups/G1_instruction/judge_success_rate", metrics)
        self.assertNotIn("groups/G2_category/judge_success_rate", metrics)
        self.assertEqual(metrics["groups/G1_instruction/judge_coverage"], 0.5)

    def test_no_scored_tasks_has_no_success_rate(self):
        metrics = summarize_evaluations([self.record("unscored"), self.record("error")])
        self.assertEqual(metrics["judge_scored_tasks"], 0)
        self.assertEqual(metrics["judge_coverage"], 0)
        self.assertFalse(any("success_rate" in key for key in metrics))
        self.assertEqual(summarize_evaluations([]), {})


if __name__ == "__main__":
    unittest.main()
