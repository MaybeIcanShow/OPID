"""Environment regressions: no trajectory replay and no Finish-as-success rule."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from agent_system.environments.env_package.toolbench.envs import ToolBenchMultiProcessEnv
from agent_system.environments.env_package.toolbench.evaluation import EvaluationResult
from agent_system.environments.env_package.toolbench.projection import parse_tool_action, toolbench_projection
from agent_system.environments.env_package.toolbench.stabletoolbench import StableToolBenchError


MAPPING = {"category": "Weather", "tool_name": "weather", "api_name": "forecast"}
TOOL_ACTION = 'Action: forecast\nAction Input: {"city": "Beijing"}'


def finish_action(answer):
    return "Action: Finish\nAction Input: " + json.dumps({"return_type": "give_answer", "final_answer": answer})


class ToolBenchEnvTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def make_env(self, env_num=1, max_steps=12):
        config = OmegaConf.create({
            "max_steps": max_steps,
            "toolbench": {
                "cache_root": str(self.directory / "cache"), "tools_root": str(self.directory / "tools"),
                "backend": "mirrorapi_cache", "request_timeout": 1, "evaluator": {"enabled": False},
            },
        })
        with patch("agent_system.environments.env_package.toolbench.envs.logger.warning"):
            env = ToolBenchMultiProcessEnv(0, env_num, 1, True, config)
        self.addCleanup(env.close)
        return env

    def record(self, **changes):
        return {"query": "What is Beijing's weather?", "system_prompt": "Available tools: forecast(city).",
                "function_map": {"forecast": dict(MAPPING)}, "record_id": "test-1", **changes}

    def set_judge(self, env, result):
        env.evaluator = SimpleNamespace(configured=True, model="MirrorAPI", evaluate=MagicMock(return_value=result))
        return env.evaluator.evaluate

    def test_parquet_null_tool_mapping_does_not_break_other_trajectories(self):
        records = [
            {"query": "a", "system_prompt": "", "offline_tool_responses": {"first": ["first old response"]},
             "function_map": {"first": {**MAPPING, "api_name": "first"}}},
            {"query": "b", "system_prompt": "", "offline_tool_responses": {"second": ["second old response"]},
             "function_map": {"second": {**MAPPING, "api_name": "second"}}},
        ]
        path = self.directory / "records.parquet"
        pq.write_table(pa.Table.from_pylist(records), path)
        restored = pq.read_table(path).to_pylist()
        self.assertIsNone(restored[0]["function_map"]["second"])
        self.assertIsNone(restored[0]["offline_tool_responses"]["second"])
        env = self.make_env(env_num=2)
        env.client.execute = MagicMock(return_value=({"error": "", "response": "fresh result"}, "mirrorapi_cache"))
        env.reset(restored)
        observations, rewards, dones, infos = env.step(['Action: second\nAction Input: {}'] * 2)
        self.assertEqual(infos[0]["response_source"], "invalid_action")
        self.assertEqual(rewards[0], -0.05)
        self.assertFalse(dones[0])
        self.assertEqual(infos[1]["response_source"], "mirrorapi_cache")
        self.assertIn("fresh result", observations[1])
        self.assertNotIn("old response", observations[1])
        self.assertEqual(rewards[1], 0)
        env.client.execute.assert_called_once()

    def test_offline_trajectory_responses_are_ignored(self):
        env = self.make_env()
        env.client.execute = MagicMock(return_value=({"error": "", "response": "current request response"}, "mirrorapi_cache"))
        env.reset([self.record(offline_tool_responses={"forecast": ["WRONG ORIGINAL ARGUMENTS"]})])
        for _ in range(2):
            observations, rewards, dones, infos = env.step([TOOL_ACTION])
            self.assertIn("current request response", observations[0])
            self.assertNotIn("WRONG ORIGINAL ARGUMENTS", observations[0])
            self.assertEqual(rewards[0], 0)
            self.assertFalse(dones[0])
            self.assertEqual(infos[0]["response_source"], "mirrorapi_cache")
        self.assertEqual(env.client.execute.call_count, 2)
        env.client.execute.assert_called_with(MAPPING, {"city": "Beijing"})

    def test_real_disk_cache_hit_works_through_environment_without_remote_request(self):
        env = self.make_env()
        path = self.directory / "cache" / "Weather" / "weather_for_Weather" / "forecast.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"{'city': 'Beijing'}": {"error": "", "response": {"temperature": 22}}}))
        env.reset([self.record(offline_tool_responses={"forecast": ["WRONG OFFLINE RESPONSE"]})])
        with patch.object(env.client, "_session", side_effect=AssertionError("cache hit must not use network")):
            observations, rewards, _, infos = env.step([TOOL_ACTION])
        self.assertEqual(infos[0]["response_source"], "disk_cache")
        self.assertIn('"temperature": 22', observations[0])
        self.assertEqual(rewards, [0.0])
        self.assertEqual(env.get_evaluation_records()[0]["response_sources"], {"disk_cache": 1})

    def test_backend_error_is_exposed_without_fake_tool_success(self):
        env = self.make_env()
        env.client.execute = MagicMock(side_effect=StableToolBenchError("request_timeout", "MirrorAPI request timed out."))
        env.reset([self.record()])
        observations, rewards, dones, infos = env.step([TOOL_ACTION])
        self.assertIn("timed out", observations[0])
        self.assertEqual(rewards, [-0.05])
        self.assertEqual(dones, [False])
        self.assertFalse(infos[0]["won"])
        self.assertEqual(infos[0]["response_source"], "unavailable")
        self.assertEqual(infos[0]["response_error_code"], "request_timeout")

    def test_finish_without_judge_has_no_success_bonus(self):
        env = self.make_env()
        env.reset([self.record()])
        _, rewards, dones, infos = env.step([finish_action("Please provide more information.")])
        self.assertEqual(rewards, [0.0])
        self.assertEqual(dones, [True])
        self.assertFalse(infos[0]["won"])
        self.assertEqual(infos[0]["evaluation_status"], "unscored")
        record = env.get_evaluation_records()[0]
        self.assertTrue(record["answer_submitted"])
        self.assertIsNone(record["evaluation"]["score"])

    def test_judge_verdict_controls_final_reward_and_success(self):
        for status, score, answer in (
            ("failure", 0.0, "Please provide more information."),
            ("success", 1.0, "Beijing is 22 degrees Celsius."),
            ("error", None, "The answer could not be evaluated."),
        ):
            with self.subTest(status=status):
                env = self.make_env()
                judge = self.set_judge(env, EvaluationResult(status, score, "judge reason"))
                env.reset([self.record()])
                _, rewards, dones, infos = env.step([finish_action(answer)])
                self.assertEqual(rewards, [score if score is not None else 0.0])
                self.assertEqual(dones, [True])
                self.assertEqual(infos[0]["won"], status == "success")
                self.assertEqual(infos[0]["evaluation_status"], status)
                judge.assert_called_once_with("What is Beijing's weather?", answer)

    def test_completed_trajectory_is_not_executed_or_scored_again(self):
        env = self.make_env()
        judge = self.set_judge(env, EvaluationResult("success", 1.0, "complete"))
        env.client.execute = MagicMock()
        env.reset([self.record()])
        env.step([finish_action("22 degrees Celsius.")])
        _, rewards, dones, infos = env.step([TOOL_ACTION])
        self.assertEqual(rewards, [0.0])
        self.assertEqual(dones, [True])
        self.assertTrue(infos[0]["won"])
        self.assertEqual(env.envs[0]["_state"]["step"], 1)
        env.client.execute.assert_not_called()
        judge.assert_called_once()
        env.step([finish_action("different answer")])
        judge.assert_called_once()

    def test_unknown_tools_and_invalid_json_objects_never_reach_backend(self):
        actions = (
            'Action: unknown\nAction Input: {}', 'Action: Finish->give_answer\nAction Input: {}',
            'Action: forecast\nAction Input: []', 'Action: forecast\nAction Input: null',
            'Action: forecast\nAction Input: {broken}', 'Action: forecast\nAction Input: {"city": NaN}',
            'Action: forecast\nAction Input: {"city": "Beijing", "city": "Shanghai"}',
        )
        env = self.make_env()
        env.client.execute = MagicMock(return_value=({"error": "", "response": "unexpected"}, "mirrorapi_cache"))
        for action in actions:
            with self.subTest(action=action):
                env.reset([self.record()])
                _, rewards, dones, infos = env.step([action])
                self.assertEqual(rewards, [-0.05])
                self.assertEqual(dones, [False])
                self.assertEqual(infos[0]["is_action_valid"], 0)
        env.client.execute.assert_not_called()

    def test_malformed_finish_does_not_earn_or_request_a_verdict(self):
        actions = (
            'Action: Finish\nAction Input: {"return_type": "give_answer"}',
            'Action: Finish\nAction Input: {"return_type": "give_answer", "final_answer": " "}',
            'Action: Finish\nAction Input: {"return_type": "give_answer", "final_answer": 123}',
            'Action: Finish\nAction Input: {"return_type": "something_else", "final_answer": "done"}',
        )
        env = self.make_env()
        judge = self.set_judge(env, EvaluationResult("success", 1.0, "must not be used"))
        for action in actions:
            with self.subTest(action=action):
                env.reset([self.record()])
                _, rewards, dones, infos = env.step([action])
                self.assertEqual(rewards, [-0.05])
                self.assertEqual(dones, [False])
                self.assertEqual(infos[0]["is_action_valid"], 0)
        judge.assert_not_called()

    def test_step_budget_terminates_once_and_keeps_no_judge_unscored(self):
        env = self.make_env(max_steps=1)
        env.client.execute = MagicMock(return_value=({"error": "", "response": "22 degrees"}, "mirrorapi_cache"))
        env.reset([self.record()])
        _, rewards, dones, infos = env.step([TOOL_ACTION])
        self.assertEqual(rewards, [0.0])
        self.assertEqual(dones, [True])
        self.assertEqual(infos[0]["termination_reason"], "max_steps")
        self.assertEqual(infos[0]["evaluation_status"], "unscored")
        env.step([TOOL_ACTION])
        env.client.execute.assert_called_once()
        self.assertEqual(env.get_evaluation_records()[0]["tool_calls"], 1)

    def test_short_validation_batch_has_no_padded_fake_tasks(self):
        env = self.make_env(env_num=4)
        observations, _ = env.reset([self.record()])
        self.assertEqual(len(observations), 1)
        _, rewards, _, _ = env.step([finish_action("answer")])
        self.assertEqual(len(rewards), 1)
        self.assertEqual(len(env.get_evaluation_records()), 1)
        with self.assertRaises(ValueError):
            env.step([TOOL_ACTION, TOOL_ACTION])


class ToolBenchActionParsingTest(unittest.TestCase):
    def test_reasoning_action_is_not_executed(self):
        text = '<think>Maybe use\nAction: bad\nAction Input: {}</think>\n' + TOOL_ACTION
        name, arguments = parse_tool_action(text)
        self.assertEqual(name, "forecast")
        self.assertEqual(json.loads(arguments), {"city": "Beijing"})
        self.assertEqual(parse_tool_action('<think>Action: forecast\nAction Input: {}'), ("", "{}"))

    def test_supported_single_action_formats(self):
        for action in (
            TOOL_ACTION,
            '<tool_call>{"name": "forecast", "arguments": {"city": "Beijing"}}</tool_call>',
            '<function=forecast>{"city": "Beijing"}</function>',
        ):
            with self.subTest(action=action):
                name, arguments = parse_tool_action(action)
                self.assertEqual(name, "forecast")
                self.assertEqual(json.loads(arguments), {"city": "Beijing"})

    def test_multiple_or_mixed_calls_are_not_silently_reduced_to_one(self):
        tagged = '<tool_call>{"name": "forecast", "arguments": {}}</tool_call>'
        function = '<function=forecast>{}</function>'
        for action in (
            TOOL_ACTION + '\n' + TOOL_ACTION, tagged + tagged, function + function,
            TOOL_ACTION + '\n' + tagged, tagged + function, tagged + tagged + function,
        ):
            with self.subTest(action=action):
                self.assertEqual(parse_tool_action(action), ("", "{}"))

    def test_projection_retains_actions_and_validates_argument_object(self):
        actions = [TOOL_ACTION, 'Action: forecast\nAction Input: []', 'plain answer']
        projected, valid = toolbench_projection(actions)
        self.assertEqual(projected, actions)
        self.assertEqual(valid, [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
