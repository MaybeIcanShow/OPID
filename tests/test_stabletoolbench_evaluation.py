"""FAC protocol tests run without a GPU, external judge, or pytest installation."""
import hashlib
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import requests

from agent_system.environments.env_package.toolbench.evaluation import (
    FAC_PROMPT,
    StableToolBenchEvaluator,
)


class StableToolBenchEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "api_base": "http://evaluator.test:8002/v1",
            "model": "Evaluator",
            "api_key": "test-secret",
            "request_timeout": 2,
        }

    def invoke(self, content=None, envelope=None, error=None, config=None):
        response = MagicMock()
        response.json.return_value = envelope if envelope is not None else {
            "choices": [{"finish_reason": "stop", "message": {"content": content}}]
        }
        session = MagicMock()
        session.post.return_value = response
        if error is not None:
            session.post.side_effect = error
        context = MagicMock()
        context.__enter__.return_value = session
        with patch("agent_system.environments.env_package.toolbench.evaluation.requests.Session", return_value=context):
            result = StableToolBenchEvaluator(config or self.config).evaluate(
                "What is the weather in Beijing?", "I am sorry, please provide more information."
            )
        return result, session

    def test_prompt_matches_pinned_official_fac_source(self):
        self.assertEqual(
            hashlib.sha256(FAC_PROMPT.encode()).hexdigest(),
            "974e0df4d55babd3a59c5c80575f3e37ca88f9e3afd7a92c72dde7c12e02fc92",
        )

    def test_missing_evaluator_never_awards_success_or_failure(self):
        for config in ({}, {"api_base": self.config["api_base"]}, {"model": "Evaluator"}, {"enabled": False}):
            with self.subTest(config=config), patch("agent_system.environments.env_package.toolbench.evaluation.requests.Session") as factory:
                evaluator = StableToolBenchEvaluator(config)
                result = evaluator.evaluate("q", 'Finish({"return_type": "give_answer"})')
                self.assertFalse(evaluator.configured)
                self.assertEqual(result.status, "unscored")
                self.assertIsNone(result.score)
                factory.assert_not_called()

    def test_finish_apology_is_only_scored_by_judge(self):
        result, session = self.invoke("Answer Status\nUnsolved\nReason\nThe request was not addressed.")
        self.assertEqual(result.status, "failure")
        self.assertEqual(result.score, 0.0)
        self.assertTrue(result.scored)
        self.assertFalse(session.trust_env)
        args, kwargs = session.post.call_args
        self.assertEqual(args[0], "http://evaluator.test:8002/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-secret")
        self.assertEqual(kwargs["timeout"], 2)
        payload = kwargs["json"]
        self.assertEqual(payload["model"], "Evaluator")
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_tokens"], 512)
        self.assertEqual(payload["messages"], [{"role": "user", "content": FAC_PROMPT.format(
            query="What is the weather in Beijing?", answer="I am sorry, please provide more information."
        )}])

    def test_status_field_overrides_words_mentioned_in_reason(self):
        result, _ = self.invoke("Answer Status: Solved\nReason: The agent resolved every previously unsolved part.")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.scored)

    def test_observed_mirrorapi_first_line_status_protocol(self):
        for status, score in (("Solved", 1.0), ("Unsolved", 0.0)):
            with self.subTest(status=status):
                result, _ = self.invoke(status + "\nThe answer addresses the requested information.")
                self.assertEqual(result.score, score)
                self.assertEqual(result.status, "success" if score else "failure")

    def test_invalid_json_or_label_text_has_no_score(self):
        for content in (
            '{"answer_status": "Solved", "reason": "looks good"}',
            "Solved",
            "This was solved by the model.",
            "Answer Status\nUnsure\nReason\nUncertain.",
            "Answer Status\nSolved0304\nReason\nMalformed label from an example.",
            "Solved\nUnsolved\nContradictory verdicts.",
            "Solved\nAnswer Status: Unsolved\nReason: Contradictory verdicts.",
            "Answer Status\nSolved\nReason\n",
            "Answer Status: Unsolved\nReason: Bad\nAnswer Status: Solved\nReason: Good",
            None, {"answer_status": "Solved"},
        ):
            with self.subTest(content=content):
                result, _ = self.invoke(content)
                self.assertEqual(result.status, "error")
                self.assertIsNone(result.score)

    def test_envelope_and_truncation_errors_have_no_score(self):
        for envelope in (
            [], {}, {"choices": []}, {"choices": [{}]},
            {"choices": [{"finish_reason": "stop", "message": None}]},
            {"choices": [{"finish_reason": "length", "message": {"content": "Answer Status\nSolved\nReason\nComplete"}}]},
        ):
            with self.subTest(envelope=envelope):
                result, _ = self.invoke(envelope=envelope)
                self.assertEqual(result.status, "error")
                self.assertIsNone(result.score)

    def test_transport_errors_do_not_expose_credentials(self):
        for error in (requests.Timeout("test-secret"), requests.ConnectionError("test-secret"), requests.HTTPError("test-secret")):
            with self.subTest(error=type(error)):
                result, _ = self.invoke(error=error)
                self.assertEqual(result.status, "error")
                self.assertIsNone(result.score)
                self.assertNotIn("test-secret", result.reason)

    def test_invalid_response_json_is_not_task_failure(self):
        session = MagicMock()
        session.__enter__.return_value = session
        session.post.return_value.json.side_effect = ValueError("test-secret")
        with patch("agent_system.environments.env_package.toolbench.evaluation.requests.Session", return_value=session):
            result = StableToolBenchEvaluator(self.config).evaluate("query", "answer")
        self.assertEqual(result.status, "error")
        self.assertEqual(result.reason, "fac_response_invalid_json")
        self.assertIsNone(result.score)

    def test_environment_credential_is_used_without_logging(self):
        config = {**self.config, "api_key": "", "api_key_env": "FAC_TEST_KEY"}
        with patch.dict("os.environ", {"FAC_TEST_KEY": "environment-secret"}):
            result, session = self.invoke("Answer Status\nSolved\nReason\nComplete.", config=config)
        self.assertEqual(session.post.call_args.kwargs["headers"]["Authorization"], "Bearer environment-secret")
        self.assertNotIn("environment-secret", repr(result))

    def test_invalid_configuration_cannot_produce_score(self):
        changes = (
            {"api_base": "file:///tmp/model"}, {"api_base": "http://secret@judge/v1"},
            {"api_base": "http://judge/v1?key=secret"}, {"request_timeout": 0},
            {"request_timeout": float("nan")}, {"max_tokens": 0}, {"max_concurrency": 0},
            {"mode": "invented"},
        )
        for change in changes:
            with self.subTest(change=change), patch("agent_system.environments.env_package.toolbench.evaluation.requests.Session") as factory:
                evaluator = StableToolBenchEvaluator({**self.config, **change})
                result = evaluator.evaluate("q", "a")
                self.assertFalse(evaluator.configured)
                self.assertEqual(result.status, "error")
                self.assertIsNone(result.score)
                factory.assert_not_called()

    def test_explicit_custom_mirrorapi_judge_is_supported(self):
        result, session = self.invoke(
            "Answer Status\nSolved\nReason\nComplete.",
            config={**self.config, "model": "MirrorAPI", "mode": "fac_prompt"},
        )
        self.assertEqual(result.score, 1.0)
        self.assertEqual(session.post.call_args.kwargs["json"]["model"], "MirrorAPI")

    def test_missing_query_or_answer_is_unscored(self):
        for query, answer in (("", "answer"), ("query", ""), ("query", None)):
            with self.subTest(query=query, answer=answer), patch("agent_system.environments.env_package.toolbench.evaluation.requests.Session") as factory:
                result = StableToolBenchEvaluator(self.config).evaluate(query, answer)
                self.assertEqual(result.status, "unscored")
                self.assertIsNone(result.score)
                factory.assert_not_called()

    def test_remote_judge_concurrency_is_bounded(self):
        active = 0
        peak = 0
        lock = threading.Lock()
        two_started = threading.Event()
        release = threading.Event()

        class Session:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, *args, **kwargs):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                    if active == 2:
                        two_started.set()
                try:
                    if not release.wait(3):
                        raise requests.Timeout()
                    response = MagicMock()
                    response.json.return_value = {"choices": [{"finish_reason": "stop", "message": {
                        "content": "Answer Status\nSolved\nReason\nComplete."
                    }}]}
                    return response
                finally:
                    with lock:
                        active -= 1

        evaluator = StableToolBenchEvaluator({**self.config, "max_concurrency": 2})
        with patch("agent_system.environments.env_package.toolbench.evaluation.requests.Session", Session), ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(evaluator.evaluate, "q", "a") for _ in range(5)]
            try:
                self.assertTrue(two_started.wait(2), "Judge requests did not start")
                self.assertEqual(peak, 2)
            finally:
                release.set()
            self.assertTrue(all(f.result().score == 1.0 for f in futures))
        self.assertEqual(peak, 2)


if __name__ == "__main__":
    unittest.main()
