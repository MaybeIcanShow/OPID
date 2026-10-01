"""CPU-only backend contract tests; no GPU, remote service or pytest required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import requests


ROOT = Path(__file__).resolve().parents[1]
# Import this CPU client without importing unrelated optional environment stacks.
SPEC = importlib.util.spec_from_file_location(
    "stabletoolbench_backend", ROOT / "agent_system/environments/env_package/toolbench/stabletoolbench.py"
)
backend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backend)
Client = backend.StableToolBenchClient
Error = backend.StableToolBenchError


class StableToolBenchBackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.mapping = {"category": "News, Media", "tool_name": "News Tool", "api_name": "Get Article"}
        self.config = {"cache_root": str(self.root / "cache"), "tools_root": str(self.root / "tools"), "request_timeout": 2}
        document = {
            "tool_description": "Trusted news articles.",
            "api_list": [{"name": "Get Article", "description": "Find article by source.",
                          "required_parameters": [{"name": "source", "type": "STRING", "description": "News source"}],
                          "optional_parameters": None}],
        }
        path = self.root / "tools/News_Media/news_tool.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(document))
        self.client = Client(self.config)
        self.addCleanup(self.client.close)

    def cache(self, values, api="get_article"):
        path = self.root / f"cache/News_Media/news_tool_for_News_Media/{api}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(values))
        return path

    def response(self, content=None, finish_reason="stop"):
        result = MagicMock()
        result.json.return_value = {"choices": [{"finish_reason": finish_reason, "message": {
            "content": content if content is not None else '{"error":"","response":{"title":"News"}}'
        }}]}
        return result

    def mock_session(self, result=None):
        session = MagicMock()
        session.post.return_value = result or self.response()
        self.client._local.session = session
        return session

    def test_cache_uses_category_case_and_tool_suffix_ignoring_key_order(self):
        path = self.cache({"{'source': 'Nasa', 'limit': 2}": {"error": "", "response": {"articles": ["cached"]}}})
        original = path.read_bytes()
        with patch.object(self.client, "_session", side_effect=AssertionError("Cache hit must not call model")):
            response, source = self.client.execute(self.mapping, '{"limit":2,"source":"Nasa"}')
        self.assertEqual(source, "disk_cache")
        self.assertEqual(response["response"], {"articles": ["cached"]})
        self.assertEqual(path.read_bytes(), original)

    def test_cache_distinguishes_boolean_number_and_string(self):
        self.cache({"{'source': True}": {"error": "", "response": "bool"},
                    "{'source': 1}": {"error": "", "response": "int"},
                    '{"source": "1"}': {"error": "", "response": "str"}})
        for argument, expected in ((True, "bool"), (1, "int"), ("1", "str")):
            self.assertEqual(self.client.execute(self.mapping, {"source": argument})[0]["response"], expected)
        session = self.mock_session()
        self.assertEqual(self.client.execute(self.mapping, {"source": 1.0})[1], "mirrorapi_cache")
        session.post.assert_called_once()

    def test_cache_miss_default_uses_cache_model_and_complete_document(self):
        session = self.mock_session()
        response, source = self.client.execute(self.mapping, {"source": "Nasa", "toolbench_key": "private"})
        self.assertEqual(source, "mirrorapi_cache")
        self.assertEqual(response["response"]["title"], "News")
        args, kwargs = session.post.call_args
        self.assertEqual(args[0], "http://10.8.176.56:8001/v1/chat/completions")
        self.assertEqual(kwargs["json"]["model"], "MirrorAPI-Cache")
        messages = kwargs["json"]["messages"]
        self.assertEqual(messages[0]["content"], backend.SIMULATOR_SYSTEM_PROMPT)
        for phrase in ("Trusted news articles.", "Find article by source.", "required_parameters", "News source", "optional_parameters", "Nasa"):
            self.assertIn(phrase, messages[1]["content"])
        self.assertNotIn("private", messages[1]["content"])
        self.assertEqual(kwargs["timeout"], 2)
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer EMPTY")

    def test_mirrorapi_explicit_switch_and_cache_disable(self):
        client = Client({**self.config, "backend": "mirrorapi", "cache_enabled": False})
        self.addCleanup(client.close)
        self.cache({"{}": {"error": "", "response": "disk"}})
        session = MagicMock()
        session.post.return_value = self.response()
        client._local.session = session
        self.assertEqual(client.execute(self.mapping, {})[1], "mirrorapi")
        self.assertEqual(session.post.call_args.args[0], "http://10.8.176.56:8000/v1/chat/completions")
        self.assertEqual(session.post.call_args.kwargs["json"]["model"], "MirrorAPI")

    def test_name_normalization_handles_reserved_digits_and_suffixed_tool(self):
        self.assertEqual(backend.normalize_mapping({"category": "News, Media", "tool_name": "news_tool_for_News_Media", "api_name": "id_for_news_tool"}),
                         ("News_Media", "news_tool", "is_id"))
        self.assertEqual(backend.standardize("123 Weather"), "get_123_weather")
        self.assertEqual(backend.change_name("class"), "is_class")

    def test_bad_arguments_fail_before_any_network_request(self):
        session = self.mock_session()
        for arguments in ([('x', 1)], '[]', '{bad', '{"x":1,"x":2}', {"x": float("nan")}, {"x": (1, 2)}):
            with self.subTest(arguments=arguments), self.assertRaises(Error) as raised:
                self.client.execute(self.mapping, arguments)
            self.assertEqual(raised.exception.code, "invalid_arguments")
        session.post.assert_not_called()

    def test_missing_document_is_explicit_failure(self):
        mapping = {**self.mapping, "tool_name": "absent"}
        with self.assertRaises(Error) as raised:
            self.client.execute(mapping, {})
        self.assertEqual(raised.exception.code, "invalid_resource")
        self.assertEqual(raised.exception.source, "mirrorapi_cache")

    def test_timeout_http_and_non_json_failures_remain_errors(self):
        for exception, code in ((requests.Timeout("timeout"), "request_timeout"),
                                (requests.ConnectionError("connection"), "request_failed"),
                                (ValueError("json"), "invalid_completion")):
            with self.subTest(code=code):
                session = self.mock_session()
                session.post.side_effect = exception
                with self.assertRaises(Error) as raised:
                    self.client.execute(self.mapping, {})
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(raised.exception.source, "mirrorapi_cache")
        # Errors release their semaphore slots, so the next request can succeed.
        self.mock_session()
        self.assertEqual(self.client.execute(self.mapping, {})[1], "mirrorapi_cache")

    def test_completion_schema_and_truncation_are_checked(self):
        for content in ('not JSON', '{"response":"x"}', '{"error":null,"response":"x"}',
                        '```json\n{"error":"","response":"x"}\n```'):
            with self.subTest(content=content):
                self.mock_session(self.response(content))
                with self.assertRaises(Error) as raised:
                    self.client.execute(self.mapping, {})
                self.assertEqual(raised.exception.code, "invalid_response")
        self.mock_session(self.response(finish_reason="length"))
        with self.assertRaises(Error) as raised:
            self.client.execute(self.mapping, {})
        self.assertEqual(raised.exception.code, "invalid_completion")

    def test_valid_tool_error_is_preserved(self):
        self.mock_session(self.response('{"error":"Missing source","response":""}'))
        response, source = self.client.execute(self.mapping, {})
        self.assertEqual(response["error"], "Missing source")
        self.assertEqual(source, "mirrorapi_cache")

    def test_cache_invalid_response_and_conflicting_keys_are_errors(self):
        self.cache({"{}": {"response": "bad"}})
        with self.assertRaises(Error) as raised:
            self.client.execute(self.mapping, {})
        self.assertEqual(raised.exception.code, "invalid_response")
        self.client._cache_index.cache_clear()
        self.cache({"{'a': 1, 'b': 2}": {"error": "", "response": "first"},
                    '{"b":2,"a":1}': {"error": "", "response": "second"}})
        with self.assertRaises(Error) as raised:
            self.client.execute(self.mapping, {"a": 1, "b": 2})
        self.assertEqual(raised.exception.code, "ambiguous_cache")

    def test_conflicting_cache_key_does_not_disable_unrelated_hits_or_misses(self):
        self.cache({"{'city': 'Sacramento'}": {"error": "", "response": "first"},
                    '{"city":"Sacramento"}': {"error": "", "response": "second"},
                    '{"city":"Memphis"}': {"error": "", "response": "correct"}})
        session = self.mock_session()
        response, source = self.client.execute(self.mapping, {"city": "Memphis"})
        self.assertEqual((response["response"], source), ("correct", "disk_cache"))
        session.post.assert_not_called()
        self.assertEqual(self.client.execute(self.mapping, {"city": "Paris"})[1], "mirrorapi_cache")
        with self.assertRaises(Error) as raised:
            self.client.execute(self.mapping, {"city": "Sacramento"})
        self.assertEqual(raised.exception.code, "ambiguous_cache")

    def test_proxy_disabled_and_api_key_environment_supported(self):
        self.assertFalse(self.client._session().trust_env)
        with patch.dict("os.environ", {"TEST_MIRROR_KEY": "secret"}):
            client = Client({**self.config, "api_key_env": "TEST_MIRROR_KEY"})
        self.addCleanup(client.close)
        self.assertEqual(client._api_key, "secret")
        self.assertEqual(client.max_concurrency, 8)

    def test_concurrent_requests_are_bounded(self):
        client = Client({**self.config, "max_concurrency": 2})
        self.addCleanup(client.close)
        active = 0
        peak = 0
        lock = threading.Lock()
        def post(*args, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return self.response()
        session = MagicMock()
        session.post.side_effect = post
        with patch.object(client, "_session", return_value=session), ThreadPoolExecutor(max_workers=6) as pool:
            outputs = list(pool.map(lambda _: client.execute(self.mapping, {}), range(6)))
        self.assertEqual(len(outputs), 6)
        self.assertEqual(peak, 2)

    def test_embedded_api_schema_can_supply_missing_local_document(self):
        schema = {"name": "Get Article", "description": "Embedded API", "required_parameters": [], "optional_parameters": []}
        mapping = {**self.mapping, "tool_name": "not_on_disk", "api_schema_json": json.dumps(schema), "tool_description": "Embedded tool"}
        session = self.mock_session()
        self.assertEqual(self.client.execute(mapping, {})[1], "mirrorapi_cache")
        content = session.post.call_args.kwargs["json"]["messages"][1]["content"]
        self.assertIn("Embedded API", content)
        self.assertIn("Embedded tool", content)

    def test_official_query_schema_field_names(self):
        schema = {"api_name": "Get Article", "api_description": "Official API description", "required_parameters": [], "optional_parameters": []}
        mapping = {**self.mapping, "api_schema_json": json.dumps(schema)}
        session = self.mock_session()
        self.assertEqual(self.client.execute(mapping, {})[1], "mirrorapi_cache")
        content = session.post.call_args.kwargs["json"]["messages"][1]["content"]
        self.assertIn("Official API description", content)

    def test_null_cache_entry_is_an_error_not_a_fallback(self):
        self.cache({"{}": None})
        with self.assertRaises(Error) as raised:
            self.client.execute(self.mapping, {})
        self.assertEqual(raised.exception.code, "invalid_response")
        self.assertEqual(raised.exception.source, "disk_cache")

    def test_real_installed_cache_fixture(self):
        cache_root = ROOT / "data/StableToolBench/tool_response_cache"
        path = cache_root / "News_Media/climate_news_feed_for_News_Media/get_articles.json"
        if not path.is_file():
            self.skipTest("Optional downloaded StableToolBench cache is absent")
        client = Client({**self.config, "cache_root": str(cache_root)})
        self.addCleanup(client.close)
        with patch.object(client, "_session", side_effect=AssertionError("Real fixture must be a disk hit")):
            response, source = client.execute(
                {"category": "News, Media", "tool_name": "climate_news_feed", "api_name": "Get Articles"},
                {"source": "United Nations, Nasa Climate, Carbon Brief"},
            )
        self.assertEqual(source, "disk_cache")
        self.assertEqual(response["error"], "")
        self.assertIn("articles", response["response"])


if __name__ == "__main__":
    unittest.main()
