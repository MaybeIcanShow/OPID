"""Remote adapter for StableToolBench's final-answer-correctness (FAC) judge.

The official FAC user prompt is retained verbatim and sent through the selected
model's normal chat template. ``stabletoolbench_fac`` is intended for the dedicated
``stabletoolbench/Evaluator`` model. ``fac_prompt`` supports a user-selected judge,
including MirrorAPI, but its results are not official StableToolBench FAC scores.
MirrorAPI and MirrorAPI-Cache were originally trained to simulate tools. This adapter
never interprets a Finish action as proof of success. Missing judges and malformed
judge responses have no score and must be excluded from success-rate denominators.

Source (Apache-2.0):
https://github.com/THUNLP-MT/StableToolBench/blob/aa4ed9f4737ad98bd706663f01d63623c3427812/toolbench/tooleval/fac_eval.py
The FAC prompt below is unmodified; HTTP transport and strict parsing are local.
FAC assesses answer completeness, not a separate factual-grounding guarantee.
"""
# ruff: noqa: E501
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from threading import BoundedSemaphore
from typing import Any, Literal, Mapping
from urllib.parse import urlsplit

import requests

FAC_SOURCE_REVISION = "aa4ed9f4737ad98bd706663f01d63623c3427812"
FAC_PROMPT = """
gpt-4-turbo

Given a query and an answer provided by an AI agent, you now need to determine the answer_status of whether the well solved the query, i.e. whether the need of the query is satisfied. You need to output "Unsolved" or "Solved" and your reason. You must obey the following rules:

You should response "Solved" when:
    1. If the answer well provides the information needed by the query, then it is "Solved". The answer does not need to be perfect, and it only needs to make a genuine attempt to address the query.
	2.	Consider only Completeness:
	    •	The answer attempts to address every part of the query, regardless of whether the information provided is factually correct or accurate, unless there is a severe factual error.
	3.	For Multi-part Queries:
	    •	For queries with multiple parts, all parts must be addressed for the answer to be considered “Solved”.
	4.	Genuine Attempt :
	    •	The answer makes a genuine attempt to provide the requested information or perform the requested task for all parts of the query. This includes scenarios where the answer concludes that “nothing” is a reasonable response (e.g., when the requested information does not exist or is not available, or a possible answer of the query is nothing and the model answers nothing after reasonable attempts).

You should response "Unsolved" when:
    1.	Refusal, Apology, or Non-engagement:
	    •	The answer includes a refusal or apology (e.g., “I'm sorry, I can't help with that”).
	    •	The answer does not directly engage with or address the query in any way.
	2.	Multi-part Queries:
	    •	If the query has multiple parts and at least one part is not well addressed.
    3. Severe Factual Error:
        •   If the answer contains a severe factual error that significantly impacts the usefulness of the information provided.


Additional Guidelines:
    1. VERY IMPORTANT: DO NOT BE TOO HARSH. The model does not need to be perfect, and the answer does not need to be flawless. It only needs to make a genuine attempt to address the query.
    1. DO NOT evaluate factual accuracy or correctness of the information provided based on your knowledge. Assume that the information provided is accurate and focus solely on whether the answer attempts to address all parts of the query, unless there is a severe factual error that conficts common knowledge.
	2.	Focus on Final Answer: Only the final answer is provided and should be considered, disregarding any processes that were used to generate the answer. You only need to judge whether the information need is satisfied.
	3.	Answer Completion: The agent does not need to detail how it arrived at the answer, only that the answer itself is complete and attempts to address the query.

Here are some examples:

Query 1: I'm planning a movie night with my friends and we want to watch a horror film. Can you recommend some popular horror movies available on streaming platforms in the US? Also, provide the runtime and IMDb ratings for these movies.
Answer 1: Here are some popular horror movies available on streaming platforms in the US:\n\n1. Knives Out\n   - Runtime: 130 minutes\n   - IMDb Rating: 7.9/10\n   - Available on: Netflix, Prime Video, Hulu, Amazon Prime Video\n\n2. Jumanji: The Next Level\n   - Runtime: 110 minutes\n   - IMDb Rating: 6.7/10\n   - Available on: Hulu, Amazon Prime Video, Netflix\n\nPlease note that availability may vary depending on your location and streaming platform subscriptions. Enjoy your movie night!
Answer Status: Solved0304
Reason: The answer addressed all parts of subqueries by providing a list of popular horror movies available on streaming platforms in the US, along with their runtime and IMDb ratings. Whether the film is horror is a factual matter that does not to be checked.

Query 2: I'm a screenwriter looking for inspiration for my next project. Can you help me find a list of critically acclaimed movies available on streaming platforms in the US? Also, provide me with the streaming sources for the movie 'Citizen Kane' and the basic information about the cast, including their names and professions.
Answer 2: The movie 'Citizen Kane' is available on the following streaming platforms:
- HBO Max: [Watch on HBO Max](https://play.hbomax.com/feature/urn:hbo:feature:GXduU_gwwz-NvjAEAAAAC)
- Amazon Prime Video: [Watch on Amazon Prime Video](https://www.amazon.com/Citizen-Kane-Orson-Welles/dp/B000I9YLWG)
- AppleTV+: [Watch on AppleTV+](https://tv.apple.com/us/movie/citizen-kane/umc.cmc.21zj0v11gnqbbqrebieh0vpk7)
- Disney+: Available on Disney+
- Netflix: Available on Netflix
Answer Status: Unsolved
Reason: The answer only addressed the first and second part of the query by providing the streaming sources for the movie 'Citizen Kane' but did not provide information about the cast or a list of critically acclaimed movies available on streaming platforms in the US. The response was incomplete and did not fully address the query.

Query:\x20
{query}
Answer:\x20
{answer}

Now give your reason and answer status in the following format:

Answer Status
xxx (can only be "Solved" or "Unsolved")
Reason
xxx
"""


@dataclass(frozen=True)
class EvaluationResult:
    status: Literal["success", "failure", "unscored", "error"]
    score: float | None
    reason: str

    @property
    def scored(self) -> bool:
        return self.score is not None


def parse_fac_response(content: Any) -> EvaluationResult:
    """Parse the explicit FAC answer-status field, never a substring in a reason.

    Upstream scans for ``solved``/``unsolved`` anywhere, including explanations.
    Here malformed, ambiguous, or truncated output is an evaluator error instead
    of a false failure/success. JSON is not the official FAC output protocol.
    """
    if not isinstance(content, str):
        return EvaluationResult("error", None, "fac_response_content_not_text")
    labels = re.findall(r"(?im)^[ \t]*Answer[ \t]+Status\b", content)
    if not labels:
        # The user-selected MirrorAPI judge emits a bare status on its first
        # line, followed by its reason. Accept that observed protocol without
        # falling back to upstream's unsafe substring search.
        bare = re.fullmatch(r"\s*(Solved|Unsolved)[ \t]*\r?\n(.+?)\s*", content, flags=re.IGNORECASE | re.DOTALL)
        if bare is not None and bare.group(2).strip():
            reason = bare.group(2).strip()
            if not re.search(r"(?im)^[ \t]*(Solved|Unsolved)[ \t]*$", reason):
                solved = bare.group(1).lower() == "solved"
                return EvaluationResult("success" if solved else "failure", float(solved), reason)
        return EvaluationResult("error", None, "fac_response_missing_or_ambiguous_status")
    if len(labels) != 1:
        return EvaluationResult("error", None, "fac_response_missing_or_ambiguous_status")
    match = re.fullmatch(
        r"\s*Answer[ \t]+Status[ \t]*:?[ \t]*(?:\r?\n[ \t]*)?"
        r"(Solved|Unsolved)[ \t]*\r?\n[ \t]*"
        r"Reason[ \t]*:?[ \t]*(?:\r?\n[ \t]*)?(.+?)\s*",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None or not match.group(2).strip():
        return EvaluationResult("error", None, "fac_response_invalid_format")
    solved = match.group(1).lower() == "solved"
    return EvaluationResult("success" if solved else "failure", float(solved), match.group(2).strip())


class StableToolBenchEvaluator:
    """Use a separately configured FAC judge over OpenAI-compatible chat HTTP.

    Configuration keys are ``api_base`` (including ``/v1``), ``model``, ``api_key``
    or ``api_key_env``, ``request_timeout`` (seconds), ``max_tokens``, ``max_concurrency`` and ``enabled``.
    ``mode`` defaults to ``fac_prompt``. ``stabletoolbench_fac`` describes use of
    the dedicated StableToolBench Evaluator with its chat template. A different
    judge using the same prompt must be reported as a custom judge metric.
    No endpoint, model, or credential is borrowed from the tool simulator.
    """

    def __init__(self, config: Mapping[str, Any] | None = None):
        config = config or {}
        self.enabled = bool(config.get("enabled", True))
        self.mode = str(config.get("mode", "fac_prompt"))
        self.api_base = str(config.get("api_base", "") or "").strip().rstrip("/")
        self.model = str(config.get("model", "") or "").strip()
        key_env = str(config.get("api_key_env", "TOOLBENCH_EVALUATOR_API_KEY") or "")
        self._api_key = str(config.get("api_key", "") or os.environ.get(key_env, ""))
        self._configuration_error = ""
        max_concurrency = 8
        try:
            self.request_timeout = float(config.get("request_timeout", 60))
            self.max_tokens = int(config.get("max_tokens", 512))
            max_concurrency = int(config.get("max_concurrency", 8))
            if not math.isfinite(self.request_timeout) or self.request_timeout <= 0 or self.max_tokens <= 0 or max_concurrency <= 0:
                self._configuration_error = "fac_invalid_timeout_or_token_limit"
        except (TypeError, ValueError, OverflowError):
            self._configuration_error = "fac_invalid_timeout_or_token_limit"
        self._semaphore = BoundedSemaphore(max(1, max_concurrency))
        try:
            url = urlsplit(self.api_base)
            if self.api_base and (
                url.scheme not in ("http", "https") or not url.netloc or url.username
                or url.password or url.query or url.fragment
            ):
                self._configuration_error = "fac_invalid_api_base"
        except ValueError:
            self._configuration_error = "fac_invalid_api_base"
        if self.mode not in ("stabletoolbench_fac", "fac_prompt"):
            self._configuration_error = "fac_unsupported_mode"

    @property
    def configured(self) -> bool:
        return bool(self.enabled and self.api_base and self.model and not self._configuration_error)

    def evaluate(self, query: str, final_answer: str) -> EvaluationResult:
        if not self.enabled:
            return EvaluationResult("unscored", None, "fac_evaluator_disabled")
        if self._configuration_error:
            return EvaluationResult("error", None, self._configuration_error)
        if not self.api_base or not self.model:
            return EvaluationResult("unscored", None, "fac_evaluator_not_configured")
        if not isinstance(query, str) or not query.strip():
            return EvaluationResult("unscored", None, "fac_missing_query")
        if not isinstance(final_answer, str) or not final_answer.strip():
            return EvaluationResult("unscored", None, "fac_missing_final_answer")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": FAC_PROMPT.format(query=query, answer=final_answer)}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "n": 1,
            "stream": False,
        }
        try:
            with self._semaphore, requests.Session() as session:
                # Internal model endpoints must not inherit workstation proxies.
                # Sessions stay request-local because evaluation runs in threads.
                session.trust_env = False
                response = session.post(
                    f"{self.api_base}/chat/completions", json=payload, headers=headers,
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                result = response.json()
        except requests.Timeout:
            return EvaluationResult("error", None, "fac_request_timeout")
        except requests.RequestException:
            # Do not expose response bodies, URLs, headers, or exception text:
            # authentication errors and proxies can echo credentials.
            return EvaluationResult("error", None, "fac_request_failed")
        except ValueError:
            return EvaluationResult("error", None, "fac_response_invalid_json")
        if not isinstance(result, dict):
            return EvaluationResult("error", None, "fac_response_invalid_envelope")
        choices = result.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            return EvaluationResult("error", None, "fac_response_invalid_choices")
        choice = choices[0]
        if choice.get("finish_reason") != "stop":
            return EvaluationResult("error", None, "fac_response_incomplete")
        message = choice.get("message")
        if not isinstance(message, dict):
            return EvaluationResult("error", None, "fac_response_invalid_message")
        return parse_fac_response(message.get("content"))
