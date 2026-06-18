from __future__ import annotations

import importlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_COHERE_MODEL = "rerank-v4.0-fast"
DEFAULT_COHERE_TOP_K = 25
DEFAULT_COHERE_TIMEOUT_SECONDS = 3.0

CANDIDATE_TEXT_FIELDS = [
    "action_id",
    "name",
    "title",
    "description",
    "keywords",
    "category",
    "service",
    "action",
    "audience",
    "request_type",
    "support_group",
    "action_description_added",
]


def cohere_enabled() -> bool:
    return os.getenv("COHERE_ENABLED", "false").strip().lower() == "true"


def cohere_top_k() -> int:
    raw_value = os.getenv("COHERE_TOP_K", str(DEFAULT_COHERE_TOP_K)).strip()
    try:
        return max(1, int(raw_value))
    except ValueError:
        logger.warning("Invalid COHERE_TOP_K=%r; using default %s", raw_value, DEFAULT_COHERE_TOP_K)
        return DEFAULT_COHERE_TOP_K


def cohere_timeout_seconds() -> float:
    raw_value = os.getenv("COHERE_TIMEOUT_SECONDS", str(DEFAULT_COHERE_TIMEOUT_SECONDS)).strip()
    try:
        return max(0.1, float(raw_value))
    except ValueError:
        logger.warning(
            "Invalid COHERE_TIMEOUT_SECONDS=%r; using default %s",
            raw_value,
            DEFAULT_COHERE_TIMEOUT_SECONDS,
        )
        return DEFAULT_COHERE_TIMEOUT_SECONDS


def _candidate_value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item is not None)
    if isinstance(value, dict):
        return " ".join(
            f"{key}: {_candidate_value_to_text(val)}" for key, val in value.items() if val is not None
        )
    return str(value)


def candidate_to_document(candidate: dict[str, Any]) -> str:
    parts = []
    for field in CANDIDATE_TEXT_FIELDS:
        value = _candidate_value_to_text(candidate.get(field)).strip()
        if value:
            parts.append(f"{field}: {value}")
    return "\n".join(parts)


def _response_results(response: Any) -> list[Any]:
    results = getattr(response, "results", None)
    if results is None and isinstance(response, dict):
        results = response.get("results")
    return list(results or [])


def _result_value(result: Any, key: str) -> Any:
    if isinstance(result, dict):
        return result.get(key)
    return getattr(result, key, None)


def cohere_rerank_action(query: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not cohere_enabled():
        return None
    if not candidates:
        return None

    api_key = os.getenv("COHERE_API_KEY", "").strip()
    if not api_key:
        logger.warning("COHERE_ENABLED=true but COHERE_API_KEY is missing; using local ranking fallback")
        return None

    documents = [candidate_to_document(candidate) for candidate in candidates]
    model = os.getenv("COHERE_MODEL", DEFAULT_COHERE_MODEL).strip() or DEFAULT_COHERE_MODEL

    try:
        cohere = importlib.import_module("cohere")
        client = cohere.ClientV2(api_key=api_key, timeout=cohere_timeout_seconds())
        response = client.rerank(model=model, query=query, documents=documents, top_n=1)
        results = _response_results(response)
        if not results:
            logger.warning("Cohere rerank returned no results; using local ranking fallback")
            return None

        index = _result_value(results[0], "index")
        if not isinstance(index, int) or index < 0 or index >= len(candidates):
            logger.warning("Cohere rerank returned invalid index %r; using local ranking fallback", index)
            return None

        selected = dict(candidates[index])
        score = _result_value(results[0], "relevance_score")
        if score is not None:
            selected["cohere_relevance_score"] = score
        return selected
    except Exception as exc:
        logger.warning("Cohere rerank failed; using local ranking fallback: %s", exc)
        return None
