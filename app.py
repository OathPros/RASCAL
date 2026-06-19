from __future__ import annotations

import csv
import difflib
import importlib
import importlib.util
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unicodedata import normalize

from flask import Flask, jsonify, render_template, request

from cohere_reranker import cohere_enabled, cohere_rerank_action, cohere_top_k
from intent_refinement import (
    build_ranking_query,
    get_intent_refinement_config,
    refine_intent_with_llm,
    should_refine_intent,
)


def load_local_dotenv() -> None:
    if importlib.util.find_spec("dotenv"):
        importlib.import_module("dotenv").load_dotenv()


load_local_dotenv()

BASE_DIR = Path(__file__).parent
CATALOGUE_PATH = Path(os.getenv("CATALOGUE_PATH", str(BASE_DIR / "data" / "catalogue.csv")))
FIELDS_PATH = BASE_DIR / "config" / "fields.json"
RANKING_WEIGHTS_PATH = BASE_DIR / "config" / "ranking_weights.csv"
DISPLAY_CANDIDATE_LIMIT = 3
LOGGER = logging.getLogger(__name__)

DEFAULT_RANKING_WEIGHTS: dict[str, dict[str, float | bool]] = {
    "title_description_overlap": {"enabled": True, "weight": 3.0, "bonus": 0.0},
    "keyword_overlap": {"enabled": True, "weight": 5.0, "bonus": 0.0},
    "priority": {"enabled": True, "weight": 0.4, "bonus": 0.0},
    "fuzzy_similarity": {"enabled": True, "weight": 10.0, "bonus": 0.0},
    "exact_phrase": {"enabled": True, "weight": 0.0, "bonus": 8.0},
}


@dataclass
class ServiceAction:
    action_id: str
    title: str
    description: str
    keywords: list[str]
    priority: int
    owner_email: str
    required_information: str
    action_description_added: str



def infer_field_shape(title: str, desc: str) -> dict[str, Any]:
    lowered = f"{title} {desc}".lower()

    if any(token in lowered for token in ["date", "when", "timing", "start", "end", "deadline", "needed by"]):
        return {"type": "text", "inputType": "date"}

    if any(token in lowered for token in ["email", "e-mail"]):
        return {"type": "text", "inputType": "email"}

    if any(token in lowered for token in ["phone", "telephone", "mobile", "contact number"]):
        return {"type": "text", "inputType": "tel"}

    if any(token in lowered for token in ["department", "faculty", "campus", "location", "environment", "urgency", "priority"]):
        return {"type": "dropdown", "choices": ["Please select", "N/A"]}

    if any(token in lowered for token in ["approval", "manager", "consent", "confirm", "yes/no", "yes or no"]):
        return {"type": "radiogroup", "choices": ["Yes", "No"]}

    if any(token in lowered for token in ["account", "role", "group", "permission", "profile", "access level"]):
        return {"type": "dropdown", "choices": ["Please select", "Other"]}

    if len(desc) > 70 or any(token in lowered for token in ["details", "description", "reason", "comments", "explain"]):
        return {"type": "comment", "inputType": "text"}

    return {"type": "text", "inputType": "text"}

def parse_required_information(raw: str) -> list[dict[str, Any]]:
    if not raw:
        return []

    cleaned = raw.replace("\\n", "\n").strip().strip('"')
    chunks = [c.strip(" -\t") for c in re.split(r"\n+|;", cleaned) if c.strip()]
    fields: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks, start=1):
        title, _, desc = chunk.partition(":")
        title = title.strip()
        desc = desc.strip()
        field_name = slugify(title) or f"required_info_{idx}"
        field_shape = infer_field_shape(title, desc)
        field: dict[str, Any] = {
            "name": field_name,
            "title": title,
            "type": field_shape.get("type", "text"),
            "required": True,
        }

        if field_shape.get("inputType"):
            field["inputType"] = field_shape["inputType"]
        if field_shape.get("choices"):
            field["choices"] = field_shape["choices"]

        if desc:
            field["description"] = desc

        fields.append(field)

    return fields


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def slugify(value: str) -> str:
    text = normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return "_".join(tokenize(text))


def parse_priority(value: str) -> int:
    cleaned = (value or "").strip()
    if not cleaned:
        return 0
    try:
        return int(cleaned)
    except ValueError:
        return 0


def parse_keywords(raw: str) -> list[str]:
    if not raw:
        return []
    bits = re.split(r"[,;]|\bor\b", raw, flags=re.IGNORECASE)
    return [b.strip().lower() for b in bits if b.strip()]


def row_to_action(row: dict[str, str]) -> ServiceAction:
    action_id = (row.get("action_id") or row.get("Process ID") or "").strip()
    service_name = (row.get("Service Name") or row.get("title") or "").strip()
    action_name = (row.get("Action") or "").strip()
    action_desc = (row.get("Service Action Description (added)") or row.get("Service Action Description") or row.get("description") or "").strip()
    service_desc = (row.get("Entity Description (Added)") or row.get("Service Description") or "").strip()

    title = action_name or service_name or "Untitled service action"
    description = " ".join(part for part in [action_desc, service_desc] if part)
    if not action_id:
        action_id = slugify(" ".join(part for part in [service_name, action_name] if part))

    keyword_parts = [
        row.get("keywords") or "",
        row.get("Service Entity Name") or "",
        row.get("Categorization Level 1") or "",
        row.get("Categorization Level 2") or "",
        row.get("Categorization Level 3") or "",
        row.get("Categorization Level 4") or "",
        row.get("Entity Hamburger L1") or "",
        row.get("Entity Hamburger L2") or "",
        row.get("Qualified Requestors") or "",
        row.get("Required Information (Name, Description, Restrictions)") or "",
    ]

    owner_email = (
        row.get("owner_email")
        or row.get("Entity Owner Contact Email")
        or row.get("Send Request to")
        or "service-desk@company.test"
    ).strip()
    required_information = (
        row.get("Required Information (Name, Description, Restrictions)")
        or row.get("required_information")
        or row.get("required info")
        or ""
    ).strip()

    return ServiceAction(
        action_id=action_id,
        title=title,
        description=description,
        keywords=parse_keywords(";".join(keyword_parts)),
        priority=parse_priority(row.get("priority") or ""),
        owner_email=owner_email,
        required_information=required_information,
        action_description_added=action_desc,
    )


def load_catalogue(path: Path = CATALOGUE_PATH) -> list[ServiceAction]:
    actions: list[ServiceAction] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.excel
        if sample:
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",|\t;")
            except csv.Error:
                dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        for row in reader:
            action = row_to_action(row)
            if action.action_id and action.action_id.lower() != "process_id":
                actions.append(action)
    return actions


def load_fields(path: Path = FIELDS_PATH) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_ranking_weights(path: Path = RANKING_WEIGHTS_PATH) -> dict[str, dict[str, float | bool]]:
    defaults = {signal: values.copy() for signal, values in DEFAULT_RANKING_WEIGHTS.items()}
    if not path.exists():
        return defaults

    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except (OSError, csv.Error, UnicodeDecodeError):
        return defaults

    required_columns = {"signal", "enabled", "weight", "bonus"}
    if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
        return defaults

    loaded: dict[str, dict[str, float | bool]] = {}
    try:
        for row in rows:
            signal = (row.get("signal") or "").strip()
            if signal not in defaults:
                continue
            enabled = (row.get("enabled") or "").strip().lower()
            if enabled not in {"true", "false"}:
                return defaults
            loaded[signal] = {
                "enabled": enabled == "true",
                "weight": float(row.get("weight") or 0),
                "bonus": float(row.get("bonus") or 0),
            }
    except (TypeError, ValueError):
        return defaults

    if set(loaded) != set(defaults):
        return defaults
    return loaded


def ranking_signal(weights: dict[str, dict[str, float | bool]], signal: str) -> dict[str, float | bool]:
    return weights.get(signal, DEFAULT_RANKING_WEIGHTS[signal])


def signal_weight(weights: dict[str, dict[str, float | bool]], signal: str) -> float:
    config = ranking_signal(weights, signal)
    return float(config["weight"]) if config["enabled"] else 0.0


def signal_bonus(weights: dict[str, dict[str, float | bool]], signal: str) -> float:
    config = ranking_signal(weights, signal)
    return float(config["bonus"]) if config["enabled"] else 0.0


def score_action(
    query_tokens: list[str],
    action: ServiceAction,
    query_text: str = "",
    weights: dict[str, dict[str, float | bool]] | None = None,
) -> dict[str, Any]:
    ranking_weights = weights or load_ranking_weights()
    text_tokens = set(tokenize(action.title + " " + action.description))
    keyword_tokens = set(tokenize(" ".join(action.keywords)))
    q = set(query_tokens)

    matched_title_description_tokens = sorted(q & text_tokens)
    matched_keyword_tokens = sorted(q & keyword_tokens)
    title_description_overlap_count = len(matched_title_description_tokens)
    keyword_overlap_count = len(matched_keyword_tokens)

    title_description_score = title_description_overlap_count * signal_weight(
        ranking_weights, "title_description_overlap"
    )
    keyword_score = keyword_overlap_count * signal_weight(ranking_weights, "keyword_overlap")
    priority_score = action.priority * signal_weight(ranking_weights, "priority")

    haystack = f"{action.title} {action.description} {' '.join(action.keywords)}".lower()
    fuzzy_ratio = difflib.SequenceMatcher(a=query_text, b=haystack).ratio()
    fuzzy_score = fuzzy_ratio * signal_weight(ranking_weights, "fuzzy_similarity")
    exact_phrase_match = bool(query_text and query_text in haystack)
    exact_phrase_bonus = (
        signal_bonus(ranking_weights, "exact_phrase") if exact_phrase_match else 0.0
    )

    total_score = (
        title_description_score + keyword_score + priority_score + fuzzy_score + exact_phrase_bonus
    )
    return {
        "action_id": action.action_id,
        "title": action.title,
        "total_score": total_score,
        "title_description_overlap_count": title_description_overlap_count,
        "title_description_score": title_description_score,
        "keyword_overlap_count": keyword_overlap_count,
        "keyword_score": keyword_score,
        "priority": action.priority,
        "priority_score": priority_score,
        "fuzzy_ratio": fuzzy_ratio,
        "fuzzy_score": fuzzy_score,
        "exact_phrase_match": exact_phrase_match,
        "exact_phrase_bonus": exact_phrase_bonus,
        "matched_title_description_tokens": matched_title_description_tokens,
        "matched_keyword_tokens": matched_keyword_tokens,
    }


def local_rank(query: str, actions: list[ServiceAction], top_n: int = 5) -> list[dict[str, Any]]:
    q_tokens = tokenize(query)
    query_text = " ".join(q_tokens)
    ranking_weights = load_ranking_weights()
    scored = []
    for action in actions:
        breakdown = score_action(q_tokens, action, query_text, ranking_weights)
        scored.append((breakdown["total_score"], action, breakdown))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "action_id": a.action_id,
            "title": a.title,
            "description": a.description,
            "score": round(s, 3),
            "owner_email": a.owner_email,
            "action_description_added": a.action_description_added,
            "ranking_breakdown": {
                key: round(value, 3) if isinstance(value, float) else value
                for key, value in breakdown.items()
            },
        }
        for s, a, breakdown in scored[:top_n]
    ]


def catalogue_service_areas(actions: list[ServiceAction]) -> list[str]:
    areas: set[str] = set()
    for action in actions:
        for keyword in action.keywords:
            cleaned = str(keyword).strip()
            if 2 <= len(cleaned) <= 80:
                areas.add(cleaned)
    return sorted(areas)[:50]


def select_ranked_candidates(search_query: str) -> dict[str, Any]:
    top_k = cohere_top_k()
    candidates = local_rank(search_query, ACTIONS, top_n=top_k)
    local_best = candidates[0] if candidates else None
    cohere_candidates = candidates[:top_k]
    cohere_was_enabled = cohere_enabled()
    cohere_best = cohere_rerank_action(search_query, cohere_candidates)
    selected_candidate = cohere_best or local_best
    selected = selected_candidate["action_id"] if selected_candidate else None
    selection_source = "cohere" if cohere_best else "local" if local_best else "none"

    display_action_ids = []
    if selected:
        display_action_ids.append(selected)
    for candidate in candidates:
        if candidate["action_id"] not in display_action_ids:
            display_action_ids.append(candidate["action_id"])
        if len(display_action_ids) >= DISPLAY_CANDIDATE_LIMIT:
            break

    candidate_map = {c["action_id"]: c for c in candidates}
    enhanced_candidates = []
    for action_id in display_action_ids:
        action = next((a for a in ACTIONS if a.action_id == action_id), None)
        if not action:
            continue
        fields = fields_for_action(action)
        candidate = candidate_map[action_id] | {
            "required_fields": [f["name"] for f in fields if f.get("required")],
            "field_count": len(fields),
        }
        enhanced_candidates.append(candidate)

    return {
        "query": search_query,
        "selected_action_id": selected,
        "selection_source": selection_source,
        "candidates": enhanced_candidates,
        "ranking_debug": {
            "selection_source": selection_source,
            "local_best_action_id": local_best["action_id"] if local_best else None,
            "cohere_best_action_id": cohere_best["action_id"] if cohere_best else None,
            "cohere_score": cohere_best.get("cohere_relevance_score") if cohere_best else None,
            "cohere_enabled": cohere_was_enabled,
            "cohere_top_k": top_k,
            "fallback_used": cohere_was_enabled and local_best is not None and cohere_best is None,
        },
        "all_candidates": candidates,
    }


def fields_for_action(action: ServiceAction) -> list[dict[str, Any]]:
    return (
        FIELD_MAP.get(action.action_id)
        or parse_required_information(action.required_information)
        or infer_generic_fields(action)
    )


def infer_generic_fields(action: ServiceAction) -> list[dict[str, Any]]:
    return [
        {"name": "requester_name", "title": "Requester Name", "type": "text", "required": True, "inputType": "text"},
        {
            "name": "contact_email",
            "title": "Contact Email",
            "type": "text",
            "required": True,
            "inputType": "email",
        },
        {
            "name": "requested_for",
            "title": "Is this request for you or someone else?",
            "type": "radiogroup",
            "required": True,
            "choices": ["Myself", "Another user"],
        },
        {
            "name": "urgency",
            "title": "Urgency",
            "type": "dropdown",
            "required": True,
            "choices": ["Low", "Medium", "High", "Critical"],
        },
        {"name": "request_details", "title": f"Details for {action.title}", "type": "comment", "required": True},
    ]


def to_survey_schema(action: ServiceAction, fields: list[dict[str, Any]]) -> dict[str, Any]:
    elements = []
    required = []
    for field in fields:
        element = {
            "type": field.get("type", "text"),
            "name": field["name"],
            "title": field["title"],
            "description": field.get("description", ""),
        }

        if field.get("choices"):
            element["choices"] = field["choices"]

        if field.get("inputType"):
            element["inputType"] = field["inputType"]

        if field.get("required"):
            required.append(field["name"])
            element["isRequired"] = True

        elements.append(element)

    return {
        "title": action.title,
        "description": action.description,
        "elements": elements,
        "required": required,
    }


def validate_submission(schema: dict[str, Any], answers: dict[str, Any]) -> list[str]:
    errors = []
    for req in schema.get("required", []):
        val = answers.get(req)
        if val is None or (isinstance(val, str) and not val.strip()):
            errors.append(f"Missing required field: {req}")
    return errors


def build_halo_email(action: ServiceAction, answers: dict[str, Any]) -> dict[str, Any]:
    lines = [f"Action: {action.action_id}", f"Title: {action.title}", "", "Submitted Fields:"]
    for k, v in answers.items():
        lines.append(f"- {k}: {v}")

    return {
        "to": action.owner_email,
        "subject": f"[Halo Simulation] {action.title}",
        "body": "\n".join(lines),
    }


app = Flask(__name__)
ACTIONS = load_catalogue(CATALOGUE_PATH)
FIELD_MAP = load_fields()


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/catalogue")
def api_catalogue():
    return jsonify([a.__dict__ for a in ACTIONS])


@app.post("/api/chat")
def api_chat():
    payload = request.get_json(force=True, silent=True) or {}
    query = (payload.get("message") or "").strip()
    if not query:
        return jsonify({"error": "message is required"}), 400

    context_messages = payload.get("context") or []
    if not isinstance(context_messages, list):
        context_messages = []
    clean_context = [str(message).strip() for message in context_messages if str(message).strip()]
    search_query = " ".join([*clean_context, query]).strip()

    ranking_query = search_query
    initial_candidates = local_rank(search_query, ACTIONS, top_n=cohere_top_k())
    intent_config = get_intent_refinement_config()
    refined_intent = None
    intent_used = False

    # Clarification state is request-scoped and client-provided. A frontend can pass
    # the previous clarification response's intent_state back in this field.
    previous_intent_state = payload.get("intent_state")
    if not isinstance(previous_intent_state, dict):
        previous_intent_state = None

    if intent_config.enabled and should_refine_intent(initial_candidates, search_query):
        refined_intent = refine_intent_with_llm(
            search_query,
            initial_candidates[:5],
            catalogue_service_areas(ACTIONS),
            previous_intent_state,
            intent_config,
        )
        if refined_intent:
            intent_used = True
            try:
                turn_count = int(previous_intent_state.get("turn_count", 0)) if previous_intent_state else 0
            except (TypeError, ValueError):
                turn_count = 0
            refined_intent["turn_count"] = turn_count + 1

            if (
                refined_intent["status"] == "needs_clarification"
                and refined_intent.get("clarifying_question")
                and refined_intent["turn_count"] <= intent_config.max_turns
            ):
                return jsonify({
                    "type": "clarification",
                    "question": refined_intent["clarifying_question"],
                    "intent_state": refined_intent,
                })

            if refined_intent["status"] == "ready_to_rank" or refined_intent["turn_count"] > intent_config.max_turns:
                improved_query = build_ranking_query(refined_intent)
                if improved_query and (
                    refined_intent["confidence"] >= intent_config.min_confidence
                    or refined_intent.get("clarifying_question")
                    or refined_intent["turn_count"] > intent_config.max_turns
                ):
                    ranking_query = improved_query
                elif not improved_query:
                    LOGGER.warning("Intent refinement produced an empty ranking query; falling back to original ranking")
                else:
                    LOGGER.warning("Intent refinement confidence was below threshold without clarification; falling back to original ranking")
            elif refined_intent["status"] in {"out_of_scope", "unsupported"}:
                return jsonify({
                    "type": refined_intent["status"],
                    "message": refined_intent.get("clarifying_question")
                    or "I can only help route YorkU service catalogue requests. Please ask about a YorkU service, request, or support pathway.",
                    "intent_state": refined_intent,
                    "query": search_query,
                    "latest_message": query,
                    "context": clean_context,
                    "candidates": [],
                    "selected_action_id": None,
                    "selection_source": "intent_refinement",
                })

    ranked = select_ranked_candidates(ranking_query)

    response = {
        "query": search_query,
        "latest_message": query,
        "context": clean_context,
        "selected_action_id": ranked["selected_action_id"],
        "selection_source": ranked["selection_source"],
        "candidates": ranked["candidates"],
        "ranking_debug": ranked["ranking_debug"],
    }
    if intent_config.debug:
        response.update({
            "type": "ranked_results",
            "original_query": search_query,
            "ranking_query": ranking_query,
            "intent_refinement": {
                "used": intent_used,
                "status": refined_intent.get("status") if refined_intent else None,
                "confidence": refined_intent.get("confidence") if refined_intent else None,
                "user_goal": refined_intent.get("user_goal") if refined_intent else None,
                "normalized_query": refined_intent.get("normalized_query") if refined_intent else None,
                "ranking_keywords": refined_intent.get("ranking_keywords") if refined_intent else [],
            },
            "initial_candidates": initial_candidates,
            "final_candidates": ranked["all_candidates"],
        })

    return jsonify(response)


@app.get("/api/form/<action_id>")
def api_form(action_id: str):
    action = next((a for a in ACTIONS if a.action_id == action_id), None)
    if not action:
        return jsonify({"error": "unknown action_id"}), 404

    fields = fields_for_action(action)
    schema = to_survey_schema(action, fields)
    return jsonify({"action": action.__dict__, "schema": schema})


@app.post("/api/submit")
def api_submit():
    payload = request.get_json(force=True, silent=True) or {}
    action_id = payload.get("action_id")
    answers = payload.get("answers") or {}

    action = next((a for a in ACTIONS if a.action_id == action_id), None)
    if not action:
        return jsonify({"error": "unknown action_id"}), 404

    fields = fields_for_action(action)
    schema = to_survey_schema(action, fields)
    errors = validate_submission(schema, answers)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    email_payload = build_halo_email(action, answers)
    return jsonify({"ok": True, "halo_email": email_payload})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
