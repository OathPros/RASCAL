from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request as urlrequest

from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).parent
CATALOGUE_PATH = Path(os.getenv("CATALOGUE_PATH", str(BASE_DIR / "data" / "catalogue.sample.csv")))
FIELDS_PATH = BASE_DIR / "config" / "fields.json"


@dataclass
class ServiceAction:
    action_id: str
    title: str
    description: str
    keywords: list[str]
    priority: int
    owner_email: str


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_catalogue(path: Path = CATALOGUE_PATH) -> list[ServiceAction]:
    actions: list[ServiceAction] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            actions.append(
                ServiceAction(
                    action_id=(row.get("action_id") or "").strip(),
                    title=(row.get("title") or "").strip(),
                    description=(row.get("description") or "").strip(),
                    keywords=[k.strip().lower() for k in (row.get("keywords") or "").split(",") if k.strip()],
                    priority=int((row.get("priority") or "0").strip() or 0),
                    owner_email=(row.get("owner_email") or "service-desk@company.test").strip(),
                )
            )
    return actions


def load_fields(path: Path = FIELDS_PATH) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def score_action(query_tokens: list[str], action: ServiceAction) -> float:
    text_tokens = set(tokenize(action.title + " " + action.description))
    keyword_tokens = set(tokenize(" ".join(action.keywords)))
    q = set(query_tokens)

    title_desc_overlap = len(q & text_tokens) * 3
    keyword_overlap = len(q & keyword_tokens) * 5
    priority_weight = action.priority * 0.4
    return title_desc_overlap + keyword_overlap + priority_weight


def local_rank(query: str, actions: list[ServiceAction], top_n: int = 5) -> list[dict[str, Any]]:
    q_tokens = tokenize(query)
    scored = []
    for action in actions:
        score = score_action(q_tokens, action)
        scored.append((score, action))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "action_id": a.action_id,
            "title": a.title,
            "description": a.description,
            "score": round(s, 3),
            "owner_email": a.owner_email,
        }
        for s, a in scored[:top_n]
    ]


def ollama_rerank(query: str, candidates: list[dict[str, Any]]) -> str | None:
    if os.getenv("OLLAMA_ENABLED", "false").lower() != "true":
        return None
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")

    prompt = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Pick the best action_id for this request. Return JSON only: {\"action_id\":\"...\"}.\n"
                    f"Query: {query}\nCandidates: {json.dumps(candidates)}"
                ),
            }
        ],
        "stream": False,
    }

    try:
        req = urlrequest.Request(url, data=json.dumps(prompt).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urlrequest.urlopen(req, timeout=5) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        content = raw.get("message", {}).get("content", "")
        parsed = json.loads(content)
        candidate_ids = {c["action_id"] for c in candidates}
        action_id = parsed.get("action_id")
        if action_id in candidate_ids:
            return action_id
    except Exception:
        return None
    return None


def infer_generic_fields(action: ServiceAction) -> list[dict[str, Any]]:
    return [
        {"name": "requester_name", "title": "Requester Name", "type": "string", "required": True},
        {"name": "request_details", "title": f"Details for {action.title}", "type": "text", "required": True},
        {"name": "urgency", "title": "Urgency", "type": "string", "required": False},
    ]


def to_survey_schema(action: ServiceAction, fields: list[dict[str, Any]]) -> dict[str, Any]:
    elements = []
    required = []
    for field in fields:
        q_type = "comment" if field["type"] == "text" else field["type"]
        elements.append({"type": q_type, "name": field["name"], "title": field["title"]})
        if field.get("required"):
            required.append(field["name"])

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

    candidates = local_rank(query, ACTIONS)
    best = candidates[0]["action_id"] if candidates else None
    llm_choice = ollama_rerank(query, candidates)
    selected = llm_choice or best

    return jsonify({
        "query": query,
        "selected_action_id": selected,
        "selection_source": "ollama" if llm_choice else "local",
        "candidates": candidates,
    })


@app.get("/api/form/<action_id>")
def api_form(action_id: str):
    action = next((a for a in ACTIONS if a.action_id == action_id), None)
    if not action:
        return jsonify({"error": "unknown action_id"}), 404

    fields = FIELD_MAP.get(action_id) or infer_generic_fields(action)
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

    fields = FIELD_MAP.get(action_id) or infer_generic_fields(action)
    schema = to_survey_schema(action, fields)
    errors = validate_submission(schema, answers)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    email_payload = build_halo_email(action, answers)
    return jsonify({"ok": True, "halo_email": email_payload})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
