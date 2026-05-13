from __future__ import annotations

import csv
import difflib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unicodedata import normalize
from urllib import request as urlrequest

from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).parent
CATALOGUE_PATH = Path(os.getenv("CATALOGUE_PATH", str(BASE_DIR / "data" / "catalogue.csv")))
FIELDS_PATH = BASE_DIR / "config" / "fields.json"


@dataclass
class ServiceAction:
    action_id: str
    title: str
    description: str
    keywords: list[str]
    priority: int
    owner_email: str
    required_information: str


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
        field: dict[str, Any] = {
            "name": field_name,
            "title": title,
            "type": "comment" if desc else "text",
            "required": True,
            "inputType": "text",
        }

        lowered = f"{title} {desc}".lower()
        if any(token in lowered for token in ["date", "needed by", "start"]):
            field["inputType"] = "date"
        elif "email" in lowered:
            field["inputType"] = "email"

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
    action_desc = (row.get("Service Action Description") or row.get("Service Action Description (added)") or row.get("description") or "").strip()
    service_desc = (row.get("Entity Description (Added)") or row.get("Service Description") or "").strip()

    title = service_name or action_name or "Untitled service action"
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
    )


def load_catalogue(path: Path = CATALOGUE_PATH) -> list[ServiceAction]:
    actions: list[ServiceAction] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t") if sample else csv.excel
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
    query_text = " ".join(q_tokens)
    scored = []
    for action in actions:
        score = score_action(q_tokens, action)
        haystack = f"{action.title} {action.description} {' '.join(action.keywords)}".lower()
        fuzzy = difflib.SequenceMatcher(a=query_text, b=haystack).ratio() * 10
        phrase_bonus = 8 if query_text and query_text in haystack else 0
        score += fuzzy + phrase_bonus
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


def fields_for_action(action: ServiceAction) -> list[dict[str, Any]]:
    return (
        FIELD_MAP.get(action.action_id)
        or parse_required_information(action.required_information)
        or infer_generic_fields(action)
    )


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

    candidates = local_rank(query, ACTIONS)
    best = candidates[0]["action_id"] if candidates else None
    llm_choice = ollama_rerank(query, candidates)
    selected = llm_choice or best

    candidate_map = {c["action_id"]: c for c in candidates}
    enhanced_candidates = []
    for action_id in [c["action_id"] for c in candidates]:
        action = next((a for a in ACTIONS if a.action_id == action_id), None)
        if not action:
            continue
        fields = fields_for_action(action)
        candidate = candidate_map[action_id] | {
            "required_fields": [f["name"] for f in fields if f.get("required")],
            "field_count": len(fields),
        }
        enhanced_candidates.append(candidate)

    return jsonify({
        "query": query,
        "selected_action_id": selected,
        "selection_source": "ollama" if llm_choice else "local",
        "candidates": enhanced_candidates,
    })


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
