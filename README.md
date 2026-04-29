# Service Catalogue Chat Prototype

Thin deterministic backend + simple chat UI with optional Ollama ranking.

## Features
- CSV catalogue is source of truth.
- Deterministic local weighted ranking (works without LLM).
- Optional Ollama reranking in `/api/chat` with strict validation.
- Deterministic form generation:
  - explicit field map first
  - inferred generic fields fallback
- Form validation and simulated Halo email payload builder.

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open: `http://localhost:8000`

## Test prompts
- "I need a new laptop for engineering"
- "Please reset my VPN account"
- "I need software access to Figma"

## API
- `GET /api/catalogue` -> normalized catalogue preview
- `POST /api/chat` -> candidate actions and chosen action
- `GET /api/form/<action_id>` -> survey schema + ui schema
- `POST /api/submit` -> validate + simulate Halo email

## Optional Ollama
Set:
- `OLLAMA_ENABLED=true`
- `OLLAMA_MODEL=llama3.1:8b`
- `OLLAMA_URL=http://localhost:11434/api/chat`

When unavailable/failing, backend automatically falls back to deterministic local ranking.
