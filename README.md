# Service Catalogue Chat Prototype

Thin deterministic backend + simple chat UI with optional Cohere reranking.

## Data privacy model (public repo safe)
- The real service catalogue CSV should **not** be committed.
- Provide your private CSV via local filesystem path using `CATALOGUE_PATH`.
- The repository includes only `data/catalogue.sample.csv` with dummy/example records.

## Features
- CSV catalogue is source of truth.
- Deterministic local weighted ranking (works without LLM).
- Optional Cohere Rerank second-stage reranking in `/api/chat` with local fallback.
- Deterministic form generation:
  - explicit field map first
  - inferred generic fields fallback
- Form validation and simulated Halo email payload builder.

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export CATALOGUE_PATH=/absolute/path/to/your/private/catalogue.csv
python app.py
```

If `CATALOGUE_PATH` is not set, the app uses `data/catalogue.sample.csv`.

Open: `http://localhost:8000`

## Required catalogue columns
`action_id,title,description,keywords,priority,owner_email`

## Test prompts
- "I need a new laptop for engineering"
- "Please reset my VPN account"
- "I need software access to Figma"

## API
- `GET /api/catalogue` -> normalized catalogue preview
- `POST /api/chat` -> candidate actions and chosen action
- `GET /api/form/<action_id>` -> survey schema + ui schema
- `POST /api/submit` -> validate + simulate Halo email

## Optional Cohere Rerank
Local deterministic ranking always runs first. When Cohere is enabled, only the top local candidates are sent to Cohere for a second-stage rerank. When Cohere is disabled, misconfigured, unavailable, or failing, the backend automatically falls back to deterministic local ranking.

Set safe defaults in `.env.example` and put real secrets only in your local `.env` file:
- `COHERE_ENABLED=false`
- `COHERE_API_KEY=`
- `COHERE_MODEL=rerank-v4.0-fast`
- `COHERE_TOP_K=25`
- `COHERE_TIMEOUT_SECONDS=3`
