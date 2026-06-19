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
- Optional Claude Haiku / LLM target intent clarification layer that only normalizes
  unclear requests before deterministic catalogue ranking.
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

## Optional intent clarification with Claude Haiku or another LLM target
Intent refinement is disabled by default. When enabled, `/api/chat` first performs
local ranking, then calls the configured LLM target only when the local ranking
looks ambiguous or the request uses vague language. The LLM never selects the
final service action; it can only return structured intent metadata, ask one
clarifying question, or produce a stronger query that is sent back through the
existing local ranking and optional Cohere reranking pipeline.

For Claude Haiku through an institutional target URL, set these values in your
local `.env`:
- `INTENT_REFINEMENT_ENABLED=true`
- `INTENT_REFINEMENT_PROVIDER=anthropic`
- `INTENT_REFINEMENT_TARGET_URL=<your Claude Haiku target URL>`
- `INTENT_REFINEMENT_API_KEY=<your API key, if that is the provided secret>`
- `INTENT_REFINEMENT_API_CODE=<your API code, if that is the provided secret>`
- `INTENT_REFINEMENT_AUTH_HEADER=x-api-key`
- `INTENT_REFINEMENT_MODEL=claude-haiku-4-5`
- `INTENT_REFINEMENT_TIMEOUT_SECONDS=8`
- `INTENT_REFINEMENT_MAX_TURNS=2`
- `INTENT_REFINEMENT_MIN_CONFIDENCE=0.65`
- `INTENT_REFINEMENT_DEBUG=false`

`INTENT_REFINEMENT_API_KEY` and `INTENT_REFINEMENT_API_CODE` are aliases; set
only the one that matches what your institution provides. Set
`INTENT_REFINEMENT_AUTH_HEADER` to the header your target expects, such as
`x-api-key`, `api-key`, or `Authorization`; `Authorization` is sent as a bearer
token. The code also keeps
backward-compatible Azure Foundry aliases (`AZURE_FOUNDRY_ENDPOINT`,
`AZURE_FOUNDRY_API_KEY`, `AZURE_FOUNDRY_MODEL`) for future provider swaps.

If intent refinement is enabled but LLM settings are missing, timeout, return
invalid JSON, or otherwise fail, RASCAL logs a safe warning and continues with
the original deterministic ranking path. API keys/codes and provider headers are
never logged.

Clarification state is request-scoped. If `/api/chat` returns:

```json
{
  "type": "clarification",
  "question": "Which system do you need access to?",
  "intent_state": {}
}
```

the frontend should send the user's next answer along with that `intent_state`:

```json
{
  "message": "SharePoint",
  "intent_state": {}
}
```
