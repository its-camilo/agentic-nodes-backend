# Agentic Nodes Backend

API backend for **Supply Chain AI Agents**: demand parsing, supplier validation, logistics routing, negotiation, and execution planning. Built with FastAPI and deployable on Vercel.

## Features

- **Process intent** — POST an intent (e.g. “1000 units of product X to US”); get validated suppliers, routes, negotiation, and an AI summary.
- **Agent registry** — Discover agents by role, materials, jurisdiction.
- **World & schemas** — Load or generate world data; expose MCP schemas for frontend.
- **SSE events** — Real-time phase and graph updates during processing.
- **Instructions** — Apply follow-up instructions to an existing simulation by `trace_id`.

## Requirements

- Python 3.10+
- Optional: OpenAI or OpenRouter API key for LLM-powered routes, negotiation, and summaries.

## Setup

```bash
# Clone and enter project
cd agentic-nodes-backend

# Create virtualenv and install
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# Copy env and set your keys
copy .env.example .env
# Edit .env: LLM_PROVIDER, OPENAI_API_KEY or OPENROUTER_API_KEY, etc.
```

## Run locally

```bash
uvicorn main:app --reload --port 8000
```

- API: http://localhost:8000  
- Docs: http://localhost:8000/docs  
- Root (for uptime checks): http://localhost:8000/ → 200 OK for GET/HEAD  

## Environment

| Variable | Description |
|----------|-------------|
| `PORT` | Server port (default 8000). |
| `DEBUG` | Enable debug logging. |
| `LLM_PROVIDER` | `openai` or `openrouter`. |
| `OPENAI_API_KEY` | Required if `LLM_PROVIDER=openai`. |
| `OPENAI_MODEL` | e.g. `gpt-4o-mini`. |
| `OPENROUTER_API_KEY` | Required if `LLM_PROVIDER=openrouter`. |
| `OPENROUTER_MODEL` | e.g. `openai/gpt-4o-mini`. |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins (defaults include `https://its-camilo.github.io` and localhost). |

## API overview

| Method | Path | Description |
|--------|------|-------------|
| GET/HEAD | `/` | Uptime/monitoring; returns 200. |
| GET/HEAD | `/head` | Same as `/`; alternate for checks. |
| GET | `/health` | Health + LLM/world/simulation stats. |
| GET | `/schemas` | MCP schemas. |
| GET | `/registry` | List all agents. |
| POST | `/registry/discover` | Discover agents by query. |
| GET | `/events` | SSE stream (phases, graph updates). |
| POST | `/process-intent` | Main flow: intent → report + summary. |
| POST | `/process-intent/{trace_id}/instructions` | Apply instructions to a run. |
| POST | `/world/generate` | Generate world from optional prompt. |

For uptime monitoring (e.g. UptimeRobot), use the **root URL** with GET or HEAD; the server returns **200 OK** with body `OK`.

## Deploy (Vercel)

1. Connect the repo to Vercel.
2. Set environment variables in the Vercel project (same as `.env`). You do **not** need to set `ALLOWED_ORIGINS` if your frontend is at `https://its-camilo.github.io` (it is included by default).
3. Ensure the project uses the correct build command and output for a Python/FastAPI app (e.g. Vercel’s Python runtime or a serverless function that runs `main:app`).

After deploy, the root `https://your-project.vercel.app/` will respond 200 to GET/HEAD so monitors report success. CORS is configured (in both `vercel.json` and FastAPI) so the GitHub Pages frontend at `https://its-camilo.github.io` can call `/events` and `/process-intent`. If you use **Deployment Protection** (Vercel Auth, Password, etc.), add an **OPTIONS Allowlist** for `/` (or the paths you use) so preflight requests are not blocked.

## License

See repository license file.
