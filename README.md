# Parallel Care Engine

> "Waiting time in the emergency department is not wasted time.
>  It is undeployed treatment time."

## What it does

NHS emergency departments hit 61% of their 4-hour target in 2024 — a structural failure rooted in sequential, doctor-gated workflows. PCE deploys seven autonomous AI agents during the waiting window: the moment a patient registers, their risk is scored, red flags are screened, a protocol-validated workup is ordered, and results are tracked — all before the doctor arrives. When results come in, the doctor is auto-assigned. When they leave, the next patient is auto-assigned. The doctor sees a patient who is already worked up, not one who has been waiting.

## Quick Start

```
git clone https://github.com/YOUR_REPO/PCE.git
cd PCE
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Add your API key to `.env`:
```
GEMINI_API_KEY=your_key_here
LLM_PROVIDER=gemini
```

```
venv\Scripts\python -m uvicorn src.api.server:app --port 8000
venv\Scripts\streamlit run dashboard\app.py
```

Open http://localhost:8501 — click "Seed Demo Patients" in the sidebar to see a realistic ED state instantly.

## Architecture

```
Patient arrival
      │
      ▼
ESI Algorithm (deterministic, Decision Points A→D)
      │
      ├── ESI-1 ──► Resus Bay (no LLM, no queue)
      │
      ▼
asyncio.gather() ── true parallel
  ├── Agent 1: Risk Score Triage     (Gemini, temp=0.15)
  └── Agent 2: Red Flag Guardian     (rules → Gemini, temp=0.0)
      │
      ▼
Orchestrator sets is_red (score ≥ threshold OR emergency)
      │
      ▼
Agent 3: Workup Plan                 (protocol whitelist → Gemini)
      │
      ▼
Queue (in-memory, asyncio.Lock, persisted to SQLite)
      │
      ├── Lab result arrives ──► Score Update Engine ──► Agent 1 re-runs
      │                                │
      │                                └── 2+ results or critical → Agent 5 auto-assigns
      │
      ├── Agent 5: Load Balancer      (pure logic, seniority matching)
      │                └── doctor freed → cascade assigns next patient
      │
      ├── Agent 6: Disposition Forecaster  (Gemini, admit probability)
      │                └── admit > 60% → bed reservation logged
      │
      └── Agent 7: Exit Coordinator   (Gemini, 4 disposition paths)
                   └── discharge/admit/ICU/transfer documentation
                         → patient removed from queue
```

## The 7 Agents

| # | Name | Type | Scope |
|---|------|------|-------|
| 1 | Risk Score Triage | Gemini 2.5 Flash, temp=0.15 | All ESI 2–5 |
| 2 | Red Flag Guardian | Rules + Gemini, temp=0.0 | All ESI 2–5 |
| 3 | Workup Plan | Protocol whitelist + Gemini | All ESI 2–5 |
| 4 | Batch Coordinator | Pure logic | All ESI 2–5 |
| 5 | Doctor Load Balancer | Pure logic, seniority matching | Fires on 2+ results |
| 6 | Disposition Forecaster | Gemini 2.5 Flash, temp=0.3 | After results available |
| 7 | Exit Coordinator | Gemini 2.5 Flash, temp=0.3 | On doctor confirmation |

## API Endpoints

`POST /triage` · `GET /queue` · `POST /result` · `POST /assign/{id}` · `GET /doctors` · `POST /exit/{id}` · `POST /demo/seed` · `GET /analytics` · `GET /cases/list` · `POST /cases/run/{id}`

## Deploy to Railway

PCE ships with `nixpacks.toml`, `Procfile`, `railway.json`, and `Dockerfile` so it deploys to Railway as two services from one repo.

### 1. Push to GitHub

```
git init
git add -A
git commit -m "Initial PCE commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/pce.git
git push -u origin main
```

### 2. Create the Railway project

1. railway.app → **New Project** → **Deploy from GitHub repo** → select your PCE repo.
2. Railway detects `nixpacks.toml` and builds with Python 3.11.
3. The first service it creates is the **API**. Leave the start command as-is — it picks up `web` from the Procfile.

### 3. Add the dashboard as a second service

1. In the same project, **+ New** → **GitHub repo** → pick PCE again.
2. Open the new service → **Settings** → **Deploy** → **Custom Start Command**:
   ```
   /opt/venv/bin/streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
   ```
3. **Settings** → **Networking** → **Generate Domain** (so users can reach the UI).

### 4. Environment variables

On **both** services:

| Var | Value |
|---|---|
| `LLM_PROVIDER` | `gemini` |
| `GEMINI_API_KEY` | your key |
| `PCE_DB_PATH` | `/data/pce.db` |

On the **dashboard** service only:

| Var | Value |
|---|---|
| `PCE_API_URL` | the public URL of the API service (e.g. `https://pce-api-production.up.railway.app`) |

### 5. Persistent volume (so the SQLite DB survives redeploys)

On the **API** service → **Settings** → **Volumes** → **+ New Volume** → mount path `/data`.

### 6. Custom domain

On the dashboard service → **Settings** → **Networking** → **Custom Domain** → enter your domain (e.g. `pce.yourdomain.com`) → add the CNAME record Railway shows you at your DNS provider.

## License

MIT

## Built at

AI Agent Olympics @ Milan AI Week 2026 | lablab.ai
