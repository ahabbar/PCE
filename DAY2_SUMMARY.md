# PCE — Day 2 Build Summary
**Date:** May 14, 2026  
**Goal:** Agent 3 + Orchestrator + Database + FastAPI + Queue Dashboard

---

## What Was Built

### 1. Agent 3 — Workup Plan Generator (`src/agents/workup.py`)
Generates whitelist-constrained investigation orders — every order must pass `is_allowed()` before being issued.

**Flow:**
- ESI-1 fast path → returns empty plan immediately (no LLM, patient goes to Resus)
- Protocol match via `ProtocolWhitelist.match()` → auto-orders all pre-approved tests
- Conditional evaluation → natural-language conditions parsed (temp, HR, SBP, SpO2, gender, age, immunocompromised, CKD)
- LLM refinement (ESI ≤ 3 only) → Gemini suggests additional tests, each gated by `is_allowed()`
- `requires_doctor` items are deferred, not ordered

**Condition parser supports:** `OR`, `AND`, numeric comparisons (`temp > 38.5`), patient flags (`immunocompromised`, `ckd`), demographic filters (`gender == female`, `age > 65`)

### 2. Database Layer (`src/database/`)

**`models.py`** — `SCHEMA_SQL` constant with 4 tables:

| Table | Purpose |
|---|---|
| `patients` | Core triage record per visit |
| `investigations` | Individual orders with status + timing |
| `audit_log` | Per-agent action log (agent_id, action, input, output, latency) |
| `nurse_confirmations` | Red flag confirmations by nursing staff |

**`db.py`** — 8 async functions using `aiosqlite` (raw SQL, no ORM):
- `init_db()` — creates all tables
- `save_patient()` — upsert patient triage record
- `save_investigation_orders()` — bulk insert workup orders
- `log_agent_action()` — append audit entry
- `get_patient()`, `get_patient_investigations()`, `get_recent_patients()`
- All accept `db_path=None` → defaults to `PCE_DB_PATH` env var, or `":memory:"` in tests

### 3. Orchestrator Engine (`src/orchestrator/engine.py`)
Single `process_patient()` coroutine that orchestrates all agents:

| Step | Action |
|---|---|
| 1 | ESI algorithm (deterministic, always first) |
| 2 | ESI-1 fast path — no LLM at all, returns immediately |
| 3 | Agents 1 + 2 via `asyncio.gather(return_exceptions=True)` |
| 4 | `is_red` set by orchestrator: `(score ≥ threshold) OR red_flag.is_emergency` |
| 5 | ESI-1 red flag override → skip workup |
| 6 | Agent 3 workup (ESI 2–5 only) |
| 7 | DB persist best-effort — DB failure never crashes triage |

Returns `TriageResult` dataclass. `parallel_confirmed=True` when Agents 1+2 ran via gather.

### 4. Queue Manager (`src/orchestrator/queue.py`)
In-memory `QueueManager` with `asyncio.Lock()` for thread safety:
- `add_patient()` — inserts `PatientRecord`, returns queue position
- `get_sorted_queue()` — calls `sort_queue()` from Agent 1 (reds first, then by score + waiting bonus)
- `update_score()` — re-scores and detects reorder
- `assign_doctor()`, `discharge()` — status transitions

Singleton via `get_queue()`.

### 5. Batch Coordinator (`src/agents/batch.py`)
Groups investigations across patients to reduce lab trips:

- ESI-2 patients marked `timing="immediate"` — bypass batch, go immediately
- Groups by `test_name` for ≥ 2 patients → `CollectionRound`
- Single orders past `max_wait` (8 min) get solo round
- `get_batch_summary()` returns `{test_name: patient_count}` for dashboard display

### 6. FastAPI Backend (`src/api/server.py`)
5 endpoints:

| Method | Path | Description |
|---|---|---|
| `POST` | `/triage` | Full triage — runs all agents, saves to DB |
| `POST` | `/queue/add` | Add pre-computed result directly — no LLM re-run |
| `GET` | `/queue` | Sorted queue with wait times and batch summary |
| `POST` | `/queue/{patient_id}/score` | Update score and reorder |
| `GET` | `/health` | Status check |

CORS open (`allow_origins=["*"]`). Lifespan calls `init_db()` on startup.

**Key design:** `/queue/add` accepts pre-computed scores from the dashboard — avoids running LLM twice and prevents score drift between triage display and queue display.

### 7. Dashboard — Day 2 Update (`dashboard/app.py`)
Two-tab Streamlit UI:

**Tab 1 — Triage Input** (updated):
- All results stored in `st.session_state["result"]` after Analyze
- "Add to Queue" button reads from session state → POST to `/queue/add` (instant, no re-run)
- Spinner shown while posting; button replaced by success banner with queue position
- Prevents double-adding the same patient

**Tab 2 — Live Queue** (new):
- Auto-fetches `GET /queue` on load
- Refresh button triggers `st.rerun()`
- Color-coded patient cards: red ≥ 90%, orange ≥ 75%, amber ≥ 50%, green < 50%
- Metrics: Total Patients, Red Alerts, Queue Depth
- Batch summary panel (tests shared by 2+ patients)
- ESI-1 patients excluded (directed to Resus Bay)

### 8. Multi-Patient Demo (`scripts/multi_patient_demo.py`)
Processes 5 hardcoded patients through the full orchestrator, prints sorted queue and batch summary.

---

## Bug Fixes During Day 2

| Bug | Root Cause | Fix |
|---|---|---|
| Live Queue tab blank on page load | `st.stop()` inside Tab 1 killed entire script before Tab 2 rendered | Replaced `st.stop()` with `if/else` block |
| "Add to Queue" did nothing | Button click triggers re-run where `run_btn=False`, so `intake` was never built, POST never fired | Store all data in `st.session_state`; button reads from state |
| Score mismatch (dashboard 97% vs queue 92%) | `/triage` re-ran LLM agents → different stochastic output | New `/queue/add` endpoint accepts pre-computed scores, no LLM |
| Patient not appearing in queue after POST | `/triage` endpoint called `process_patient()` but never called `queue.add_patient()` | Added `await get_queue().add_patient(result)` in `/triage` |
| Old API server not picking up code changes | `--reload` uvicorn didn't rebind; old process was still running | Kill by PID, restart clean |

---

## Test Results

| Suite | Tests | Result |
|---|---|---|
| `test_patient.py` | 7 | All passed |
| `test_whitelist.py` | 12 | All passed |
| `test_esi_algorithm.py` | 8 | All passed |
| `test_llm.py` | 2 | All passed |
| `test_triage_score.py` | 8 | All passed |
| `test_red_flag.py` | 7 | All passed |
| `test_workup.py` | 7 | All passed |
| `test_database.py` | 4 | All passed |
| `test_batch.py` | 7 | All passed |
| `test_orchestrator.py` | 6 | All passed |
| `test_api.py` | 1 | All passed |
| **Total** | **69** | **All passed** |

---

## How to Run

```cmd
cd C:\PCE

# Terminal 1 — API server (keep open)
venv\Scripts\python -m uvicorn src.api.server:app --port 8000

# Terminal 2 — Dashboard (keep open)
venv\Scripts\streamlit run dashboard\app.py
# Open: http://localhost:8501

# Optional — multi-patient demo
venv\Scripts\python scripts\multi_patient_demo.py
```

---

## Full System Architecture (Day 1 + 2)

```
Patient intake (Streamlit Tab 1)
        │
        ▼
ESI Algorithm (deterministic)
        │
        ├─── ESI-1 ──► Resus Bay (no LLM, no queue)
        │
        ▼
asyncio.gather()
  ├── Agent 1: Risk Score Triage  (Gemini, temp=0.15)
  └── Agent 2: Red Flag Guardian  (rules → Gemini, temp=0.0)
        │
        ▼
Orchestrator sets is_red
        │
        ├─── ESI-1 red flag ──► skip workup
        │
        ▼
Agent 3: Workup Plan  (protocol whitelist → Gemini, whitelist-gated)
        │
        ▼
DB persist (aiosqlite, best-effort)
        │
        ▼
Dashboard: "Add to Queue" ──► POST /queue/add ──► QueueManager (in-memory)
                                                         │
                                                         ▼
                                               Live Queue Tab (Tab 2)
                                               sorted: reds first,
                                               then score + wait bonus
```

---

## Key Technical Decisions (Day 2)

| Decision | Reason |
|---|---|
| `/queue/add` endpoint (pre-computed scores) | Avoids LLM re-run; prevents score drift between triage display and queue |
| `st.session_state` for all triage results | Streamlit re-runs entire script on every button click — state must survive across re-runs |
| `asyncio.Lock()` in QueueManager | FastAPI runs in async event loop; concurrent `/triage` POSTs could corrupt queue dict |
| DB persist as `best-effort` (try/except gather) | DB failure must never crash an active triage — patient safety over data integrity |
| ESI-1 excluded from queue | ESI-1 patients bypass PCE entirely; showing them in queue would confuse triage staff |
| Raw SQL with aiosqlite, no ORM | ORM adds latency and complexity; schema is stable and simple (4 tables) |

---

## Day 3 — What's Next
- [ ] Nurse confirmation workflow (confirm/escalate red flags)
- [ ] Doctor override interface (modify orders, set diagnosis)
- [ ] Real-time queue auto-refresh (WebSocket or polling)
- [ ] Discharge / complete patient from queue
- [ ] Analytics panel: average wait times, ESI distribution, batch efficiency
- [ ] Export triage report (PDF or CSV)
