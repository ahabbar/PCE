# PCE — Day 3 Build Summary
**Date:** May 14, 2026
**Goal:** Score Update Engine + Agents 5, 6, 7 + Complete Patient Journey

---

## What Was Built

### 1. Score Update Engine (`src/orchestrator/score_engine.py`)

The most important architectural piece in Day 3. Fires every time a lab result arrives for a waiting patient.

**`LabResult` dataclass:**
- `patient_id`, `test_name`, `result_value`, `result_time`, `is_critical`
- Critical values (Troponin positive, Creatinine elevated, K+ >6.5, Lactate >4.0) trigger immediate re-score

**`ScoreUpdateEvent` dataclass:**
- Captures before/after scores, delta, threshold crossing, queue reorder flag, latency

**`update_score_on_result()` flow:**
1. Skip if patient status is `"seen"` or `"discharged"`
2. Append lab result to `additional_context` (preserves prior context)
3. Re-run Agent 1 (`run_triage_score`) with enriched intake
4. Calculate delta vs stored score
5. Update queue only if `|delta| > 8` OR `is_critical` OR threshold crossed
6. Log both score update and threshold crossing to `audit_log`
7. Return `ScoreUpdateEvent`

**`reconstruct_intake_from_record(record)`** — rebuilds `IntakeForm` from DB dict for re-scoring without requiring original object.

---

### 2. Agent 5 — Doctor Load Balancer (`src/agents/load_balancer.py`)

Pure logic — no LLM. Deterministic scheduling.

**Service time standards (PCE spec):**
| ESI | PCE time | Traditional |
|---|---|---|
| 2 | 22 min | 40 min |
| 3 | 15 min | 28 min |
| 4 | 10 min | 18 min |
| 5 | 7 min | 12 min |

**Seniority matching with escalation:**
- ESI-2 → Consultant first, escalates to Senior Registrar → Registrar → SHO if needed
- ESI-3/4/5 → SHO/Registrar preferred

**Selection algorithm:** Among eligible available doctors, picks the one with lowest `total_remaining_min` (least loaded by time, not count).

**Default pool:** Dr. Rahman (Consultant), Dr. Chen (Senior Registrar), Dr. Patel (Registrar), Dr. Okafor (SHO), Dr. Al-Sayd (SHO). Each with `max_cases=2`.

**`get_load_report()`** — sorted by `load_pct` descending, used by dashboard Doctor Load panel.

---

### 3. Agent 6 — Disposition Forecaster (`src/agents/disposition.py`)

Predicts patient's final destination before the doctor arrives. Eliminates boarding delay.

**Model:** Gemini 2.5 Flash | **Temperature:** 0.3 | **Trigger:** After lab results available

**Four destinations:** `discharge`, `admit`, `icu`, `transfer`

**Calibration (built into system prompt):**
- Most internal medicine ED: ~65% discharge, 30% admit, 4% ICU, 1% transfer
- CKD + AKI → strongly admit
- Young + stable + minor infection → discharge
- Haemodynamic instability → ICU

**Probability normalization:** If LLM output doesn't sum to 100, scales proportionally then fixes rounding on the largest component.

**Bed reservation:** If `admit_pct > 60` → `bed_reservation_sent=True` + logged to `audit_log`.

Three calibrated few-shot examples embedded in system prompt (NSTEMI shock → ICU, UTI+AKI → admit, minor UTI → discharge).

---

### 4. Agent 7 — Exit Coordinator (`src/agents/exit_coord.py`)

Generates the complete exit package when doctor confirms disposition. Doctor approves with one tap.

**Model:** Gemini 2.5 Flash | **Temperature:** 0.3

**Four system prompts — one per disposition path:**

| Path | Output |
|---|---|
| `discharge` | Patient instructions (plain language) + GP letter (3-4 sentences) + prescription notes |
| `admit` | Structured ward handover: summary, background, findings, treatment given, outstanding results, monitoring |
| `icu` | SBAR format critical care handover: haemodynamics, interventions, immediate ICU needs |
| `transfer` | Referral letter + transport requirements + receiving team handover |

All use loose intermediate parse model (`_ExitParse`) before building `ExitPlan`.

---

### 5. New Patient Models (`src/core/patient.py`)

**`DispositionPrediction`:**
- `discharge_pct`, `admit_pct`, `icu_pct`, `transfer_pct` (floats, sum to 100)
- `predicted_destination` (Literal)
- `reasoning` (max 300 chars)
- `bed_reservation_sent` (bool)

**`ExitPlan`:**
- `disposition`, `instructions`, `follow_up`, `handover_note`
- `transport_needed`, `gp_letter_draft`, `prescription_notes`

---

### 6. Database Updates (`src/database/`)

**`models.py`** — two new columns in `patients` table:
- `disposition_prediction TEXT` — JSON string of `DispositionPrediction`
- `bed_reservation_sent INTEGER DEFAULT 0`

**`db.py`** — new function:
- `update_patient_status(patient_id, status)` — updates patient status in DB (used on discharge/seen)

---

### 7. FastAPI — New Endpoints (`src/api/server.py`)

| Method | Path | Description |
|---|---|---|
| `POST` | `/result` | Inject lab result → Score Update Engine → queue re-sort |
| `POST` | `/assign/{patient_id}` | Agent 5 assigns doctor, updates queue status |
| `GET` | `/doctors` | Doctor load report (sorted by load_pct) |
| `POST` | `/exit/{patient_id}` | Agent 7 generates exit plan, discharges from queue |
| `POST` | `/queue/{patient_id}/discharge` | Quick discharge — no LLM, instant queue removal |

---

### 8. Dashboard — Day 3 Updates (`dashboard/app.py`)

**Live Queue tab — three additions:**

**Doctor Load Panel:**
- Fetches `GET /doctors` on every render
- Shows each doctor: name, role, progress bar (`load_pct`), active cases, minutes remaining
- Graceful offline handling

**Lab Result Injection:**
- Form: select patient (by first 8 chars of ID), test name, result value, critical flag
- POSTs to `/result` → shows score delta + queue reorder warning
- Triggers `st.rerun()` so queue updates immediately

**Patient Actions Expander (per card):**

*Left column — Quick Discharge (instant):*
- Select outcome (discharge/admit/ICU/transfer)
- Enter diagnosis
- Click **Discharge / Complete** → patient removed from queue immediately, no LLM

*Right column — Generate Exit Plan (AI, 15–20s):*
- Agent 7 writes full documentation
- Discharge instructions or handover note displayed inline
- Patient also discharged from queue after generation

---

### 9. Full Journey Demo (`scripts/full_journey_demo.py`)

Demonstrates complete patient journey through all 7 agents via API calls. Prints `T+MM:SS` timeline.

**Simulated journey (34F UTI+CKD):**
- `T+00:00` ARRIVAL → `T+00:14` AGENT 1+2+3 triage
- `T+00:25` UA POCT result → score update
- `T+00:30` CBC result → score update + doctor assigned (Agent 5)
- `T+00:40` Creatinine 1.9 (AKI) → score crosses threshold → RED zone alert
- `T+01:17` Doctor confirms → Agent 7 generates admission handover

**Usage:**
```cmd
venv\Scripts\python scripts\full_journey_demo.py
venv\Scripts\python scripts\full_journey_demo.py --slow   :: live demo mode (0.5s pauses)
```

---

### 10. Git Repository

Initialised `git init` at `C:\PCE`. Two commits:
- `1b8bdea` — Day 1+2 complete (55 files, 6,237 lines)
- `dd3aff6` — Day 3: Score Engine + Agents 5+6+7 + Full Journey demo

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
| `test_score_engine.py` | 5 | All passed |
| `test_load_balancer.py` | 7 | All passed |
| `test_disposition.py` | 1 | All passed |
| **Total** | **82** | **All passed** |

*13 real-LLM tests deselected (require API keys, run with `-m real_llm`)*

---

## How to Run

```cmd
cd C:\PCE

:: Terminal 1 — API server (keep open)
venv\Scripts\python -m uvicorn src.api.server:app --port 8000

:: Terminal 2 — Dashboard (keep open)
venv\Scripts\streamlit run dashboard\app.py
:: Open: http://localhost:8501

:: Optional — full journey demo (API must be running)
venv\Scripts\python scripts\full_journey_demo.py --slow

:: All unit tests
venv\Scripts\python -m pytest -v -k "not real_llm"
```

---

## Complete System Architecture (Day 1 + 2 + 3)

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
Queue (in-memory, asyncio.Lock)
      │
      ├── Lab result arrives ──► Score Update Engine ──► Agent 1 re-runs
      │                                │
      │                                └── Threshold crossed? → RED alert
      │
      ├── Agent 5: Load Balancer      (pure logic, seniority matching)
      │
      ├── Agent 6: Disposition Forecaster  (Gemini, admit probability)
      │                └── admit > 60% → bed reservation logged
      │
      └── Agent 7: Exit Coordinator   (Gemini, 4 disposition paths)
                   └── discharge/admit/ICU/transfer documentation
                         → patient removed from queue
```

---

## Complete Patient Journey (Real Use)

1. **Triage Input tab** — enter vitals, symptoms, comorbidities → click **Analyze**
2. Results show: ESI level, risk score, red flag status, key factors, workup orders
3. Click **Add to Queue** → patient appears in Live Queue
4. **Live Queue tab** → patients sorted by risk score (reds first)
5. When lab results arrive → **Inject Lab Result** → score updates automatically
6. Click **Assign Doctor** → load balancer picks best available doctor
7. **Quick Discharge** (instant) or **Generate Exit Plan** (AI documentation)
8. Patient removed from queue — encounter complete

---

## Key Technical Decisions (Day 3)

| Decision | Reason |
|---|---|
| Score update only when `|delta| > 8` or critical | Prevents constant LLM re-runs on minor results; focus re-scoring on clinically significant changes |
| `reconstruct_intake_from_record()` uses Vitals() empty | Vitals not persisted in Day 2 DB schema; re-score uses CC + history + new lab context instead |
| Probability normalization on LLM output | Gemini may return values summing to 99 or 101; scale then adjust largest to guarantee exactly 100 |
| Quick Discharge endpoint (no LLM) | Doctors need to clear stable patients fast; AI exit plan is optional, not mandatory |
| `complete_case()` called on discharge | Frees doctor slot in load balancer so they can accept new patients |
| Four separate system prompts in Agent 7 | Each disposition path requires entirely different documentation structure and tone |

---

## Day 4 — What's Next
- [ ] Queue persistence on server restart (reload from DB on startup)
- [ ] Real-time queue auto-refresh (WebSocket or polling interval)
- [ ] Nurse confirmation workflow (confirm/escalate red flags)
- [ ] Analytics panel: average wait times, ESI distribution, disposition breakdown
- [ ] Batch coordinator visible in dashboard with dispatch button
- [ ] Export triage report (PDF or structured JSON)
- [ ] Multi-shift support (reset queue at shift change, archive records)
