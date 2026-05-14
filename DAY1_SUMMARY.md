# PCE — Day 1 Build Summary
**Date:** May 14, 2026  
**Goal:** Agents 1 + 2 running in parallel with live Gemini + Claude APIs

---

## What Was Built

### 1. Project Scaffolding
- Full directory structure: `src/core/`, `src/agents/`, `src/orchestrator/`, `src/database/`, `src/api/`, `dashboard/`, `tests/`
- `requirements.txt`, `.env.example`, `.gitignore`, `LICENSE` (MIT 2026), `pyproject.toml`
- Python venv at `C:\PCE\venv\` — Python 3.14

### 2. Protocol Whitelist (`protocols/whitelist.yaml`)
6 clinical protocols with English + Arabic keywords, auto-orders, conditional orders, and doctor-only restrictions:

| Protocol | Auto-orders |
|---|---|
| `fever_infectious` | 4 |
| `uti_pyelonephritis` | 5 |
| `chest_pain` | 5 |
| `dyspnea` | 5 |
| `abdominal_pain` | 6 |
| `altered_consciousness` | 6 |

### 3. LLM Client (`src/agents/llm.py`)
- **GeminiProvider** — uses `google.genai.Client` (new SDK, `gemini-2.5-flash` default)
- **ClaudeProvider** — uses `anthropic` SDK with tool-use for structured output (`claude-sonnet-4-5` default)
- `LLMClient.call()` — retries once on failure, returns `safe_default` after 2 failures, never raises
- Singleton pattern via `get_llm_client()`
- Provider selected from `LLM_PROVIDER` env var (`gemini` default)

### 4. Pydantic Schemas (`src/core/patient.py`)
12 models based on ESI-5 ENA 2023 handbook:
`AgeGroup`, `ArrivalMode`, `Vitals`, `ChiefComplaint`, `MedicalHistory`, `NurseObservation`, `IntakeForm`, `TriageScoreResult`, `RedFlagResult`, `OrderItem`, `WorkupPlan`, `PatientRecord`

Key features:
- `IntakeForm` auto-derives `AgeGroup` from `age_years` via model validator
- All optional vitals default to `None`
- `risk_score` validated 0–100, `key_factors` 1–6 items, `reasoning` max 200 chars

### 5. Whitelist Loader (`src/core/whitelist.py`)
- `ProtocolWhitelist` — loads YAML with UTF-8 encoding (handles Arabic)
- `match()` — case-insensitive keyword search with word-boundary regex for short keywords (prevents "MI" matching "migraine")
- `is_allowed()` — safety guard: returns `True` only if test appears in `auto_order` or `conditional` lists, `False` if in `requires_doctor`
- `from_env()` classmethod reads `PCE_WHITELIST_PATH`

### 6. ESI Algorithm (`src/core/esi_algorithm.py`)
Fully deterministic — no LLM, no randomness. Implements Decision Points A→D from ESI-5 ENA 2023:

| Decision Point | Outcome | Trigger |
|---|---|---|
| A | ESI-1 | Apneic, cyanotic, absent pulse, AVPU P/U, active seizure, SpO2 <90%, glucose <2.8, SBP <80 |
| B | ESI-2 | Altered mentation, pain ≥7, immunocompromised+fever, neonate/infant fever |
| C | ESI 3/4/5 | Resource count (0→5, 1→4, 2+→3) |
| D | Upgrade to ESI-3 | Age-adjusted HR/RR thresholds, SpO2 <92% |

Age-adjusted vital sign thresholds for 7 age groups (neonate through adult).

### 7. Agent 1 — Risk Score Triage (`src/agents/triage_score.py`)
- Produces a continuous **0–100 risk score** using Gemini (temperature 0.15)
- Uses a loose intermediate parse model to avoid strict validator rejection of LLM output
- Dynamic threshold based on ED load: ≤5 patients → 90%, ≤15 → 85%, ≤25 → 80%, >25 → 75%
- `is_red = risk_score >= threshold` — set by orchestrator, never by LLM
- `sort_queue()` — reds first, then by score + waiting bonus (max +15 points after 50 min)
- Calibrated example: 34F UTI+CKD → score 62–70%, not red

### 8. Agent 2 — Red Flag Guardian (`src/agents/red_flag.py`)
Two-layer safety system running **in parallel** with Agent 1 (always via `asyncio.gather`):

**Layer 1 — Fast Rules (<100ms, no LLM):**
- 7 ESI-1 rules: cardiac arrest, severe respiratory failure, haemodynamic shock, severe hypoglycaemia, active seizure, unresponsive, severe respiratory distress+hypoxia
- 5 ESI-2 rules: sepsis criteria, stroke, STEMI presentation, altered mentation, immunocompromised fever

**Layer 2 — LLM Screen (Gemini, temperature 0.0):**
- Runs only if Layer 1 is clear
- Detects: STEMI, massive PE, aortic dissection, tension pneumothorax, anaphylaxis, opioid OD, hypertensive emergency, meningitis, ectopic pregnancy, bowel ischaemia
- Anti-bias instructions for women, elderly, atypical presentations

### 9. Streamlit Dashboard (`dashboard/app.py`)
Full English UI with:
- **Primary Chief Complaint** — single select (drives protocol matching + HIS auto-load)
- **Associated Symptoms** — multi-select (24 options: diaphoresis, syncope, haemoptysis, leg swelling, etc.)
- **Known Comorbidities** — multi-select (24 options: auto-detects immunocompromised, anticoagulated flags)
- **Vitals** — HR, Temp, BP Sys/Dia, SpO2, GCS
- **Nurse Observation** — appearance, work of breathing, skin, clinical flags (multi-select)
- Results: ESI badge, risk score (color-coded), progress bar, key factors, Red Flag Guardian alert
- Timing shown: both agents run in parallel, typically 5–15s

---

## Test Results

| Suite | Tests | Result |
|---|---|---|
| `test_patient.py` | 7 | All passed |
| `test_whitelist.py` | 12 | All passed |
| `test_esi_algorithm.py` | 8 | All passed |
| `test_llm.py` | 2 unit + 2 real LLM | All passed |
| `test_triage_score.py` | 8 unit + 2 real LLM | All passed |
| `test_red_flag.py` | 7 unit + 1 real LLM | All passed |
| **Total** | **49** | **All passed** |

---

## Integration Test Results

### UTI + CKD Calibration Case (34F, HR 98, Temp 38.6, CKD Stage 2)
```
Protocol matched : uti_pyelonephritis
ESI Level        : 3 (pce_core)
Risk Score       : 66%
Threshold        : 85.0% (load=10)
Is Red           : False
Key Factors      : ['pyelonephritis', 'fever', 'CKD-AKI risk', 'tachycardia']
Red Flag         : False — No emergency
Both agents ran  : IN PARALLEL
```

### Critical Shock Case (58M, BP 82/50, HR 118, SpO2 91%, Diaphoretic)
```
ESI Level        : 3 (upgraded by Decision Point D)
Risk Score       : 93%
Is Red           : True
Red Flag         : True — stemi_presentation (Layer 1 rules, no LLM needed)
```

---

## Key Technical Decisions

| Decision | Reason |
|---|---|
| `google.genai.Client` (new SDK) | `google-generativeai` is deprecated as of 2025 |
| Loose intermediate parse model in agents | Gemini returns >6 key_factors and >200 char reasoning; strict validators rejected valid responses |
| Word-boundary regex for short keywords | "MI" in chest_pain keywords was matching "migraine" |
| UTF-8 encoding on YAML open | Windows defaults to cp1252, breaking Arabic keywords |
| `conftest.py` with `load_dotenv()` | pytest doesn't load `.env` automatically |
| `sys.path.insert` in `app.py` | Streamlit runs from `dashboard/` dir, `src` not on path |
| `max_tokens=2000` for triage calls | Default 1000 caused JSON truncation on long Gemini responses |

---

## How to Run

```cmd
cd C:\PCE

# Run all unit tests
venv\Scripts\python -m pytest -v -k "not real_llm"

# Run live LLM tests (requires .env with API keys)
venv\Scripts\python -m pytest -v -m real_llm

# Launch dashboard
venv\Scripts\streamlit run dashboard\app.py
# Open: http://localhost:8501
```

---

## Day 2 — What's Next
- [ ] Agent 3: Workup Plan Generator (whitelist-constrained investigation orders)
- [ ] Orchestrator: combine all 3 agents, enforce is_red override
- [ ] Database: aiosqlite patient record persistence
- [ ] FastAPI backend: `/triage` endpoint
- [ ] Queue panel in dashboard: sort_queue() visualization
- [ ] Batch mode: process multiple patients simultaneously
