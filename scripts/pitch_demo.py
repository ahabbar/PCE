"""
Pitch guide for the PCE 3-minute demo at AI Agent Olympics.
Run this alongside the dashboard: python scripts/pitch_demo.py
"""
import sys

CHECKLIST = """
╔══════════════════════════════════════════════════════════╗
║   PCE — PRE-DEMO CHECKLIST                              ║
╚══════════════════════════════════════════════════════════╝

□ API server running:  venv\\Scripts\\uvicorn src.api.server:app --port 8000
□ Dashboard open:      http://localhost:8501
□ Tab 3 (Demo Mode) visible
□ Demo patients seeded — click "Seed Demo Patients" in sidebar
□ Tab 2 (Live Queue) showing 5 patients, chest pain at top (88%)
□ Backup video open in VLC, paused at 0:00
"""

FULL_SCRIPT = """
╔══════════════════════════════════════════════════════════╗
║   PCE — 3-MINUTE PITCH GUIDE                            ║
║   AI Agent Olympics @ Milan AI Week 2026                ║
╠══════════════════════════════════════════════════════════╣
║  PRE-DEMO CHECKLIST (do before stepping on stage)       ║
╚══════════════════════════════════════════════════════════╝

□ API server running:  venv\\Scripts\\uvicorn src.api.server:app --port 8000
□ Dashboard open:      http://localhost:8501
□ Tab 3 (Demo Mode) visible
□ Demo patients seeded — click "Seed Demo Patients" in sidebar
□ Tab 2 (Live Queue) showing 5 patients, chest pain at top (88%)
□ Backup video open in VLC, paused at 0:00

══════ 0:00 — 0:25  THE HOOK ══════════════════════════════

SAY:  "NHS emergency departments hit 61% of their 4-hour target in 2024.
       In January 2025 it fell to 34% — the worst month ever recorded.
       This is not a staffing problem. It is an architecture problem.

       Patients wait before the doctor.
       They wait after the doctor.
       Most of that time — nothing clinical is happening.

       What if that waiting time wasn't empty?"

══════ 0:25 — 1:15  LIVE DEMO ═════════════════════════════

ACTION: Switch to Tab 3 (Demo Mode)
SAY:    "Give me a chief complaint."

[Judge gives a complaint — type it + basic vitals]

ACTION: Click "Analyze — Run All 7 Agents"

WHILE LOADING (5-15s):
       "Agents 1 and 2 are running simultaneously — not sequentially.
        The red flag guardian is checking for emergencies at exactly the
        same time as the risk scorer is calibrating acuity."

WHEN RESULT APPEARS:
POINT TO score gauge:
       "This is a continuous 0–100 risk score — not a fixed ESI category.
        It updates every time a lab result arrives."

POINT TO workup table:
       "Every test you see is validated against a physician-signed protocol
        whitelist. The agent cannot order anything outside this list."

POINT TO parallel confirmation:
       "asyncio.gather — confirmed parallel execution."

ACTION: Click "Add to Queue"
ACTION: Switch to Tab 2 (Live Queue)
SAY:    "The patient enters the queue — sorted by risk score, reds first."

══════ 1:15 — 1:50  REAL CASE ═════════════════════════════

ACTION: Scroll down in Tab 3 to Case Validation section
SAY:    "Here is a real case from our internal medicine ED.
         Chief complaint only — no diagnosis, no results.
         Watch what PCE proposes."

ACTION: Select a case → click "Run PCE on this case"

POINT TO comparison:
       "Green ticks — tests PCE proposed that were actually ordered.
        Agreement rate: [X]%.
        PCE proposed this based solely on the chief complaint,
        protocol whitelist, and patient history.
        No diagnosis. No results. Just pattern."

══════ 1:50 — 2:20  ECONOMICS ═════════════════════════════

ACTION: Switch to Tab 4 (Analytics)
SAY:    "Same resources. Same staff. Different architecture."

POINT TO comparison metrics:
       "43 minutes per patient → 20 minutes.
        Sepsis to antibiotics: 142 minutes → 38 minutes.
        That is not an incremental improvement — that is lives."

POINT TO economic impact:
       "£16.99 saved per patient. Annual impact: £11.7 million.
        Break-even on implementation: 13 days."

══════ 2:20 — 3:00  THE CLOSE ═════════════════════════════

SAY:  "We will pilot this in our hospital's internal medicine emergency
       department within 12 months.

       The protocol whitelist is physician-authored.
       Every order requires nurse confirmation.
       The doctor remains in full control.

       This is not AI replacing emergency physicians.
       This is AI converting 85 minutes of dead waiting time
       into 85 minutes of parallel diagnostic work.

       Tonight is the proof of concept."

══════ IF DEMO FAILS ═══════════════════════════════════════

1. Calmly say: "Let me load the recorded demo"
2. Switch to VLC → play backup video
3. Continue narrating over the video — same script
4. Never apologise. The clinical story is stronger than any demo.

════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    if "--checklist" in sys.argv:
        print(CHECKLIST)
    else:
        print(FULL_SCRIPT)
