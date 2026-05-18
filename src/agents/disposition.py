from __future__ import annotations
import logging, time
from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel
from src.agents.llm import LLMClient
from src.core.patient import IntakeForm, TriageScoreResult, WorkupPlan, DispositionPrediction

logger = logging.getLogger("pce.disposition")

DISPOSITION_SYSTEM = """You are a senior emergency physician predicting a patient's most likely
disposition in an internal medicine emergency department.

Your role is to provide an early probability estimate BEFORE the doctor arrives.
This allows the team to pre-reserve a bed and reduce boarding delays.

Four possible destinations:
- DISCHARGE: patient can go home with oral treatment and follow-up
- ADMIT: needs inpatient care (general ward, step-down, or monitored bed)
- ICU: needs intensive care or high-dependency unit
- TRANSFER: needs a tertiary centre for specialist care not available here

PROBABILITY CALIBRATION:
- Probabilities must sum to exactly 100
- Most internal medicine ED patients: ~65% discharge, 30% admit, 4% ICU, 1% transfer
- Elevated inflammatory markers + haemodynamic instability → push toward admit/ICU
- Young patient + isolated minor infection + stable vitals → push toward discharge
- Known CKD + AKI on results → strongly consider admit

ANTI-BIAS: Do not predict discharge simply because the patient appears calm
or communicative. Base prediction on physiological data, not behaviour.

FEW-SHOT EXAMPLES:
Example A: 58M, chest pain, Troponin elevated, NSTEMI, HR 115, BP 88/55
→ discharge_pct:5, admit_pct:30, icu_pct:62, transfer_pct:3, predicted_destination:"icu"

Example B: 34F, dysuria, UA positive, WBC 16800, CRP 180, Creatinine 1.9 (AKI)
→ discharge_pct:12, admit_pct:82, icu_pct:5, transfer_pct:1, predicted_destination:"admit"

Example C: 29F, mild dysuria, UA: Nitrites+, haemodynamically stable, no comorbidities
→ discharge_pct:92, admit_pct:7, icu_pct:1, transfer_pct:0, predicted_destination:"discharge"

Return JSON only:
{
  "discharge_pct": 0-100,
  "admit_pct": 0-100,
  "icu_pct": 0-100,
  "transfer_pct": 0-100,
  "predicted_destination": "discharge" | "admit" | "icu" | "transfer",
  "reasoning": "2-3 clinical sentences maximum"
}"""


@dataclass
class DispositionInput:
    intake: IntakeForm
    triage_score: TriageScoreResult
    workup: WorkupPlan
    results_so_far: list[str]
    is_red: bool


async def run_disposition_forecast(inp: DispositionInput, llm: LLMClient) -> DispositionPrediction:
    v = inp.intake.vitals
    hx = inp.intake.medical_history
    cc = inp.intake.chief_complaint

    user_prompt = (
        f"Patient: {inp.intake.age_years:.0f}y {inp.intake.gender}\n"
        f"Chief complaint: {cc.free_text_en}\n"
        f"Vitals: HR={v.heart_rate}, BP={v.systolic_bp}/{v.diastolic_bp}, "
        f"Temp={v.temperature_c}°C, SpO2={v.spo2_pct}%\n"
        f"Comorbidities: {hx.known_diagnoses}\n"
        f"Risk score: {inp.triage_score.risk_score:.0f}% | Key factors: {inp.triage_score.key_factors}\n"
        f"Workup ordered: {[o.test_name for o in inp.workup.orders] if inp.workup.orders else 'pending'}\n"
        f"Results so far: {inp.results_so_far if inp.results_so_far else 'none yet'}\n"
        f"Is red: {inp.is_red}\n\n"
        "Predict disposition probabilities."
    )

    class _DispParse(BaseModel):
        discharge_pct: float = 65.0
        admit_pct: float = 30.0
        icu_pct: float = 4.0
        transfer_pct: float = 1.0
        predicted_destination: str = "discharge"
        reasoning: str = ""

    try:
        raw = await llm.call(
            system=DISPOSITION_SYSTEM,
            user=user_prompt,
            response_schema=_DispParse,
            temperature=0.3,
            max_tokens=500,
        )
    except Exception as exc:
        logger.warning("Disposition LLM failed: %s", exc)
        raw = _DispParse()

    # Normalize probabilities to sum to 100
    total = raw.discharge_pct + raw.admit_pct + raw.icu_pct + raw.transfer_pct
    if total <= 0:
        total = 100.0
    scale = 100.0 / total
    d = round(raw.discharge_pct * scale)
    a = round(raw.admit_pct * scale)
    i = round(raw.icu_pct * scale)
    t = round(raw.transfer_pct * scale)
    # Fix rounding error on the largest component
    diff = 100 - (d + a + i + t)
    vals = {"discharge_pct": d, "admit_pct": a, "icu_pct": i, "transfer_pct": t}
    largest = max(vals, key=vals.get)
    vals[largest] += diff

    # Clamp predicted_destination to valid literal
    valid_dests = {"discharge", "admit", "icu", "transfer"}
    dest = raw.predicted_destination.lower().strip()
    if dest not in valid_dests:
        dest = max(vals, key=vals.get).replace("_pct", "")

    bed_reservation_sent = vals["admit_pct"] > 60

    if bed_reservation_sent:
        try:
            from src.database.db import log_agent_action
            await log_agent_action(
                patient_id=inp.intake.patient_id, agent_id=5,
                action="bed_reservation_sent",
                inputs_summary=f"admit_pct={vals['admit_pct']}%",
                outputs_summary="Provisional bed requested",
                latency_ms=0, model_used="gemini-2.5-flash",
            )
        except Exception:
            pass

    return DispositionPrediction(
        discharge_pct=float(vals["discharge_pct"]),
        admit_pct=float(vals["admit_pct"]),
        icu_pct=float(vals["icu_pct"]),
        transfer_pct=float(vals["transfer_pct"]),
        predicted_destination=dest,
        reasoning=raw.reasoning[:300] if raw.reasoning else "",
        bed_reservation_sent=bed_reservation_sent,
    )
