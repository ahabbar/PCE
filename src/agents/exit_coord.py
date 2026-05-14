from __future__ import annotations
import logging, time
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from pydantic import BaseModel
from src.agents.llm import LLMClient
from src.core.patient import IntakeForm, TriageScoreResult, DispositionPrediction, ExitPlan

logger = logging.getLogger("pce.exit_coord")


class DispositionType(str, Enum):
    DISCHARGE = "discharge"
    ADMIT     = "admit"
    ICU       = "icu"
    TRANSFER  = "transfer"


@dataclass
class ExitInput:
    intake: IntakeForm
    triage_score: TriageScoreResult
    disposition: DispositionType
    confirmed_diagnosis: str
    results_summary: list[str]
    assigned_doctor: str
    disposition_prediction: Optional[DispositionPrediction] = None


DISCHARGE_SYSTEM = """You are generating a discharge package for an emergency department patient.
Produce three documents:
1. PATIENT INSTRUCTIONS: Clear, simple language. Include: what the diagnosis is, medications
   to take, return precautions (EXACTLY which symptoms should bring them back urgently), follow-up.
2. GP LETTER: Brief (3-4 sentences). Presentation, key findings, diagnosis, treatment given, follow-up.
3. PRESCRIPTION NOTES: List what medications are likely needed (doctor confirms).
Return JSON only:
{"instructions": "...", "gp_letter_draft": "...", "prescription_notes": "...",
 "follow_up": "...", "transport_needed": false}"""

ADMIT_SYSTEM = """You are generating an admission handover note for an emergency department patient.
Produce a structured handover for the receiving ward team. Include:
- One-line summary (age, sex, presenting complaint, diagnosis)
- Relevant background (comorbidities, medications)
- Key investigation findings, treatment given in ED, outstanding results pending
- Specific concerns for the receiving team, suggested monitoring parameters
Return JSON only:
{"handover_note": "...", "transport_needed": false,
 "instructions": null, "follow_up": null, "gp_letter_draft": null,
 "prescription_notes": "Medications continued: ..."}"""

ICU_SYSTEM = """You are generating an ICU transfer note for a critically ill patient.
Produce a concise critical care handover in SBAR format (Situation, Background, Assessment,
Recommendation). Include haemodynamic status, interventions initiated, immediate ICU requirements.
Return JSON only:
{"handover_note": "...", "transport_needed": false,
 "instructions": null, "follow_up": null, "gp_letter_draft": null, "prescription_notes": null}"""

TRANSFER_SYSTEM = """You are generating a transfer documentation package for a patient being
transferred to a tertiary centre. Include referral letter, transport requirements, receiving handover.
Return JSON only:
{"handover_note": "...", "transport_needed": true,
 "instructions": null, "follow_up": "Receiving centre to arrange",
 "gp_letter_draft": null, "prescription_notes": null}"""

_SYSTEM_MAP = {
    DispositionType.DISCHARGE: DISCHARGE_SYSTEM,
    DispositionType.ADMIT: ADMIT_SYSTEM,
    DispositionType.ICU: ICU_SYSTEM,
    DispositionType.TRANSFER: TRANSFER_SYSTEM,
}


async def run_exit_coordinator(inp: ExitInput, llm: LLMClient) -> ExitPlan:
    system = _SYSTEM_MAP[inp.disposition]
    v = inp.intake.vitals
    hx = inp.intake.medical_history

    user_prompt = (
        f"Patient: {inp.intake.age_years:.0f}y {inp.intake.gender}\n"
        f"Confirmed diagnosis: {inp.confirmed_diagnosis}\n"
        f"Disposition: {inp.disposition.value.upper()}\n"
        f"Assigned doctor: {inp.assigned_doctor}\n"
        f"Risk score: {inp.triage_score.risk_score:.0f}% | Key factors: {inp.triage_score.key_factors}\n"
        f"Vitals: HR={v.heart_rate}, BP={v.systolic_bp}/{v.diastolic_bp}, "
        f"Temp={v.temperature_c}°C, SpO2={v.spo2_pct}%\n"
        f"Comorbidities: {hx.known_diagnoses}\n"
        f"Key results: {inp.results_summary}\n\n"
        "Generate the complete exit package."
    )

    class _ExitParse(BaseModel):
        instructions: Optional[str] = None
        gp_letter_draft: Optional[str] = None
        prescription_notes: Optional[str] = None
        follow_up: Optional[str] = None
        handover_note: Optional[str] = None
        transport_needed: bool = False

    t0 = time.monotonic()
    try:
        raw = await llm.call(
            system=system,
            user=user_prompt,
            response_schema=_ExitParse,
            temperature=0.3,
            max_tokens=1000,
        )
    except Exception as exc:
        logger.warning("Exit coordinator LLM failed: %s", exc)
        raw = _ExitParse()
    latency_ms = (time.monotonic() - t0) * 1000

    plan = ExitPlan(
        disposition=inp.disposition.value,
        instructions=raw.instructions,
        follow_up=raw.follow_up,
        handover_note=raw.handover_note,
        transport_needed=raw.transport_needed,
        gp_letter_draft=raw.gp_letter_draft,
        prescription_notes=raw.prescription_notes,
    )

    try:
        from src.database.db import log_agent_action
        await log_agent_action(
            patient_id=inp.intake.patient_id, agent_id=7,
            action="exit_plan_generated",
            inputs_summary=f"disposition={inp.disposition.value}, diagnosis={inp.confirmed_diagnosis[:50]}",
            outputs_summary=f"plan generated for {inp.disposition.value}",
            latency_ms=latency_ms, model_used="gemini-2.5-flash",
        )
    except Exception:
        pass

    return plan
