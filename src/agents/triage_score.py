from __future__ import annotations

import logging
import time
from typing import Optional

from pydantic import BaseModel
from src.agents.llm import LLMClient
from src.core.esi_algorithm import ESIResult
from src.core.patient import IntakeForm, PatientRecord, TriageScoreResult

logger = logging.getLogger("pce.triage_score")

TRIAGE_SYSTEM_PROMPT = """You are a senior emergency medicine physician performing rapid triage scoring.
Your task: assign a single continuous risk score from 0 to 100.

IMPORTANT: The ESI level has already been determined by a rules-based algorithm.
You are adding clinical nuance — not replacing the ESI level.

SCORING CALIBRATION:
  90-100: ESI-1 equivalent — immediate life threat (very rare)
  75-89:  ESI-2 territory — emergent, high risk for deterioration
  50-74:  ESI-3 zone — urgent, needs workup (most internal medicine patients)
  25-49:  ESI-4 — less urgent, stable
  0-24:   ESI-5 — non-urgent, minimal risk

MANDATORY ADJUSTMENTS (apply these arithmetically):
  If GCS < 15 (newly):              +15 points
  If immunocompromised + fever:      +12 points
  If SpO2 < 92% (not baseline):     +15 points
  If HR > age-adjusted threshold:    +10 to +15 points
  If SBP < 100 (adult):             +15 to +20 points
  If age > 75:                      +8 points (higher deterioration risk)
  If CKD background + renal complaint: +10 points (AKI risk)
  If immunosuppressed (transplant/chemo): +12 points

ANTI-BIAS (ESI-5 Handbook, ENA 2023):
  Do NOT undertriage based on gender, ethnicity, age, or psychiatric history.
  Women with chest pain are frequently undertriaged — do not repeat this error.
  Elderly patients have atypical presentations — higher index of suspicion.
  Pain score self-report must be corroborated by clinical observation.

CALIBRATION EXAMPLES:
  58y male, chest pain + diaphoresis, HR 115, BP 85/55, SpO2 91%
  → risk_score: 93, factors: ["hemodynamic instability","ACS","hypoxia"]
  → reasoning: "Cardiogenic shock presentation — immediate intervention."

  34y female, dysuria + flank pain, HR 98, temp 38.6°C, SpO2 97%, CKD background
  → risk_score: 66, factors: ["pyelonephritis","tachycardia","fever","CKD-AKI risk"]
  → reasoning: "Likely pyelonephritis in CKD patient — urgent BMP for AKI."

  75y male, confusion + weakness, HR 108, temp 38.8°C, BP 102/68, GCS 13
  → risk_score: 82, factors: ["altered GCS","elderly","fever+tachy","borderline BP"]
  → reasoning: "Sepsis with early hemodynamic compromise in elderly — urgent."

  29y female, mild dysuria, HR 72, BP 118/76, temp 36.8°C, SpO2 99%, no comorbidities
  → risk_score: 18, factors: ["uncomplicated UTI","stable vitals","no risk factors"]
  → reasoning: "Low-risk uncomplicated lower UTI."

OUTPUT — respond ONLY with valid JSON, no other text:
{"risk_score": 0-100, "confidence": 0.0-1.0, "key_factors": ["factor1","factor2"], "reasoning": "one sentence"}"""


class _LLMTriageOutput(TriageScoreResult):
    pass


def get_dynamic_threshold(patient_load: int) -> float:
    if patient_load <= 5:
        return 90.0
    elif patient_load <= 15:
        return 85.0
    elif patient_load <= 25:
        return 80.0
    else:
        return 75.0


def sort_queue(patients: list[PatientRecord]) -> list[PatientRecord]:
    now = time.time()

    def waiting_bonus(p: PatientRecord) -> float:
        minutes_waiting = (now - p.arrival_time) / 60
        return min(minutes_waiting * 0.3, 15.0)

    def sort_key(p: PatientRecord):
        score = p.triage_score.risk_score if p.triage_score else 0.0
        is_red = p.triage_score.is_red if p.triage_score else False
        if is_red:
            return (1, score, 0.0)
        return (0, score + waiting_bonus(p), 0.0)

    return sorted(patients, key=sort_key, reverse=True)


async def run_triage_score(
    intake: IntakeForm,
    esi_result: ESIResult,
    llm: LLMClient,
    patient_load: int = 10,
) -> TriageScoreResult:
    v = intake.vitals
    obs = intake.nurse_observation
    hx = intake.medical_history
    cc = intake.chief_complaint

    user_prompt = (
        f"Patient: {intake.age_years}y {intake.gender}\n"
        f"Chief complaint: {cc.free_text_en}\n"
        f"Vitals: HR {v.heart_rate}, RR {v.respiratory_rate}, "
        f"BP {v.systolic_bp}/{v.diastolic_bp}, Temp {v.temperature_c}°C, "
        f"SpO2 {v.spo2_pct}%, GCS {v.gcs}\n"
        f"Blood glucose: {v.blood_glucose}\n"
        f"Nurse observation: appearance={obs.general_appearance}, "
        f"WOB={obs.work_of_breathing}, skin={obs.skin_assessment}, AVPU={obs.avpu}\n"
        f"Altered mentation: {obs.altered_mentation}\n\n"
        f"Rules-based ESI: Level {esi_result.esi_level} "
        f"(Decision Point {esi_result.decision_point_reached})\n"
        f"High-risk vitals flagged: {esi_result.vital_sign_flags}\n\n"
        f"Medical history (from HIS):\n"
        f"  Diagnoses: {hx.known_diagnoses}\n"
        f"  Medications: {hx.current_medications}\n"
        f"  Immunocompromised: {hx.immunocompromised}\n"
        f"  Known CKD (baseline Cr): {hx.baseline_cr_abnormal}\n\n"
        f"Additional context: {intake.additional_context}\n\n"
        f"Assign risk score 0-100. Apply anti-bias reasoning."
    )

    class _LLMParse(BaseModel):
        risk_score: float = 50.0
        confidence: float = 0.5
        key_factors: list[str] = ["unknown"]
        reasoning: str = "LLM output"

    t0 = time.time()
    try:
        raw = await llm.call(
            system=TRIAGE_SYSTEM_PROMPT,
            user=user_prompt,
            response_schema=_LLMParse,
            temperature=0.15,
            max_tokens=2000,
        )
    except Exception:
        raw = _LLMParse()

    threshold = get_dynamic_threshold(patient_load)
    is_red = raw.risk_score >= threshold
    latency_ms = int((time.time() - t0) * 1000)

    result = TriageScoreResult(
        risk_score=raw.risk_score,
        esi_level=esi_result.esi_level,
        confidence=raw.confidence,
        is_red=is_red,
        threshold_used=threshold,
        key_factors=raw.key_factors[:6] if raw.key_factors else ["unknown"],
        reasoning=raw.reasoning[:200] if raw.reasoning else "",
        pce_scope=esi_result.pce_scope,
    )

    logger.info(
        "Triage score | patient=%s risk=%.1f threshold=%.1f is_red=%s esi=%d latency=%dms",
        intake.patient_id[:8],
        result.risk_score,
        result.threshold_used,
        result.is_red,
        result.esi_level,
        latency_ms,
    )

    return result
