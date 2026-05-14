from __future__ import annotations

import logging
import time
from typing import Optional

from src.agents.llm import LLMClient
from src.core.patient import IntakeForm, RedFlagResult

logger = logging.getLogger("pce.red_flag")

RED_FLAG_SYSTEM_PROMPT = """You are a senior emergency physician performing safety screening.
Your ONLY role: detect life-threatening emergencies not caught by simple rules.
Be conservative — if uncertain, flag as emergency. False positives are safe.
False negatives are dangerous.

CONDITIONS TO DETECT:
ESI-1 (esi1_immediate=true, override_all_agents=true):
  • STEMI or STEMI-equivalent (posterior, new LBBB, Wellens)
  • Massive PE with haemodynamic compromise
  • Aortic dissection (tearing chest/back pain, unequal pulses)
  • Tension pneumothorax
  • Severe anaphylaxis with airway compromise
  • Opioid overdose with respiratory compromise

ESI-2 (esi2_high_risk=true):
  • Early sepsis (infection + any haemodynamic concern)
  • Hypertensive emergency (SBP >180 + headache/visual change/chest pain)
  • Evolving NSTEMI (chest pain + risk factors + diaphoresis)
  • Meningitis (headache + fever + photophobia or neck stiffness)
  • Ectopic pregnancy (female + abdominal pain + hypotension or syncope)
  • Bowel ischaemia (abdominal pain + elderly + AF)

ANTI-BIAS: Women with cardiac symptoms are frequently undertriaged.
Elderly have atypical presentations. Do not under-detect in any demographic.

OUTPUT — ONLY valid JSON, no prose:
{
  "is_emergency": true/false,
  "esi1_immediate": true/false,
  "esi2_high_risk": true/false,
  "flag_type": "condition_name or null",
  "immediate_action": "specific action or null",
  "reasoning": "maximum two sentences"
}"""


def check_red_flags_fast(intake: IntakeForm) -> Optional[RedFlagResult]:
    obs = intake.nurse_observation
    v = intake.vitals
    hx = intake.medical_history
    cc = intake.chief_complaint

    sbp = v.systolic_bp
    spo2 = v.spo2_pct
    hr = v.heart_rate
    temp = v.temperature_c
    glucose = v.blood_glucose

    def make(
        flag_type: str,
        action: str,
        esi1: bool = True,
        esi2: bool = False,
        override: bool = False,
    ) -> RedFlagResult:
        return RedFlagResult(
            is_emergency=True,
            esi1_immediate=esi1,
            esi2_high_risk=esi2,
            flag_type=flag_type,
            immediate_action=action,
            override_all_agents=override,
            layer_triggered="rules",
        )

    # ESI-1 rules
    if obs.pulse_quality == "absent" or obs.avpu == "U":
        return make(
            "cardiac_arrest",
            "CPR / ACLS. Crash cart. Code team now.",
            override=True,
        )

    if obs.work_of_breathing == "apneic" or (
        spo2 is not None and spo2 < 85 and not hx.baseline_spo2_low
    ):
        return make(
            "severe_respiratory_failure",
            "Airway management. Assisted ventilation. Anaesthesia STAT.",
            override=True,
        )

    if (
        sbp is not None
        and sbp < 80
        and obs.skin_assessment in ("mottled", "pale", "cyanotic")
    ):
        return make(
            "haemodynamic_shock",
            "Large bore IV x2. Fluid resuscitation. Type & crossmatch. Resus bay.",
            override=True,
        )

    if glucose is not None and glucose < 2.5 and obs.altered_mentation:
        return make(
            "severe_hypoglycaemia",
            "IV Dextrose 50% immediately. Recheck glucose in 5 minutes.",
        )

    if obs.seizure_active:
        return make(
            "active_seizure",
            "Seizure protocol. Airway. IV access. Benzodiazepine.",
        )

    if obs.avpu in ("P", "U") and obs.general_appearance == "critically_ill":
        return make(
            "unresponsive",
            "Immediate full assessment. Airway + IV. Glucose POCT.",
        )

    if (
        obs.work_of_breathing == "severe_distress"
        and (spo2 is None or spo2 < 92)
        and not hx.baseline_spo2_low
    ):
        return make(
            "severe_respiratory_distress_hypoxia",
            "O2 immediately. NIV if indicated. Senior airway cover.",
        )

    # ESI-2 rules
    if (
        temp is not None
        and temp > 38.5
        and hr is not None
        and hr > 100
        and sbp is not None
        and sbp < 100
    ):
        return make(
            "sepsis_criteria",
            "Sepsis protocol. Lactate + blood cultures. IV access. Antibiotics within 1 hour.",
            esi1=False,
            esi2=True,
        )

    if cc.category == "neurological" and obs.altered_mentation:
        return make(
            "stroke_presentation",
            "Stroke alert. CT head STAT. Neurology. Last known well time.",
            esi1=False,
            esi2=True,
        )

    if (
        cc.category == "chest_pain"
        and obs.skin_assessment in ("diaphoretic", "pale")
        and obs.general_appearance in ("unwell", "distressed")
    ):
        return make(
            "stemi_presentation",
            "ECG within 10 minutes. Cardiology alert if STEMI confirmed.",
            esi1=False,
            esi2=True,
        )

    if obs.altered_mentation and not obs.seizure_active:
        return make(
            "altered_mentation",
            "Glucose POCT immediately. ECG. IV access. Sepsis screen.",
            esi1=False,
            esi2=True,
        )

    if hx.immunocompromised and temp is not None and temp > 38.5:
        return make(
            "immunocompromised_fever",
            "Broad-spectrum antibiotics within 1 hour. Blood cultures first.",
            esi1=False,
            esi2=True,
        )

    return None


async def check_red_flags_llm(intake: IntakeForm, llm: LLMClient) -> RedFlagResult:
    v = intake.vitals
    obs = intake.nurse_observation
    cc = intake.chief_complaint
    hx = intake.medical_history

    user_prompt = (
        f"Patient: {intake.age_years}y {intake.gender}\n"
        f"Chief complaint: {cc.free_text_en}\n"
        f"Category: {cc.category}\n"
        f"Vitals: HR={v.heart_rate}, BP={v.systolic_bp}/{v.diastolic_bp}, "
        f"Temp={v.temperature_c}°C, SpO2={v.spo2_pct}%, GCS={v.gcs}\n"
        f"Nurse: appearance={obs.general_appearance}, WOB={obs.work_of_breathing}, "
        f"skin={obs.skin_assessment}, pulse={obs.pulse_quality}, AVPU={obs.avpu}\n"
        f"Altered mentation: {obs.altered_mentation}\n"
        f"Diagnoses: {hx.known_diagnoses}\n"
        f"Immunocompromised: {hx.immunocompromised}\n\n"
        "Screen for life-threatening emergencies. Apply anti-bias rules."
    )

    from pydantic import BaseModel as _BM

    class _RFParse(_BM):
        is_emergency: bool = False
        esi1_immediate: bool = False
        esi2_high_risk: bool = False
        flag_type: Optional[str] = None
        immediate_action: Optional[str] = None
        override_all_agents: bool = False
        reasoning: str = ""

    try:
        result = await llm.call(
            system=RED_FLAG_SYSTEM_PROMPT,
            user=user_prompt,
            response_schema=_RFParse,
            temperature=0.0,
            max_tokens=500,
        )
        result = RedFlagResult(
            is_emergency=result.is_emergency,
            esi1_immediate=result.esi1_immediate,
            esi2_high_risk=result.esi2_high_risk,
            flag_type=result.flag_type,
            immediate_action=result.immediate_action,
            override_all_agents=result.override_all_agents,
            layer_triggered="llm",
            reasoning=result.reasoning,
        )
        return result
    except Exception as exc:
        logger.warning("Red flag LLM failed: %s — returning safe default", exc)
        return RedFlagResult(
            is_emergency=False,
            layer_triggered="llm",
            reasoning="LLM call failed — conservative clear",
        )


async def run_red_flag_guardian(intake: IntakeForm, llm: LLMClient) -> RedFlagResult:
    t0 = time.time()

    fast = check_red_flags_fast(intake)
    if fast is not None:
        latency_ms = int((time.time() - t0) * 1000)
        logger.info(
            "Red flag RULES | patient=%s emergency=%s flag=%s latency=%dms",
            intake.patient_id[:8],
            fast.is_emergency,
            fast.flag_type,
            latency_ms,
        )
        return fast

    result = await check_red_flags_llm(intake, llm)
    latency_ms = int((time.time() - t0) * 1000)
    logger.info(
        "Red flag LLM | patient=%s emergency=%s flag=%s latency=%dms",
        intake.patient_id[:8],
        result.is_emergency,
        result.flag_type,
        latency_ms,
    )
    return result
