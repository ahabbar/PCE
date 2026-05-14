from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from src.core.patient import AgeGroup, IntakeForm

VITAL_SIGN_THRESHOLDS: dict[AgeGroup, dict] = {
    AgeGroup.NEONATE:    {"HR_max": 190, "RR_max": 60},
    AgeGroup.INFANT:     {"HR_max": 180, "RR_max": 55},
    AgeGroup.TODDLER:    {"HR_max": 140, "RR_max": 40},
    AgeGroup.PRESCHOOL:  {"HR_max": 120, "RR_max": 35},
    AgeGroup.CHILD:      {"HR_max": 120, "RR_max": 30},
    AgeGroup.ADOLESCENT: {"HR_max": 100, "RR_max": 20},
    AgeGroup.ADULT:      {"HR_max": 100, "RR_max": 20},
}

SPO2_HIGH_RISK = 92.0

ESI_BY_RESOURCE_COUNT = {0: 5, 1: 4}

PCE_SCOPE_MAP = {
    1: "excluded_resus",
    2: "pce_priority",
    3: "pce_core",
    4: "pce_efficient",
    5: "pce_lite",
}


class ESIResult(BaseModel):
    esi_level: int
    decision_point_reached: Literal["A", "B", "C", "D"]
    is_immediate: bool = False
    high_risk_vital_signs: bool = False
    vital_sign_flags: list[str] = []
    rationale: str = ""
    estimated_resources: int = 0
    pce_scope: Literal[
        "excluded_resus", "pce_priority", "pce_core", "pce_efficient", "pce_lite"
    ] = "pce_core"


def _esi_to_pce_scope(level: int) -> str:
    return PCE_SCOPE_MAP.get(level, "pce_core")


def check_high_risk_vitals(
    vitals, age_group: AgeGroup, baseline_spo2_low: bool = False
) -> tuple[bool, list[str]]:
    flags: list[str] = []
    thresholds = VITAL_SIGN_THRESHOLDS[age_group]

    if vitals.heart_rate is not None and vitals.heart_rate > thresholds["HR_max"]:
        flags.append(f"HR {vitals.heart_rate} > {thresholds['HR_max']} (age-adjusted)")

    if vitals.respiratory_rate is not None and vitals.respiratory_rate > thresholds["RR_max"]:
        flags.append(f"RR {vitals.respiratory_rate} > {thresholds['RR_max']} (age-adjusted)")

    if (
        vitals.spo2_pct is not None
        and vitals.spo2_pct < SPO2_HIGH_RISK
        and not baseline_spo2_low
    ):
        flags.append(f"SpO2 {vitals.spo2_pct}% < {SPO2_HIGH_RISK}%")

    return bool(flags), flags


def run_esi_algorithm(intake: IntakeForm, estimated_resource_count: int = 0) -> ESIResult:
    obs = intake.nurse_observation
    vitals = intake.vitals
    hx = intake.medical_history
    cc = intake.chief_complaint
    age_group = intake.age_group

    # Decision Point A — ESI-1
    dp_a_triggered = False
    dp_a_reasons: list[str] = []

    if obs.work_of_breathing in ("severe_distress", "apneic"):
        dp_a_triggered = True
        dp_a_reasons.append(f"WOB={obs.work_of_breathing}")
    if obs.skin_assessment in ("cyanotic", "mottled"):
        dp_a_triggered = True
        dp_a_reasons.append(f"Skin={obs.skin_assessment}")
    if obs.pulse_quality == "absent":
        dp_a_triggered = True
        dp_a_reasons.append("Pulse absent")
    if obs.avpu in ("P", "U"):
        dp_a_triggered = True
        dp_a_reasons.append(f"AVPU={obs.avpu}")
    if obs.seizure_active:
        dp_a_triggered = True
        dp_a_reasons.append("Active seizure")
    if obs.general_appearance == "critically_ill":
        dp_a_triggered = True
        dp_a_reasons.append("Critically ill appearance")
    if (
        vitals.spo2_pct is not None
        and vitals.spo2_pct < 90
        and not hx.baseline_spo2_low
    ):
        dp_a_triggered = True
        dp_a_reasons.append(f"SpO2 {vitals.spo2_pct}% < 90%")
    if vitals.blood_glucose is not None and vitals.blood_glucose < 2.8:
        dp_a_triggered = True
        dp_a_reasons.append(f"Severe hypoglycaemia glucose={vitals.blood_glucose}")
    if vitals.systolic_bp is not None and vitals.systolic_bp < 80:
        dp_a_triggered = True
        dp_a_reasons.append(f"SBP {vitals.systolic_bp} < 80")

    if dp_a_triggered:
        return ESIResult(
            esi_level=1,
            decision_point_reached="A",
            is_immediate=True,
            estimated_resources=estimated_resource_count,
            pce_scope="excluded_resus",
            rationale="ESI-1: " + "; ".join(dp_a_reasons),
        )

    # Decision Point B — ESI-2
    dp_b_triggered = False
    dp_b_reasons: list[str] = []

    if obs.altered_mentation:
        dp_b_triggered = True
        dp_b_reasons.append("Altered mentation")
    if obs.acute_distress and obs.work_of_breathing == "moderate_distress":
        dp_b_triggered = True
        dp_b_reasons.append("Acute distress + moderate WOB")
    if cc.pain_score is not None and cc.pain_score >= 7:
        dp_b_triggered = True
        dp_b_reasons.append(f"Pain score {cc.pain_score}/10")
    if cc.category == "altered_mental_status":
        dp_b_triggered = True
        dp_b_reasons.append("CC: altered mental status")
    if hx.immunocompromised and vitals.temperature_c is not None and vitals.temperature_c > 38.5:
        dp_b_triggered = True
        dp_b_reasons.append("Immunocompromised + fever >38.5")
    if age_group == AgeGroup.NEONATE and vitals.temperature_c is not None and vitals.temperature_c > 38.0:
        dp_b_triggered = True
        dp_b_reasons.append("Neonate with fever >38.0")
    if age_group == AgeGroup.INFANT and vitals.temperature_c is not None and vitals.temperature_c > 38.0:
        dp_b_triggered = True
        dp_b_reasons.append("Infant with fever >38.0")

    if dp_b_triggered:
        return ESIResult(
            esi_level=2,
            decision_point_reached="B",
            is_immediate=False,
            estimated_resources=estimated_resource_count,
            pce_scope="pce_priority",
            rationale="ESI-2: " + "; ".join(dp_b_reasons),
        )

    # Decision Point C — resource-based ESI 3/4/5
    capped = min(estimated_resource_count, 2)
    c_level = ESI_BY_RESOURCE_COUNT.get(capped, 3)

    # Decision Point D — vital signs reassessment
    is_high_risk, vital_flags = check_high_risk_vitals(
        vitals, age_group, baseline_spo2_low=hx.baseline_spo2_low
    )

    final_level = c_level
    dp = "C"
    if is_high_risk:
        final_level = min(c_level, 3)
        dp = "D"

    rationale_parts = [f"ESI-{final_level} via Decision Point {dp}"]
    if is_high_risk:
        rationale_parts.append("Vital signs upgraded: " + "; ".join(vital_flags))

    return ESIResult(
        esi_level=final_level,
        decision_point_reached=dp,  # type: ignore[arg-type]
        high_risk_vital_signs=is_high_risk,
        vital_sign_flags=vital_flags,
        estimated_resources=estimated_resource_count,
        pce_scope=_esi_to_pce_scope(final_level),  # type: ignore[arg-type]
        rationale="; ".join(rationale_parts),
    )
