from __future__ import annotations
import json, logging, time
from dataclasses import dataclass, field

logger = logging.getLogger("pce.validator")

NORMALISATION_MAP: dict[str, str] = {
    "ua":                   "Urinalysis POCT",
    "urinalysis":           "Urinalysis POCT",
    "urinalysis poct":      "Urinalysis POCT",
    "urine culture":        "Urine culture",
    "cbc":                  "CBC",
    "full blood count":     "CBC",
    "fbc":                  "CBC",
    "bmp":                  "BMP",
    "u&e":                  "BMP",
    "renal profile":        "BMP",
    "crp":                  "CRP",
    "troponin":             "Troponin POCT",
    "troponin poct":        "Troponin POCT",
    "ecg":                  "ECG (immediate)",
    "ecg (immediate)":      "ECG (immediate)",
    "cxr":                  "CXR",
    "chest x-ray":          "CXR",
    "bnp":                  "BNP",
    "blood culture":        "Blood culture",
    "blood gas":            "Blood gas",
    "glucose poct":         "Glucose POCT",
    "blood glucose":        "Glucose POCT",
}


def _normalise(name: str) -> str:
    key = name.strip().lower()
    return NORMALISATION_MAP.get(key, name.strip())


def _normalise_list(tests: list[str]) -> list[str]:
    return [_normalise(t) for t in tests]


@dataclass
class CaseValidationResult:
    case_id:            str
    chief_complaint:    str
    actual_diagnosis:   str
    actual_disposition: str
    actual_workup:      list[str]
    proposed_workup:    list[str]
    proposed_score:     float
    proposed_esi:       int
    matched_tests:      list[str]
    missed_tests:       list[str]
    extra_tests:        list[str]
    agreement_rate:     float
    disposition_match:  bool
    latency_ms:         float


async def validate_case(case: dict) -> CaseValidationResult:
    from src.core.patient import IntakeForm, ChiefComplaint, Vitals, MedicalHistory
    from src.core.esi_algorithm import run_esi_algorithm
    from src.orchestrator.engine import process_patient

    t0 = time.monotonic()
    vitals = case.get("vitals", {})
    comorbidities = [c.strip() for c in case.get("comorbidities", "").split(",") if c.strip() and c.strip().lower() != "none"]

    intake = IntakeForm(
        patient_id=f"val_{case['id']}",
        age_years=float(case.get("age", 50)),
        gender=case.get("gender", "unknown"),
        chief_complaint=ChiefComplaint(
            free_text_en=case["chief_complaint"],
            category="other",
            pain_present="pain" in case["chief_complaint"].lower(),
        ),
        vitals=Vitals(
            heart_rate=float(vitals.get("hr", 80)),
            systolic_bp=float(vitals.get("sbp", 120)),
            temperature_c=float(vitals.get("temp", 37.0)),
            spo2_pct=float(vitals.get("spo2", 98)),
        ),
        medical_history=MedicalHistory(known_diagnoses=comorbidities),
    )

    result = await process_patient(intake, patient_load=10, run_workup=True)
    latency_ms = (time.monotonic() - t0) * 1000

    proposed_raw = [o.test_name for o in result.workup.orders] if result.workup else []
    proposed_norm = _normalise_list(proposed_raw)
    actual_norm = _normalise_list(case.get("actual_workup", []))

    proposed_set = set(proposed_norm)
    actual_set = set(actual_norm)
    matched = list(actual_set & proposed_set)
    missed = list(actual_set - proposed_set)
    extra = list(proposed_set - actual_set)
    agreement = len(matched) / len(actual_set) if actual_set else 1.0

    actual_disp = case.get("actual_disposition", "discharge")
    proposed_disp = "admit" if result.triage_score.risk_score >= 70 else "discharge"
    disposition_match = actual_disp == proposed_disp

    return CaseValidationResult(
        case_id=case["id"],
        chief_complaint=case["chief_complaint"],
        actual_diagnosis=case.get("actual_diagnosis", ""),
        actual_disposition=actual_disp,
        actual_workup=actual_norm,
        proposed_workup=proposed_norm,
        proposed_score=result.triage_score.risk_score,
        proposed_esi=result.triage_score.esi_level,
        matched_tests=matched,
        missed_tests=missed,
        extra_tests=extra,
        agreement_rate=round(agreement, 3),
        disposition_match=disposition_match,
        latency_ms=round(latency_ms, 1),
    )


async def validate_all_cases(json_path: str = "data/cases/demo_cases.json") -> list[CaseValidationResult]:
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return []

    results = []
    for case in data.get("cases", []):
        try:
            r = await validate_case(case)
            results.append(r)
        except Exception as exc:
            logger.warning("Validation failed for %s: %s", case.get("id"), exc)
    return results


def load_case_list(json_path: str = "data/cases/demo_cases.json") -> list[dict]:
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        return [{"id": c["id"], "chief_complaint": c["chief_complaint"]} for c in data.get("cases", [])]
    except FileNotFoundError:
        return []


def load_case_by_id(case_id: str, json_path: str = "data/cases/demo_cases.json") -> dict | None:
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        for c in data.get("cases", []):
            if c["id"] == case_id:
                return c
        return None
    except FileNotFoundError:
        return None
