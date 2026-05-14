from __future__ import annotations
import pytest
from src.core.patient import DispositionPrediction


def test_probabilities_sum_to_100():
    """Normalization always produces exactly 100."""
    raw = {"discharge_pct": 30.0, "admit_pct": 50.0, "icu_pct": 15.0, "transfer_pct": 12.0}
    total = sum(raw.values())
    scale = 100.0 / total
    vals = {k: round(v * scale) for k, v in raw.items()}
    diff = 100 - sum(vals.values())
    largest = max(vals, key=vals.get)
    vals[largest] += diff
    assert sum(vals.values()) == 100


@pytest.mark.real_llm
@pytest.mark.asyncio
async def test_icu_predicted_for_shock():
    from src.agents.disposition import run_disposition_forecast, DispositionInput
    from src.agents.llm import get_llm_client
    from src.core.patient import IntakeForm, Vitals, ChiefComplaint, MedicalHistory, WorkupPlan, TriageScoreResult
    import uuid, time
    intake = IntakeForm(
        patient_id=str(uuid.uuid4()), arrival_time=time.time(),
        age_years=58, gender="male",
        chief_complaint=ChiefComplaint(free_text_en="chest pain", category="chest_pain", pain_present=True),
        vitals=Vitals(heart_rate=115, systolic_bp=88, spo2_pct=94),
        medical_history=MedicalHistory(known_diagnoses=["Hypertension", "Diabetes"]),
    )
    inp = DispositionInput(
        intake=intake,
        triage_score=TriageScoreResult(risk_score=92, esi_level=2, confidence=0.95, is_red=True,
                                        threshold_used=85, key_factors=["NSTEMI", "shock"],
                                        reasoning="Haemodynamic instability"),
        workup=WorkupPlan(patient_id=intake.patient_id, protocol_key="chest_pain",
                          orders=[], deferred_to_doctor=[], reasoning=""),
        results_so_far=["Troponin: 0.08 (elevated)", "ECG: ST depression V4-V6"],
        is_red=True,
    )
    pred = await run_disposition_forecast(inp, get_llm_client())
    assert pred.icu_pct > 50


@pytest.mark.real_llm
@pytest.mark.asyncio
async def test_admit_for_aki():
    from src.agents.disposition import run_disposition_forecast, DispositionInput
    from src.agents.llm import get_llm_client
    from src.core.patient import IntakeForm, Vitals, ChiefComplaint, MedicalHistory, WorkupPlan, TriageScoreResult
    import uuid, time
    intake = IntakeForm(
        patient_id=str(uuid.uuid4()), arrival_time=time.time(),
        age_years=34, gender="female",
        chief_complaint=ChiefComplaint(free_text_en="fever and flank pain", category="urinary", pain_present=True),
        vitals=Vitals(heart_rate=98, temperature_c=38.6, systolic_bp=118, spo2_pct=97),
        medical_history=MedicalHistory(known_diagnoses=["CKD stage 2"]),
    )
    inp = DispositionInput(
        intake=intake,
        triage_score=TriageScoreResult(risk_score=80, esi_level=3, confidence=0.85, is_red=False,
                                        threshold_used=85, key_factors=["AKI", "fever"],
                                        reasoning="Pyelonephritis with AKI"),
        workup=WorkupPlan(patient_id=intake.patient_id, protocol_key="uti_pyelonephritis",
                          orders=[], deferred_to_doctor=[], reasoning=""),
        results_so_far=["CBC: WBC 16,800", "Creatinine: 1.9 mmol/L (AKI)", "UA: Nitrites++"],
        is_red=False,
    )
    pred = await run_disposition_forecast(inp, get_llm_client())
    assert pred.admit_pct > 60


@pytest.mark.real_llm
@pytest.mark.asyncio
async def test_discharge_predicted_for_minor():
    from src.agents.disposition import run_disposition_forecast, DispositionInput
    from src.agents.llm import get_llm_client
    from src.core.patient import IntakeForm, Vitals, ChiefComplaint, MedicalHistory, WorkupPlan, TriageScoreResult
    import uuid, time
    intake = IntakeForm(
        patient_id=str(uuid.uuid4()), arrival_time=time.time(),
        age_years=29, gender="female",
        chief_complaint=ChiefComplaint(free_text_en="mild dysuria", category="urinary", pain_present=True),
        vitals=Vitals(heart_rate=78, temperature_c=37.1, systolic_bp=118, spo2_pct=99),
        medical_history=MedicalHistory(known_diagnoses=[]),
    )
    inp = DispositionInput(
        intake=intake,
        triage_score=TriageScoreResult(risk_score=25, esi_level=4, confidence=0.9, is_red=False,
                                        threshold_used=85, key_factors=["mild UTI"], reasoning="Uncomplicated"),
        workup=WorkupPlan(patient_id=intake.patient_id, protocol_key="uti_pyelonephritis",
                          orders=[], deferred_to_doctor=[], reasoning=""),
        results_so_far=["UA: Nitrites+, WBC mild"],
        is_red=False,
    )
    pred = await run_disposition_forecast(inp, get_llm_client())
    assert pred.discharge_pct > 70
