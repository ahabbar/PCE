from __future__ import annotations
import pytest, uuid, time
from src.agents.exit_coord import DispositionType, ExitInput
from src.core.patient import IntakeForm, Vitals, ChiefComplaint, MedicalHistory, TriageScoreResult, ExitPlan


def _make_inp(disposition: DispositionType) -> ExitInput:
    intake = IntakeForm(
        patient_id=str(uuid.uuid4()), arrival_time=time.time(),
        age_years=34, gender="female",
        chief_complaint=ChiefComplaint(free_text_en="fever and flank pain", category="urinary", pain_present=True),
        vitals=Vitals(heart_rate=98, temperature_c=38.6),
        medical_history=MedicalHistory(known_diagnoses=["CKD stage 2"]),
    )
    score = TriageScoreResult(risk_score=80, esi_level=3, confidence=0.85, is_red=False,
                               threshold_used=85, key_factors=["AKI"], reasoning="")
    return ExitInput(
        intake=intake, triage_score=score, disposition=disposition,
        confirmed_diagnosis="Pyelonephritis + AKI stage 2",
        results_summary=["Creatinine 1.9", "WBC 16,800"],
        assigned_doctor="Dr. Rahman",
    )


@pytest.mark.real_llm
@pytest.mark.asyncio
async def test_discharge_has_instructions():
    from src.agents.exit_coord import run_exit_coordinator
    from src.agents.llm import get_llm_client
    plan = await run_exit_coordinator(_make_inp(DispositionType.DISCHARGE), get_llm_client())
    assert plan.instructions is not None and len(plan.instructions) > 20


@pytest.mark.real_llm
@pytest.mark.asyncio
async def test_admit_has_handover():
    from src.agents.exit_coord import run_exit_coordinator
    from src.agents.llm import get_llm_client
    plan = await run_exit_coordinator(_make_inp(DispositionType.ADMIT), get_llm_client())
    assert plan.handover_note is not None and len(plan.handover_note) > 20


@pytest.mark.real_llm
@pytest.mark.asyncio
async def test_icu_has_handover():
    from src.agents.exit_coord import run_exit_coordinator
    from src.agents.llm import get_llm_client
    plan = await run_exit_coordinator(_make_inp(DispositionType.ICU), get_llm_client())
    assert plan.handover_note is not None
    assert plan.transport_needed == False


@pytest.mark.real_llm
@pytest.mark.asyncio
async def test_transfer_needs_transport():
    from src.agents.exit_coord import run_exit_coordinator
    from src.agents.llm import get_llm_client
    plan = await run_exit_coordinator(_make_inp(DispositionType.TRANSFER), get_llm_client())
    assert plan.transport_needed == True
