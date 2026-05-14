from __future__ import annotations

import asyncio
import os
import time
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.triage_score import get_dynamic_threshold, sort_queue
from src.core.patient import (
    ChiefComplaint,
    IntakeForm,
    MedicalHistory,
    PatientRecord,
    TriageScoreResult,
    Vitals,
)


def make_record(risk_score: float, is_red: bool, arrival_offset: float = 0) -> PatientRecord:
    intake = IntakeForm(
        patient_id=str(uuid.uuid4()),
        age_years=30,
        chief_complaint=ChiefComplaint(free_text_en="test"),
        vitals=Vitals(),
    )
    record = PatientRecord(intake=intake)
    object.__setattr__(record, "arrival_time", time.time() - arrival_offset)
    object.__setattr__(
        record,
        "triage_score",
        TriageScoreResult(
            risk_score=risk_score,
            esi_level=3,
            confidence=0.8,
            is_red=is_red,
            key_factors=["test"],
            reasoning="test",
        ),
    )
    return record


def test_dynamic_threshold_quiet():
    assert get_dynamic_threshold(3) == 90.0


def test_dynamic_threshold_normal():
    assert get_dynamic_threshold(10) == 85.0


def test_dynamic_threshold_busy():
    assert get_dynamic_threshold(20) == 80.0


def test_dynamic_threshold_overwhelmed():
    assert get_dynamic_threshold(30) == 75.0


def test_sort_queue_reds_first():
    patients = [
        make_record(60, False),
        make_record(90, True),
        make_record(40, False),
        make_record(95, True),
    ]
    sorted_q = sort_queue(patients)
    assert sorted_q[0].triage_score.is_red
    assert sorted_q[1].triage_score.is_red
    assert not sorted_q[2].triage_score.is_red


def test_sort_queue_by_score():
    scores = [52, 78, 61, 34]
    patients = [make_record(s, False) for s in scores]
    sorted_q = sort_queue(patients)
    sorted_scores = [p.triage_score.risk_score for p in sorted_q]
    assert sorted_scores == sorted(scores, reverse=True)


def test_sort_includes_waiting_bonus():
    long_waiter = make_record(50, False, arrival_offset=50 * 60)
    fresh = make_record(60, False, arrival_offset=0)
    sorted_q = sort_queue([fresh, long_waiter])
    # long_waiter gets bonus of ~15, so 50+15=65 > 60
    assert sorted_q[0].triage_score.risk_score == 50


def test_output_schema_valid():
    t = TriageScoreResult(
        risk_score=67,
        esi_level=3,
        confidence=0.85,
        is_red=False,
        key_factors=["stable"],
        reasoning="Looks okay",
    )
    assert t.risk_score == 67


@pytest.mark.real_llm
@pytest.mark.asyncio
async def test_uti_real_llm():
    os.environ["LLM_PROVIDER"] = "gemini"
    import src.agents.llm as llm_mod
    llm_mod._singleton = None
    from src.agents.llm import get_llm_client
    from src.agents.triage_score import run_triage_score
    from src.core.esi_algorithm import run_esi_algorithm

    intake = IntakeForm(
        patient_id=str(uuid.uuid4()),
        age_years=34,
        gender="female",
        chief_complaint=ChiefComplaint(
            free_text_en="Dysuria and flank pain",
            category="urinary",
            pain_present=True,
            pain_score=5,
        ),
        vitals=Vitals(heart_rate=98, temperature_c=38.6, systolic_bp=118, spo2_pct=97),
        medical_history=MedicalHistory(known_diagnoses=["CKD stage 2"], baseline_cr_abnormal=1.4),
    )
    esi = run_esi_algorithm(intake, 3)
    llm = get_llm_client()
    result = await run_triage_score(intake, esi, llm, patient_load=10)
    assert 40 <= result.risk_score <= 75
    assert not result.is_red


@pytest.mark.real_llm
@pytest.mark.asyncio
async def test_shock_real_llm():
    os.environ["LLM_PROVIDER"] = "gemini"
    import src.agents.llm as llm_mod
    llm_mod._singleton = None
    from src.agents.llm import get_llm_client
    from src.agents.triage_score import run_triage_score
    from src.core.esi_algorithm import run_esi_algorithm
    from src.core.patient import NurseObservation

    intake = IntakeForm(
        patient_id=str(uuid.uuid4()),
        age_years=58,
        gender="male",
        chief_complaint=ChiefComplaint(free_text_en="severe chest pain", category="chest_pain"),
        vitals=Vitals(heart_rate=118, systolic_bp=82, diastolic_bp=50, spo2_pct=91),
        nurse_observation=NurseObservation(skin_assessment="diaphoretic", general_appearance="distressed"),
    )
    esi = run_esi_algorithm(intake, 3)
    llm = get_llm_client()
    result = await run_triage_score(intake, esi, llm, patient_load=10)
    assert result.risk_score >= 85
    assert result.is_red
