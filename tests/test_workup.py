from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.workup import evaluate_conditional, run_preemptive_workup
from src.core.patient import (
    ChiefComplaint,
    IntakeForm,
    MedicalHistory,
    TriageScoreResult,
    Vitals,
)
from src.core.whitelist import ProtocolWhitelist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_llm():
    from pydantic import BaseModel

    class _Empty(BaseModel):
        additional_tests: list = []

    mock = MagicMock()
    mock.call = AsyncMock(return_value=_Empty())
    return mock


def make_whitelist() -> ProtocolWhitelist:
    return ProtocolWhitelist("protocols/whitelist.yaml")


def make_intake(
    free_text: str = "dysuria",
    category: str = "urinary",
    temperature_c: float = 37.0,
    heart_rate: float = 80.0,
    age_years: float = 30,
    gender: str = "female",
) -> IntakeForm:
    return IntakeForm(
        patient_id=str(uuid.uuid4()),
        age_years=age_years,
        gender=gender,
        chief_complaint=ChiefComplaint(
            free_text_en=free_text,
            category=category,
        ),
        vitals=Vitals(
            temperature_c=temperature_c,
            heart_rate=heart_rate,
            systolic_bp=120.0,
            spo2_pct=98.0,
        ),
    )


def make_triage_score(esi_level: int = 3) -> TriageScoreResult:
    return TriageScoreResult(
        risk_score=55.0,
        esi_level=esi_level,
        confidence=0.8,
        is_red=False,
        key_factors=["test"],
        reasoning="Test score",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_evaluate_conditional_temp_high():
    whitelist = make_whitelist()
    intake = make_intake(temperature_c=39.0)
    result = evaluate_conditional("temperature > 38.5", intake, whitelist)
    assert result is True


def test_evaluate_conditional_temp_low():
    whitelist = make_whitelist()
    intake = make_intake(temperature_c=37.0)
    result = evaluate_conditional("temperature > 38.5", intake, whitelist)
    assert result is False


@pytest.mark.asyncio
async def test_auto_orders_uti():
    """UTI intake should produce Urinalysis + microscopy, CBC, and BMP orders."""
    whitelist = make_whitelist()
    llm = make_mock_llm()
    intake = make_intake(
        free_text="dysuria and urinary frequency",
        category="urinary",
        temperature_c=37.0,
    )
    triage = make_triage_score(esi_level=3)
    plan = await run_preemptive_workup(intake, triage, llm, whitelist)

    order_names = [o.test_name for o in plan.orders]
    assert "Urinalysis + microscopy" in order_names, f"Got: {order_names}"
    assert "CBC" in order_names, f"Got: {order_names}"
    assert "BMP (Creatinine + Electrolytes)" in order_names, f"Got: {order_names}"


@pytest.mark.asyncio
async def test_conditional_blood_culture_included():
    """Fever protocol with temp 39.2 should include Blood culture x2."""
    whitelist = make_whitelist()
    llm = make_mock_llm()
    intake = make_intake(
        free_text="fever and chills",
        category="fever",
        temperature_c=39.2,
        heart_rate=105.0,
    )
    triage = make_triage_score(esi_level=3)
    plan = await run_preemptive_workup(intake, triage, llm, whitelist)

    order_names = [o.test_name for o in plan.orders]
    assert "Blood culture x2" in order_names, f"Got: {order_names}"


@pytest.mark.asyncio
async def test_conditional_blood_culture_excluded():
    """Fever protocol with temp 37.5 should NOT include Blood culture x2."""
    whitelist = make_whitelist()
    llm = make_mock_llm()
    intake = make_intake(
        free_text="fever and chills",
        category="fever",
        temperature_c=37.5,
        heart_rate=80.0,
    )
    triage = make_triage_score(esi_level=3)
    plan = await run_preemptive_workup(intake, triage, llm, whitelist)

    order_names = [o.test_name for o in plan.orders]
    assert "Blood culture x2" not in order_names, f"Got: {order_names}"


@pytest.mark.asyncio
async def test_requires_doctor_deferred():
    """UTI plan should have CT urogram in deferred_to_doctor, not in orders."""
    whitelist = make_whitelist()
    llm = make_mock_llm()
    intake = make_intake(
        free_text="dysuria and flank pain",
        category="urinary",
    )
    triage = make_triage_score(esi_level=3)
    plan = await run_preemptive_workup(intake, triage, llm, whitelist)

    order_names = [o.test_name for o in plan.orders]
    assert "CT urogram" in plan.deferred_to_doctor, f"Deferred: {plan.deferred_to_doctor}"
    assert "CT urogram" not in order_names, f"Orders: {order_names}"


@pytest.mark.asyncio
async def test_esi1_empty():
    """ESI-1 patients should get an empty workup plan."""
    whitelist = make_whitelist()
    llm = make_mock_llm()
    intake = make_intake(free_text="cardiac arrest")
    triage = make_triage_score(esi_level=1)
    plan = await run_preemptive_workup(intake, triage, llm, whitelist)

    assert plan.orders == []
    assert plan.protocol_key == "none"
