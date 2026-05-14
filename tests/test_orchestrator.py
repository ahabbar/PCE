from __future__ import annotations
import pytest
import asyncio
import uuid
import time
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.patient import (
    IntakeForm,
    Vitals,
    ChiefComplaint,
    MedicalHistory,
    NurseObservation,
    TriageScoreResult,
    RedFlagResult,
    WorkupPlan,
)
from src.core.esi_algorithm import ESIResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_uti_intake() -> IntakeForm:
    return IntakeForm(
        patient_id=str(uuid.uuid4()),
        age_years=34,
        gender="female",
        chief_complaint=ChiefComplaint(
            free_text_en="dysuria and flank pain",
            category="urinary",
        ),
        vitals=Vitals(heart_rate=98, temperature_c=38.6, systolic_bp=118, spo2_pct=97),
    )


def make_critical_intake() -> IntakeForm:
    return IntakeForm(
        patient_id=str(uuid.uuid4()),
        age_years=65,
        gender="male",
        chief_complaint=ChiefComplaint(
            free_text_en="found unresponsive",
            category="altered_mental_status",
        ),
        vitals=Vitals(spo2_pct=72, heart_rate=140),
        nurse_observation=NurseObservation(
            work_of_breathing="apneic",
            avpu="P",
            general_appearance="critically_ill",
        ),
    )


def _make_mock_triage_score(risk_score: float = 65.0) -> TriageScoreResult:
    return TriageScoreResult(
        risk_score=risk_score,
        esi_level=3,
        confidence=0.8,
        is_red=False,
        threshold_used=85.0,
        key_factors=["pyelonephritis", "fever"],
        reasoning="Likely pyelonephritis — urgent BMP.",
        pce_scope="pce_core",
    )


def _make_mock_red_flag(is_emergency: bool = False, esi1_immediate: bool = False) -> RedFlagResult:
    return RedFlagResult(
        is_emergency=is_emergency,
        esi1_immediate=esi1_immediate,
        layer_triggered="rules",
        reasoning="test mock",
    )


def _make_mock_workup(patient_id: str) -> WorkupPlan:
    from src.core.patient import OrderItem
    return WorkupPlan(
        patient_id=patient_id,
        protocol_key="urinary",
        orders=[
            OrderItem(
                test_name="Urinalysis",
                reason="Suspected UTI",
                cost_tier="low",
                timing="urgent",
                whitelist_rule="UTI_001",
            )
        ],
        deferred_to_doctor=[],
        reasoning="Standard UTI workup.",
    )


# ── Patch targets ─────────────────────────────────────────────────────────────

_PATCHES = {
    "triage_score": "src.orchestrator.engine.run_triage_score",
    "red_flag": "src.orchestrator.engine.run_red_flag_guardian",
    "workup": "src.orchestrator.engine.run_preemptive_workup",
    "save_patient": "src.orchestrator.engine.save_patient",
    "save_orders": "src.orchestrator.engine.save_investigation_orders",
    "log_action": "src.orchestrator.engine.log_agent_action",
}


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_esi1_bypasses_agents():
    """ESI-1 critical patient must skip all LLM agents."""
    intake = make_critical_intake()

    with (
        patch(_PATCHES["triage_score"]) as mock_ts,
        patch(_PATCHES["red_flag"]) as mock_rf,
        patch(_PATCHES["workup"]) as mock_wu,
        patch(_PATCHES["save_patient"], new_callable=AsyncMock),
        patch(_PATCHES["save_orders"], new_callable=AsyncMock),
        patch(_PATCHES["log_action"], new_callable=AsyncMock),
    ):
        from src.orchestrator.engine import process_patient
        result = await process_patient(intake, patient_load=10)

    assert result.workup is None
    assert result.is_red is True
    assert result.parallel_confirmed is False
    mock_ts.assert_not_called()
    mock_rf.assert_not_called()
    mock_wu.assert_not_called()


@pytest.mark.asyncio
async def test_parallel_confirmed():
    """Non-ESI-1 patient should have parallel_confirmed=True."""
    intake = make_uti_intake()
    mock_score = _make_mock_triage_score(65.0)
    mock_flag = _make_mock_red_flag()
    mock_wu = _make_mock_workup(intake.patient_id)

    with (
        patch(_PATCHES["triage_score"], new_callable=AsyncMock, return_value=mock_score),
        patch(_PATCHES["red_flag"], new_callable=AsyncMock, return_value=mock_flag),
        patch(_PATCHES["workup"], new_callable=AsyncMock, return_value=mock_wu),
        patch(_PATCHES["save_patient"], new_callable=AsyncMock),
        patch(_PATCHES["save_orders"], new_callable=AsyncMock),
        patch(_PATCHES["log_action"], new_callable=AsyncMock),
    ):
        from src.orchestrator.engine import process_patient
        result = await process_patient(intake, patient_load=10)

    assert result.parallel_confirmed is True


@pytest.mark.asyncio
async def test_is_red_set_by_orchestrator():
    """Orchestrator must override LLM is_red when score >= threshold."""
    intake = make_uti_intake()
    # LLM returns is_red=False but score=88, threshold at load=10 is 85 → orchestrator sets True
    mock_score = TriageScoreResult(
        risk_score=88.0,
        esi_level=3,
        confidence=0.8,
        is_red=False,  # LLM says not red
        threshold_used=85.0,
        key_factors=["high score"],
        reasoning="High but agent says not red.",
        pce_scope="pce_core",
    )
    mock_flag = _make_mock_red_flag(is_emergency=False)
    mock_wu = _make_mock_workup(intake.patient_id)

    with (
        patch(_PATCHES["triage_score"], new_callable=AsyncMock, return_value=mock_score),
        patch(_PATCHES["red_flag"], new_callable=AsyncMock, return_value=mock_flag),
        patch(_PATCHES["workup"], new_callable=AsyncMock, return_value=mock_wu),
        patch(_PATCHES["save_patient"], new_callable=AsyncMock),
        patch(_PATCHES["save_orders"], new_callable=AsyncMock),
        patch(_PATCHES["log_action"], new_callable=AsyncMock),
    ):
        from src.orchestrator.engine import process_patient
        result = await process_patient(intake, patient_load=10)

    # 88 >= 85 threshold → orchestrator must set is_red True
    assert result.is_red is True
    assert result.triage_score.is_red is True


@pytest.mark.asyncio
async def test_red_flag_override_skips_workup():
    """When red_flag.esi1_immediate=True the workup must be skipped."""
    intake = make_uti_intake()
    mock_score = _make_mock_triage_score(70.0)
    mock_flag = _make_mock_red_flag(is_emergency=True, esi1_immediate=True)

    with (
        patch(_PATCHES["triage_score"], new_callable=AsyncMock, return_value=mock_score),
        patch(_PATCHES["red_flag"], new_callable=AsyncMock, return_value=mock_flag),
        patch(_PATCHES["workup"], new_callable=AsyncMock) as mock_wu,
        patch(_PATCHES["save_patient"], new_callable=AsyncMock),
        patch(_PATCHES["save_orders"], new_callable=AsyncMock),
        patch(_PATCHES["log_action"], new_callable=AsyncMock),
    ):
        from src.orchestrator.engine import process_patient
        result = await process_patient(intake, patient_load=10)

    assert result.workup is None
    mock_wu.assert_not_called()


@pytest.mark.asyncio
async def test_queue_sort():
    """Queue must be sorted by risk_score descending."""
    from src.orchestrator.queue import QueueManager
    from src.orchestrator.engine import TriageResult
    from src.core.esi_algorithm import ESIResult

    qm = QueueManager()
    scores = [40.0, 80.0, 60.0, 90.0]

    for score in scores:
        intake = make_uti_intake()
        esi = ESIResult(
            esi_level=3,
            decision_point_reached="C",
            pce_scope="pce_core",
        )
        ts = TriageScoreResult(
            risk_score=score,
            esi_level=3,
            confidence=0.8,
            is_red=False,
            threshold_used=85.0,
            key_factors=["test"],
            reasoning="test",
            pce_scope="pce_core",
        )
        rf = _make_mock_red_flag()
        result = TriageResult(
            patient_id=intake.patient_id,
            intake=intake,
            esi_result=esi,
            triage_score=ts,
            red_flag=rf,
            workup=None,
            total_latency_ms=100.0,
            is_red=False,
            threshold_used=85.0,
            parallel_confirmed=True,
        )
        await qm.add_patient(result)

    sorted_q = await qm.get_sorted_queue()
    assert sorted_q[0].triage_score.risk_score == 90.0


@pytest.mark.asyncio
async def test_queue_fairness_bonus():
    """Long-waiting patient with score 50 should beat a fresh patient with score 60 due to +15 bonus."""
    from src.orchestrator.queue import QueueManager
    from src.orchestrator.engine import TriageResult
    from src.core.esi_algorithm import ESIResult

    qm = QueueManager()

    def _make_result(score: float, arrival_time: float) -> TriageResult:
        intake = IntakeForm(
            patient_id=str(uuid.uuid4()),
            age_years=40,
            gender="female",
            chief_complaint=ChiefComplaint(free_text_en="test", category="other"),
            vitals=Vitals(heart_rate=80),
            arrival_time=arrival_time,
        )
        esi = ESIResult(
            esi_level=3,
            decision_point_reached="C",
            pce_scope="pce_core",
        )
        ts = TriageScoreResult(
            risk_score=score,
            esi_level=3,
            confidence=0.8,
            is_red=False,
            threshold_used=85.0,
            key_factors=["test"],
            reasoning="test",
            pce_scope="pce_core",
        )
        rf = _make_mock_red_flag()
        return TriageResult(
            patient_id=intake.patient_id,
            intake=intake,
            esi_result=esi,
            triage_score=ts,
            red_flag=rf,
            workup=None,
            total_latency_ms=100.0,
            is_red=False,
            threshold_used=85.0,
            parallel_confirmed=True,
        )

    now = time.time()
    # Long-waiter: arrived 1 hour ago, score 50 → bonus = min(60*0.3, 15) = 15 → effective 65
    long_waiter = _make_result(50.0, now - 3600)
    # Fresh patient: just arrived, score 60 → no bonus → effective 60
    fresh_patient = _make_result(60.0, now)

    await qm.add_patient(long_waiter)
    await qm.add_patient(fresh_patient)

    sorted_q = await qm.get_sorted_queue()
    # Long-waiter (effective 65) should be first
    assert sorted_q[0].triage_score.risk_score == 50.0
