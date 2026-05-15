from __future__ import annotations
import asyncio, pytest, time
from unittest.mock import AsyncMock, MagicMock, patch
from src.orchestrator.score_engine import (
    update_score_on_result, LabResult, reconstruct_intake_from_record, ScoreUpdateEvent
)
from src.core.patient import IntakeForm, TriageScoreResult


def make_record(status="waiting", risk_score=66.0, is_red=False):
    return {
        "patient_id": "pt-test",
        "arrival_time": time.time() - 1800,
        "age_years": 34,
        "gender": "female",
        "chief_complaint_text": "dysuria and flank pain",
        "chief_complaint_category": "urinary",
        "risk_score": risk_score,
        "is_red": is_red,
        "status": status,
        "known_diagnoses": '["CKD stage 2"]',
        "additional_context": "",
    }


def make_lab(is_critical=True, patient_id="pt-test"):
    return LabResult(
        patient_id=patient_id,
        test_name="Creatinine",
        result_value="1.9 mmol/L (elevated)",
        result_time=time.time(),
        is_critical=is_critical,
    )


def make_mock_score(risk_score: float) -> TriageScoreResult:
    return TriageScoreResult(
        risk_score=risk_score,
        esi_level=3,
        confidence=0.8,
        is_red=False,
        threshold_used=85.0,
        key_factors=["test"],
        reasoning="mocked",
        pce_scope="pce_core",
    )


@pytest.mark.asyncio
async def test_discharged_patient_ignored():
    record = make_record(status="discharged")
    result = await update_score_on_result(make_lab(), record)
    assert result is None


@pytest.mark.asyncio
async def test_reconstruct_intake_from_record():
    record = make_record()
    intake = reconstruct_intake_from_record(record)
    assert isinstance(intake, IntakeForm)
    assert intake.patient_id == "pt-test"
    assert intake.age_years == 34


def _setup_mock_queue(mock_queue, update_return: bool):
    q = mock_queue.return_value
    q.update_score = AsyncMock(return_value=update_return)
    q.increment_result_count = AsyncMock(return_value=(1, None))
    q.assign_doctor = AsyncMock()


@pytest.mark.asyncio
async def test_threshold_crossed_detected():
    record = make_record(risk_score=78.0, is_red=False)
    mock_score = make_mock_score(88.0)  # above 85 threshold

    with patch("src.orchestrator.score_engine.run_triage_score", new=AsyncMock(return_value=mock_score)), \
         patch("src.orchestrator.score_engine.log_agent_action", new=AsyncMock()), \
         patch("src.orchestrator.score_engine.get_queue") as mock_queue, \
         patch("src.orchestrator.score_engine.get_load_balancer"):
        _setup_mock_queue(mock_queue, True)
        event = await update_score_on_result(make_lab(is_critical=False), record)

    assert event is not None
    assert event.threshold_crossed is True
    assert event.new_score == 88.0
    assert event.old_score == 78.0


@pytest.mark.asyncio
async def test_small_delta_no_reorder():
    record = make_record(risk_score=66.0, is_red=False)
    mock_score = make_mock_score(69.0)  # delta=3, not critical

    with patch("src.orchestrator.score_engine.run_triage_score", new=AsyncMock(return_value=mock_score)), \
         patch("src.orchestrator.score_engine.log_agent_action", new=AsyncMock()), \
         patch("src.orchestrator.score_engine.get_queue") as mock_queue, \
         patch("src.orchestrator.score_engine.get_load_balancer"):
        _setup_mock_queue(mock_queue, False)
        event = await update_score_on_result(make_lab(is_critical=False), record)

    assert event is not None
    assert event.queue_reordered is False
    assert event.delta == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_queue_reorders_when_delta_large():
    record = make_record(risk_score=66.0, is_red=False)
    mock_score = make_mock_score(82.0)  # delta=16

    with patch("src.orchestrator.score_engine.run_triage_score", new=AsyncMock(return_value=mock_score)), \
         patch("src.orchestrator.score_engine.log_agent_action", new=AsyncMock()), \
         patch("src.orchestrator.score_engine.get_queue") as mock_queue, \
         patch("src.orchestrator.score_engine.get_load_balancer"):
        _setup_mock_queue(mock_queue, True)
        event = await update_score_on_result(make_lab(is_critical=False), record)

    assert event is not None
    assert event.queue_reordered is True
