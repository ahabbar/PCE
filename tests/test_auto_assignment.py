from __future__ import annotations
import asyncio, pytest
from dataclasses import dataclass
from src.orchestrator.score_engine import should_auto_assign, LabResult


def _make_result(is_critical: bool = False) -> LabResult:
    return LabResult(
        patient_id="TEST-001",
        test_name="Creatinine",
        result_value="1.9",
        result_time=0.0,
        is_critical=is_critical,
    )


def test_should_auto_assign_critical():
    result = _make_result(is_critical=True)
    assert should_auto_assign(result, result_count=1, already_assigned=False) is True


def test_should_auto_assign_two_results():
    result = _make_result(is_critical=False)
    assert should_auto_assign(result, result_count=2, already_assigned=False) is True


def test_should_not_assign_one_result():
    result = _make_result(is_critical=False)
    assert should_auto_assign(result, result_count=1, already_assigned=False) is False


def test_should_not_assign_zero_results():
    result = _make_result(is_critical=False)
    assert should_auto_assign(result, result_count=0, already_assigned=False) is False


def test_should_not_assign_already_assigned():
    result = _make_result(is_critical=True)
    assert should_auto_assign(result, result_count=5, already_assigned=True) is False


def test_should_not_assign_already_assigned_non_critical():
    result = _make_result(is_critical=False)
    assert should_auto_assign(result, result_count=3, already_assigned=True) is False


def test_cascade_assigns_unassigned_patient():
    from src.orchestrator.queue import get_queue, QueueManager
    from src.core.patient import PatientRecord, TriageScoreResult, IntakeForm, ChiefComplaint, Vitals, MedicalHistory
    from src.agents.load_balancer import create_default_doctor_pool, DoctorLoadBalancer as LoadBalancer
    import time

    # Fresh queue and LB for isolation
    q = QueueManager()
    lb = LoadBalancer(create_default_doctor_pool())

    intake = IntakeForm(
        patient_id="CASCADE-001", arrival_time=time.time(),
        age_years=45, gender="male",
        chief_complaint=ChiefComplaint(free_text_en="chest pain", category="chest_pain"),
        vitals=Vitals(heart_rate=100), medical_history=MedicalHistory(),
    )
    ts = TriageScoreResult(
        risk_score=75.0, esi_level=3, confidence=0.9, is_red=False,
        threshold_used=85.0, key_factors=["chest"], reasoning="test", pce_scope="pce_core",
    )
    pr = PatientRecord(intake=intake, triage_score=ts, status="waiting", result_count=1)

    async def run():
        await q.add_patient_record(pr)
        sorted_q = await q.get_sorted_queue()
        assert len(sorted_q) == 1
        assert sorted_q[0].result_count == 1
        # Simulate cascade: assign via LB
        assignment = lb.assign_patient("CASCADE-001", esi_level=3)
        assert assignment.assigned_doctor is not None
        await q.assign_doctor("CASCADE-001", assignment.assigned_doctor.name)
        updated = await q.get_sorted_queue()
        assert updated[0].assigned_doctor == assignment.assigned_doctor.name

    asyncio.run(run())
