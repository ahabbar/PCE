from __future__ import annotations
import time, pytest
from src.agents.load_balancer import (
    DoctorLoadBalancer, Doctor, DoctorRole, ActiveCase,
    create_default_doctor_pool, ESI_SERVICE_TIMES
)


def fresh_balancer() -> DoctorLoadBalancer:
    return DoctorLoadBalancer(create_default_doctor_pool())


def test_esi2_assigned_to_consultant():
    lb = fresh_balancer()
    result = lb.assign_patient("pt-001", esi_level=2)
    assert result.assigned_doctor is not None
    assert result.assigned_doctor.role == DoctorRole.CONSULTANT


def test_esi4_assigned_to_sho():
    lb = fresh_balancer()
    result = lb.assign_patient("pt-002", esi_level=4)
    assert result.assigned_doctor is not None
    assert result.assigned_doctor.role == DoctorRole.SHO


def test_least_loaded_selected():
    lb = fresh_balancer()
    # Give one SHO a case
    sho1 = next(d for d in lb.get_all_doctors() if d.role == DoctorRole.SHO)
    sho1.active_cases.append(ActiveCase("old-pt", 4, time.time(), 10))
    # Assign new ESI-4 patient — should go to the free SHO
    result = lb.assign_patient("pt-new", esi_level=4)
    assert result.assigned_doctor is not None
    assert result.assigned_doctor != sho1 or sho1.total_remaining_min == 0


def test_overloaded_returns_none():
    lb = fresh_balancer()
    # Fill all doctors to max_cases
    for i, doctor in enumerate(lb.get_all_doctors()):
        for j in range(doctor.max_cases):
            doctor.active_cases.append(ActiveCase(f"pt-{i}-{j}", 3, time.time(), 15))
    result = lb.assign_patient("pt-overflow", esi_level=3)
    assert result.assigned_doctor is None
    assert "capacity" in result.reason.lower()


def test_complete_case_frees_doctor():
    lb = fresh_balancer()
    result = lb.assign_patient("pt-001", esi_level=3)
    doctor = result.assigned_doctor
    assert doctor is not None
    assert not doctor.is_available or doctor.max_cases > 1
    lb.complete_case("pt-001")
    assert doctor.is_available


def test_load_report_sorted():
    lb = fresh_balancer()
    lb.assign_patient("pt-1", 2)
    lb.assign_patient("pt-2", 2)
    report = lb.get_load_report()
    loads = [r["load_pct"] for r in report]
    assert loads == sorted(loads, reverse=True)


def test_seniority_escalation():
    # Pool with only SHOs — ESI-2 must escalate to SHO
    pool = [
        Doctor("DR1", "Dr. Only SHO", DoctorRole.SHO, max_cases=2),
        Doctor("DR2", "Dr. Also SHO", DoctorRole.SHO, max_cases=2),
    ]
    lb = DoctorLoadBalancer(pool)
    result = lb.assign_patient("pt-001", esi_level=2)
    assert result.assigned_doctor is not None
    assert result.assigned_doctor.role == DoctorRole.SHO
