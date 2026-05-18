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


def test_overloaded_force_assigns():
    """Non-ESI-1 overflow: still force-assigns to least-loaded preferred role."""
    lb = fresh_balancer()
    for i, doctor in enumerate(lb.get_all_doctors()):
        for j in range(doctor.max_cases):
            doctor.active_cases.append(ActiveCase(f"pt-{i}-{j}", 3, time.time(), 15))
    result = lb.assign_patient("pt-overflow", esi_level=3)
    assert result.assigned_doctor is not None
    assert "overflow" in result.reason.lower()
    assert result.bumped_patient_id is None


def test_prefers_empty_doctor_over_loaded():
    """When two doctors of the same role are candidates, the empty one wins
    even if the loaded one has shorter remaining time."""
    pool = [
        Doctor("DR1", "Dr. Loaded", DoctorRole.SHO, max_cases=2),
        Doctor("DR2", "Dr. Empty",  DoctorRole.SHO, max_cases=2),
    ]
    # Loaded doctor has 1 patient with only 1 minute remaining
    pool[0].active_cases.append(ActiveCase("old-pt", 5, time.time(), 1))
    lb = DoctorLoadBalancer(pool)
    result = lb.assign_patient("pt-new", esi_level=4)
    assert result.assigned_doctor is pool[1], "should pick empty doctor first"


def test_esi1_bumps_non_esi1_when_no_empty_doctor():
    """ESI-1 arriving with all doctors full bumps a non-ESI-1 patient."""
    lb = fresh_balancer()
    # Saturate every doctor (fill to max_cases) with non-ESI-1 patients
    for i, doctor in enumerate(lb.get_all_doctors()):
        for j in range(doctor.max_cases):
            doctor.active_cases.append(ActiveCase(f"victim-{i}-{j}", 4, time.time(), 10))
    result = lb.assign_patient("pt-esi1", esi_level=1)
    assert result.assigned_doctor is not None
    assert result.bumped_patient_id is not None
    assert result.bumped_patient_id.startswith("victim-")
    # The chosen doctor should now have ESI-1 and exactly one fewer case than max
    doc = result.assigned_doctor
    assert doc.has_esi1
    assert any(c.esi_level == 1 for c in doc.active_cases)


def test_esi1_picks_empty_doctor_before_bumping():
    """ESI-1 should fill an empty doctor first, never bump unnecessarily."""
    lb = fresh_balancer()
    # Fill 4 of 5 doctors with ESI-4 patients; leave one (last) empty
    docs = lb.get_all_doctors()
    for i, doctor in enumerate(docs[:-1]):
        doctor.active_cases.append(ActiveCase(f"v-{i}", 4, time.time(), 10))
    result = lb.assign_patient("pt-esi1", esi_level=1)
    assert result.assigned_doctor is docs[-1] or len(result.assigned_doctor.active_cases) == 1
    assert result.bumped_patient_id is None


def test_doctor_with_esi1_is_not_available():
    """Once a doctor holds an ESI-1, no other patient can be added."""
    lb = fresh_balancer()
    lb.assign_patient("pt-esi1", esi_level=1)
    chosen = next(d for d in lb.get_all_doctors() if d.has_esi1)
    assert not chosen.is_available
    assert chosen.load_pct == 100.0


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


def test_esi1_bumps_globally_least_severe_not_preferred_role():
    """ESI-1 arriving must bump the LEAST-severe patient across all chairs,
    not whoever happens to be on the highest-seniority doctor. The previous
    code bumped the Consultant's ESI-2 even when an SHO held an ESI-5."""
    lb = fresh_balancer()
    docs = lb.get_all_doctors()
    # Put a high-acuity ESI-2 on the Consultant
    consultant = next(d for d in docs if d.role == DoctorRole.CONSULTANT)
    consultant.active_cases.append(ActiveCase("ESI2-on-consultant", 2, time.time(), 22, risk_score=90.0))
    # Fill every other doctor with low-acuity ESI-5
    for d in docs:
        if d is consultant:
            continue
        d.active_cases.append(ActiveCase(f"ESI5-on-{d.name}", 5, time.time(), 7, risk_score=10.0))

    result = lb.assign_patient("pt-esi1", esi_level=1)
    assert result.bumped_patient_id is not None
    assert result.bumped_patient_id.startswith("ESI5-"), (
        f"ESI-1 should bump a least-severe ESI-5, not the Consultant's ESI-2 "
        f"(bumped: {result.bumped_patient_id})"
    )


def test_severity_bump_picks_globally_least_severe():
    """Non-ESI-1 incoming: when multiple victims are bumpable, pick the
    one with the highest ESI number (least severe), regardless of role."""
    lb = fresh_balancer()
    docs = lb.get_all_doctors()
    # Consultant holds an ESI-3 (somewhat severe)
    consultant = next(d for d in docs if d.role == DoctorRole.CONSULTANT)
    consultant.active_cases.append(ActiveCase("c-esi3", 3, time.time(), 15, risk_score=60.0))
    # Every other doctor holds an ESI-4 or ESI-5
    sho = next(d for d in docs if d.role == DoctorRole.SHO)
    sho.active_cases.append(ActiveCase("sho-esi5", 5, time.time(), 7, risk_score=10.0))
    for d in docs:
        if d in (consultant, sho):
            continue
        d.active_cases.append(ActiveCase(f"x-{d.name}", 4, time.time(), 10, risk_score=20.0))

    # Incoming: ESI-2 risk 80
    result = lb.assign_patient("pt-urgent", esi_level=2, risk_score=80.0)
    assert result.bumped_patient_id == "sho-esi5", (
        f"should bump the globally least-severe ESI-5, not the Consultant's ESI-3 "
        f"(bumped: {result.bumped_patient_id})"
    )


def test_higher_severity_bumps_lower_when_all_full():
    """When every doctor is busy, an incoming patient that is STRICTLY more
    severe than someone in a chair bumps the lowest-severity victim."""
    lb = fresh_balancer()
    # Saturate all 5 doctors with ESI-3 risk=40 patients
    for i, doctor in enumerate(lb.get_all_doctors()):
        doctor.active_cases.append(ActiveCase(f"victim-{i}", 3, time.time(), 15, risk_score=40.0))
    # Incoming: ESI-2 risk=80 (more severe on both axes)
    result = lb.assign_patient("pt-urgent", esi_level=2, risk_score=80.0)
    assert result.assigned_doctor is not None
    assert result.bumped_patient_id is not None
    assert result.bumped_patient_id.startswith("victim-")
    assert "more severe" in result.reason.lower()


def test_lower_severity_does_not_bump_higher():
    """An incoming low-severity patient must NEVER displace a higher-severity
    one. Falls through to the overflow path instead."""
    lb = fresh_balancer()
    # Saturate all 5 doctors with HIGH severity ESI-2 risk=90 patients
    for i, doctor in enumerate(lb.get_all_doctors()):
        doctor.active_cases.append(ActiveCase(f"high-{i}", 2, time.time(), 22, risk_score=90.0))
    # Incoming: ESI-4 risk=20 (less severe)
    result = lb.assign_patient("pt-mild", esi_level=4, risk_score=20.0)
    assert result.bumped_patient_id is None, "must never bump a more-severe patient"
    assert "overflow" in result.reason.lower()


def test_risk_score_breaks_esi_tie_for_bump():
    """Equal ESI: higher risk_score wins and bumps the lower-risk victim."""
    lb = fresh_balancer()
    for i, doctor in enumerate(lb.get_all_doctors()):
        doctor.active_cases.append(ActiveCase(f"v-{i}", 3, time.time(), 15, risk_score=30.0))
    # Same ESI, higher risk
    result = lb.assign_patient("pt-tied", esi_level=3, risk_score=75.0)
    assert result.bumped_patient_id is not None
    assert result.bumped_patient_id.startswith("v-")


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
