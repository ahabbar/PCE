from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DoctorRole(str, Enum):
    CONSULTANT       = "consultant"
    SENIOR_REGISTRAR = "senior_registrar"
    REGISTRAR        = "registrar"
    SHO              = "sho"


ESI_SERVICE_TIMES = {1: 45, 2: 22, 3: 15, 4: 10, 5: 7}

# Ordered by preference for each ESI level (first = most preferred)
ESI_PREFERRED_ROLES = {
    1: [DoctorRole.CONSULTANT, DoctorRole.SENIOR_REGISTRAR, DoctorRole.REGISTRAR, DoctorRole.SHO],
    2: [DoctorRole.CONSULTANT, DoctorRole.SENIOR_REGISTRAR, DoctorRole.REGISTRAR, DoctorRole.SHO],
    3: [DoctorRole.REGISTRAR, DoctorRole.SHO, DoctorRole.SENIOR_REGISTRAR, DoctorRole.CONSULTANT],
    4: [DoctorRole.SHO, DoctorRole.REGISTRAR, DoctorRole.SENIOR_REGISTRAR, DoctorRole.CONSULTANT],
    5: [DoctorRole.SHO, DoctorRole.REGISTRAR, DoctorRole.SENIOR_REGISTRAR, DoctorRole.CONSULTANT],
}


@dataclass
class ActiveCase:
    patient_id: str
    esi_level: int
    started_at: float
    estimated_total_min: int

    @property
    def elapsed_min(self) -> float:
        return (time.time() - self.started_at) / 60

    @property
    def remaining_min(self) -> float:
        return max(0.0, self.estimated_total_min - self.elapsed_min)


@dataclass
class Doctor:
    doctor_id: str
    name: str
    role: DoctorRole
    max_cases: int = 2
    active_cases: list = field(default_factory=list)

    @property
    def total_remaining_min(self) -> float:
        return sum(c.remaining_min for c in self.active_cases)

    @property
    def is_available(self) -> bool:
        return len(self.active_cases) < self.max_cases

    @property
    def load_pct(self) -> float:
        return len(self.active_cases) / self.max_cases * 100


@dataclass
class AssignmentResult:
    patient_id: str
    assigned_doctor: Optional[Doctor]
    reason: str
    estimated_wait_min: float
    load_balanced: bool


class DoctorLoadBalancer:

    def __init__(self, doctors: list[Doctor]):
        self._doctors = doctors

    def assign_patient(self, patient_id: str, esi_level: int) -> AssignmentResult:
        esi_level = max(1, min(5, esi_level))
        preferred_roles = ESI_PREFERRED_ROLES.get(esi_level, ESI_PREFERRED_ROLES[3])
        service_time = ESI_SERVICE_TIMES.get(esi_level, 15)

        # 1) Try each role in preference order among AVAILABLE doctors first
        seen_roles: list[DoctorRole] = []
        for role in preferred_roles:
            if role in seen_roles:
                continue
            seen_roles.append(role)
            candidates = [d for d in self._doctors if d.role == role and d.is_available]
            if candidates:
                chosen = min(candidates, key=lambda d: d.total_remaining_min)
                escalated = role != preferred_roles[0]
                reason = (
                    f"Assigned to {chosen.role.value} (escalated from {preferred_roles[0].value})"
                    if escalated
                    else f"Assigned to {chosen.role.value} (preferred for ESI-{esi_level})"
                )
                load_balanced = len(candidates) > 1
                self._add_case(chosen, patient_id, esi_level, service_time)
                return AssignmentResult(
                    patient_id=patient_id,
                    assigned_doctor=chosen,
                    reason=reason,
                    estimated_wait_min=chosen.total_remaining_min - service_time,
                    load_balanced=load_balanced,
                )

        # 2) No doctor available in any preferred role — force-assign anyway
        #    (overflow). Walk preferred roles again, this time ignoring the
        #    is_available check, so role preference still wins over raw load.
        chosen = None
        chosen_role = None
        for role in preferred_roles:
            candidates = [d for d in self._doctors if d.role == role]
            if not candidates:
                continue
            chosen = min(candidates, key=lambda d: d.total_remaining_min)
            chosen_role = role
            break
        if chosen is None:
            chosen = min(self._doctors, key=lambda d: d.total_remaining_min)
            chosen_role = chosen.role
        self._add_case(chosen, patient_id, esi_level, service_time)
        reason = (
            f"ESI-1 force-assigned to {chosen_role.value} (overflow — all doctors at capacity)"
            if esi_level == 1
            else f"Overflow assignment to {chosen_role.value} (all preferred roles full)"
        )
        return AssignmentResult(
            patient_id=patient_id,
            assigned_doctor=chosen,
            reason=reason,
            estimated_wait_min=max(0.0, chosen.total_remaining_min - service_time),
            load_balanced=False,
        )

    def _add_case(self, doctor: "Doctor", patient_id: str, esi_level: int, service_time: int) -> None:
        case = ActiveCase(
            patient_id=patient_id,
            esi_level=esi_level,
            started_at=time.time(),
            estimated_total_min=service_time,
        )
        doctor.active_cases.append(case)

    def complete_case(self, patient_id: str) -> Optional[Doctor]:
        for doctor in self._doctors:
            for case in doctor.active_cases:
                if case.patient_id == patient_id:
                    doctor.active_cases.remove(case)
                    return doctor
        return None

    def get_load_report(self) -> list[dict]:
        report = [
            {
                "doctor_id": d.doctor_id,
                "name": d.name,
                "role": d.role.value,
                "active_cases": len(d.active_cases),
                "load_pct": d.load_pct,
                "total_remaining_min": d.total_remaining_min,
            }
            for d in self._doctors
        ]
        return sorted(report, key=lambda x: x["load_pct"], reverse=True)

    def reset_all(self) -> None:
        """Clear all active cases from all doctors (call on demo reset/seed)."""
        for doctor in self._doctors:
            doctor.active_cases.clear()

    def get_all_doctors(self) -> list[Doctor]:
        return self._doctors


def create_default_doctor_pool() -> list[Doctor]:
    return [
        Doctor("DR1", "Dr. Rahman",  DoctorRole.CONSULTANT,       max_cases=2),
        Doctor("DR2", "Dr. Chen",    DoctorRole.SENIOR_REGISTRAR,  max_cases=2),
        Doctor("DR3", "Dr. Patel",   DoctorRole.REGISTRAR,         max_cases=2),
        Doctor("DR4", "Dr. Okafor",  DoctorRole.SHO,               max_cases=2),
        Doctor("DR5", "Dr. Al-Sayd", DoctorRole.SHO,               max_cases=2),
    ]


_balancer: DoctorLoadBalancer | None = None


def get_load_balancer() -> DoctorLoadBalancer:
    global _balancer
    if _balancer is None:
        _balancer = DoctorLoadBalancer(create_default_doctor_pool())
    return _balancer
