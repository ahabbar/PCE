from __future__ import annotations
import asyncio, logging, time
from src.core.patient import PatientRecord, TriageScoreResult, RedFlagResult
from src.agents.triage_score import sort_queue
from src.orchestrator.engine import TriageResult

logger = logging.getLogger("pce.queue")


class QueueManager:
    def __init__(self) -> None:
        self._patients: dict[str, PatientRecord] = {}
        self._lock = asyncio.Lock()

    async def add_patient(self, result: TriageResult) -> int:
        async with self._lock:
            record = PatientRecord(
                intake=result.intake,
                esi_result=result.esi_result,
                triage_score=result.triage_score,
                red_flag=result.red_flag,
                workup=result.workup,
                status="waiting",
                arrival_time=result.intake.arrival_time,
            )
            self._patients[result.patient_id] = record
            sorted_q = sort_queue(list(self._patients.values()))
            for i, p in enumerate(sorted_q):
                if p.intake.patient_id == result.patient_id:
                    return i + 1
            return len(self._patients)

    async def update_score(
        self, patient_id: str, new_score: float, is_red: bool
    ) -> bool:
        async with self._lock:
            if patient_id not in self._patients:
                return False
            old_sorted = sort_queue(list(self._patients.values()))
            record = self._patients[patient_id]
            old_score = record.triage_score
            new_triage = TriageScoreResult(
                risk_score=new_score,
                esi_level=old_score.esi_level,
                confidence=old_score.confidence,
                is_red=is_red,
                threshold_used=old_score.threshold_used,
                key_factors=old_score.key_factors,
                reasoning=old_score.reasoning,
                pce_scope=old_score.pce_scope,
            )
            record = PatientRecord(
                intake=record.intake,
                esi_result=record.esi_result,
                triage_score=new_triage,
                red_flag=record.red_flag,
                workup=record.workup,
                status=record.status,
            )
            self._patients[patient_id] = record
            new_sorted = sort_queue(list(self._patients.values()))
            return (
                [p.intake.patient_id for p in old_sorted]
                != [p.intake.patient_id for p in new_sorted]
            )

    async def get_sorted_queue(self) -> list[PatientRecord]:
        async with self._lock:
            return sort_queue(list(self._patients.values()))

    async def assign_doctor(self, patient_id: str, doctor_name: str) -> None:
        async with self._lock:
            if patient_id in self._patients:
                r = self._patients[patient_id]
                self._patients[patient_id] = PatientRecord(
                    intake=r.intake,
                    esi_result=r.esi_result,
                    triage_score=r.triage_score,
                    red_flag=r.red_flag,
                    workup=r.workup,
                    status="assigned",
                    assigned_doctor=doctor_name,
                    result_count=r.result_count,
                )

    async def admit_to_bed(self, patient_id: str) -> bool:
        """Move a 'waiting' patient into a treatment bed (status='workup').
        Returns True if state changed. ESI-1 patients are never bedded — they
        bypass beds entirely and live in Resus."""
        async with self._lock:
            if patient_id not in self._patients:
                return False
            r = self._patients[patient_id]
            if r.status != "waiting":
                return False
            if r.triage_score and r.triage_score.esi_level == 1:
                return False
            self._patients[patient_id] = PatientRecord(
                intake=r.intake,
                esi_result=r.esi_result,
                triage_score=r.triage_score,
                red_flag=r.red_flag,
                workup=r.workup,
                status="workup",
                assigned_doctor=r.assigned_doctor,
                result_count=r.result_count,
            )
            return True

    async def unassign_doctor(self, patient_id: str) -> None:
        """Clear assigned_doctor and revert status to 'waiting'. Used when an
        incoming ESI-1 bumps this patient off their doctor so they can be
        re-cascaded later."""
        async with self._lock:
            if patient_id in self._patients:
                r = self._patients[patient_id]
                self._patients[patient_id] = PatientRecord(
                    intake=r.intake,
                    esi_result=r.esi_result,
                    triage_score=r.triage_score,
                    red_flag=r.red_flag,
                    workup=r.workup,
                    status="waiting",
                    assigned_doctor=None,
                    result_count=r.result_count,
                )

    async def discharge(self, patient_id: str) -> None:
        async with self._lock:
            self._patients.pop(patient_id, None)

    async def clear_all(self) -> None:
        async with self._lock:
            self._patients.clear()

    async def increment_result_count(self, patient_id: str) -> tuple[int, str | None]:
        """Returns (new_result_count, current_assigned_doctor)."""
        async with self._lock:
            if patient_id not in self._patients:
                return 0, None
            r = self._patients[patient_id]
            new_count = r.result_count + 1
            self._patients[patient_id] = PatientRecord(
                intake=r.intake,
                esi_result=r.esi_result,
                triage_score=r.triage_score,
                red_flag=r.red_flag,
                workup=r.workup,
                status=r.status,
                assigned_doctor=r.assigned_doctor,
                result_count=new_count,
            )
            return new_count, r.assigned_doctor

    async def add_patient_record(self, record: PatientRecord) -> None:
        async with self._lock:
            self._patients[record.intake.patient_id] = record

    async def queue_depth(self) -> int:
        async with self._lock:
            return len(self._patients)


_queue: QueueManager | None = None


def get_queue() -> QueueManager:
    global _queue
    if _queue is None:
        _queue = QueueManager()
    return _queue
