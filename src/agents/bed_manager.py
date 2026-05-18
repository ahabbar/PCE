"""Treatment Bay = 10 beds. Highest-risk waiting patients are admitted
to beds as soon as space frees. Doctors are mobile and visit the bed
of whichever patient they are assigned to. Free doctors sit in a pool
and are picked up by `auto_fill_beds` whenever a bedded patient is
doctor-less.

ESI-1 patients NEVER occupy a treatment bed — they go straight to
Resus and pull a doctor with them (handled by load_balancer bump logic).
"""
from __future__ import annotations
import logging

logger = logging.getLogger("pce.beds")

BED_CAPACITY = 10
IN_BED_STATUSES = {"workup", "results", "assigned"}


def _is_esi1(p) -> bool:
    return bool(p.triage_score and p.triage_score.esi_level == 1)


async def auto_fill_beds() -> dict:
    """Two-step tick:
      1. Promote highest-risk waiting patients to treatment beds until full.
      2. Assign a truly free doctor to any bedded patient still without one.
    Returns counts for logging.
    """
    from src.orchestrator.queue import get_queue
    from src.agents.load_balancer import get_load_balancer

    queue = get_queue()
    lb = get_load_balancer()

    sorted_q = await queue.get_sorted_queue()

    bedded = [p for p in sorted_q if p.status in IN_BED_STATUSES and not _is_esi1(p)]
    free_beds = max(0, BED_CAPACITY - len(bedded))

    waiting = [p for p in sorted_q if p.status == "waiting" and not _is_esi1(p)]

    admitted = 0
    for p in waiting[:free_beds]:
        if await queue.admit_to_bed(p.intake.patient_id):
            admitted += 1

    # Pass 2 — assign free doctors to doctor-less bedded patients
    sorted_q = await queue.get_sorted_queue()
    assigned = 0
    for p in sorted_q:
        if p.status not in IN_BED_STATUSES or _is_esi1(p):
            continue
        if p.assigned_doctor:
            continue
        free_doctors = [d for d in lb.get_all_doctors() if not d.active_cases]
        if not free_doctors:
            break  # no point continuing; nothing more to assign
        esi = p.triage_score.esi_level if p.triage_score else 3
        rs = p.triage_score.risk_score if p.triage_score else 0.0
        result = lb.assign_patient(p.intake.patient_id, esi_level=esi, risk_score=rs)
        if result.assigned_doctor:
            await queue.assign_doctor(p.intake.patient_id, result.assigned_doctor.name)
            assigned += 1
            # Bumps shouldn't happen here (non-ESI-1 doesn't bump), but be safe
            if result.bumped_patient_id:
                await queue.unassign_doctor(result.bumped_patient_id)

    if admitted or assigned:
        logger.info("bed tick | admitted=%d assigned=%d (beds=%d/%d)",
                    admitted, assigned, len(bedded) + admitted, BED_CAPACITY)
    return {"admitted": admitted, "assigned_doctors": assigned}
