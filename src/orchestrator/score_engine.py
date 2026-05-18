from __future__ import annotations
import asyncio, json, logging, time
from dataclasses import dataclass
from src.core.patient import IntakeForm, Vitals, ChiefComplaint, MedicalHistory
from src.agents.llm import get_llm_client
from src.agents.triage_score import run_triage_score, get_dynamic_threshold
from src.core.esi_algorithm import run_esi_algorithm
from src.orchestrator.queue import get_queue
from src.database.db import log_agent_action
from src.agents.load_balancer import get_load_balancer

logger = logging.getLogger("pce.score_engine")


@dataclass
class LabResult:
    patient_id: str
    test_name: str
    result_value: str
    result_time: float
    is_critical: bool


@dataclass
class ScoreUpdateEvent:
    patient_id: str
    test_name: str
    result_value: str
    old_score: float
    new_score: float
    delta: float
    threshold_used: float
    is_red_before: bool
    is_red_after: bool
    threshold_crossed: bool
    queue_reordered: bool
    latency_ms: float


def should_auto_assign(result: LabResult, result_count: int, already_assigned: bool) -> bool:
    """Returns True when Agent 5 should fire autonomously after a lab result."""
    if already_assigned:
        return False
    if result.is_critical:
        return True
    if result_count >= 2:
        return True
    return False


def reconstruct_intake_from_record(record: dict) -> IntakeForm:
    existing_context = record.get("additional_context", "") or ""
    return IntakeForm(
        patient_id=record["patient_id"],
        arrival_time=record.get("arrival_time", time.time()),
        age_years=record.get("age_years", 50),
        gender=record.get("gender", "unknown"),
        chief_complaint=ChiefComplaint(
            free_text_en=record.get("chief_complaint_text", ""),
            category=record.get("chief_complaint_category", "other"),
            pain_present=False,
        ),
        vitals=Vitals(),
        additional_context=existing_context,
        medical_history=MedicalHistory(
            known_diagnoses=json.loads(record.get("known_diagnoses", "[]")),
        ),
    )


async def update_score_on_result(
    result: LabResult,
    patient_record: dict,
    patient_load: int = 10,
) -> ScoreUpdateEvent | None:
    if patient_record is None:
        return None
    if patient_record.get("status") in ("seen", "discharged"):
        return None

    # ── Step 1 (FIRST, never skipped): Increment result count + auto-assign ──
    # Done BEFORE the LLM rescore so a slow / failing LLM can never block
    # Agent 5 from assigning a doctor.
    queue = get_queue()
    new_result_count, current_doctor = await queue.increment_result_count(result.patient_id)
    already_assigned = bool(current_doctor or patient_record.get("assigned_doctor"))

    if should_auto_assign(result, new_result_count, already_assigned):
        lb = get_load_balancer()
        esi = int(patient_record.get("esi_level", 3))
        assignment = lb.assign_patient(result.patient_id, esi_level=esi)
        if assignment.assigned_doctor:
            await queue.assign_doctor(result.patient_id, assignment.assigned_doctor.name)
            if assignment.bumped_patient_id:
                await queue.unassign_doctor(assignment.bumped_patient_id)
            try:
                await log_agent_action(
                    patient_id=result.patient_id, agent_id=5,
                    action="auto_assign_on_result",
                    inputs_summary=f"ESI-{esi} | count={new_result_count} | critical={result.is_critical} | test={result.test_name}",
                    outputs_summary=f"→ {assignment.assigned_doctor.name} ({assignment.assigned_doctor.role.value})",
                    latency_ms=0, model_used="rules",
                )
            except Exception:
                pass
            logger.info(
                "Agent 5 auto-assigned: patient %s → %s (triggered by %s)",
                result.patient_id[:8], assignment.assigned_doctor.name, result.test_name,
            )

    # ── Step 2: LLM rescore (best-effort — failures must not block assignment) ─
    old_score = float(patient_record.get("risk_score", 50.0))
    new_score = old_score
    delta = 0.0
    threshold = get_dynamic_threshold(patient_load)
    is_red_after = bool(patient_record.get("is_red", False))
    is_red_before = is_red_after
    threshold_crossed = False
    queue_reordered = False
    latency_ms = 0.0

    try:
        existing_context = patient_record.get("additional_context", "") or ""
        new_lab_text = f"Lab results received: {result.test_name}: {result.result_value}"
        updated_context = f"{existing_context}. {new_lab_text}".strip(". ")

        intake = IntakeForm(
            patient_id=patient_record["patient_id"],
            arrival_time=patient_record.get("arrival_time", time.time()),
            age_years=patient_record.get("age_years", 50),
            gender=patient_record.get("gender", "unknown"),
            chief_complaint=ChiefComplaint(
                free_text_en=patient_record.get("chief_complaint_text", ""),
                category=patient_record.get("chief_complaint_category", "other"),
                pain_present=False,
            ),
            vitals=Vitals(),
            additional_context=updated_context,
            medical_history=MedicalHistory(
                known_diagnoses=json.loads(patient_record.get("known_diagnoses", "[]")),
            ),
        )

        t0 = time.monotonic()
        llm = get_llm_client()
        esi_result = run_esi_algorithm(intake)
        new_score_result = await run_triage_score(intake, esi_result, llm, patient_load)
        latency_ms = (time.monotonic() - t0) * 1000

        new_score = new_score_result.risk_score
        delta = new_score - old_score
        is_red_after = new_score >= threshold
        threshold_crossed = is_red_after and not is_red_before

        queue_reordered = await get_queue().update_score(
            result.patient_id, new_score, is_red_after
        )

        try:
            await log_agent_action(
                patient_id=result.patient_id, agent_id=1,
                action="score_update_on_result",
                inputs_summary=f"{result.test_name}: {result.result_value}",
                outputs_summary=f"{old_score:.0f}% → {new_score:.0f}% (Δ{delta:+.0f})",
                latency_ms=latency_ms, model_used="gemini-2.5-flash",
            )
            if threshold_crossed:
                await log_agent_action(
                    patient_id=result.patient_id, agent_id=2,
                    action="red_threshold_crossed",
                    inputs_summary=f"score={new_score:.0f}%, threshold={threshold}%",
                    outputs_summary="Queue re-sorted — patient moved to RED zone",
                    latency_ms=0, model_used="rules",
                )
        except Exception as exc:
            logger.warning("DB log failed: %s", exc)
    except Exception as exc:
        logger.warning("LLM rescore failed (assignment unaffected): %s", exc)

    return ScoreUpdateEvent(
        patient_id=result.patient_id,
        test_name=result.test_name,
        result_value=result.result_value,
        old_score=old_score,
        new_score=new_score,
        delta=delta,
        threshold_used=threshold,
        is_red_before=is_red_before,
        is_red_after=is_red_after,
        threshold_crossed=threshold_crossed,
        queue_reordered=queue_reordered,
        latency_ms=latency_ms,
    )
