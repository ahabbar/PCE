from __future__ import annotations
import asyncio, logging, time
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

from src.core.patient import IntakeForm, TriageScoreResult, RedFlagResult, WorkupPlan, PatientRecord
from src.core.esi_algorithm import run_esi_algorithm, ESIResult
from src.agents.llm import get_llm_client
from src.agents.triage_score import run_triage_score, get_dynamic_threshold
from src.agents.red_flag import run_red_flag_guardian
from src.agents.workup import run_preemptive_workup
from src.core.whitelist import ProtocolWhitelist
from src.database.db import save_patient, save_investigation_orders, log_agent_action

logger = logging.getLogger("pce.orchestrator")


@dataclass
class TriageResult:
    patient_id: str
    intake: IntakeForm
    esi_result: ESIResult
    triage_score: TriageScoreResult
    red_flag: RedFlagResult
    workup: WorkupPlan | None
    total_latency_ms: float
    is_red: bool
    threshold_used: float
    parallel_confirmed: bool


async def process_patient(
    intake: IntakeForm,
    patient_load: int = 10,
    run_workup: bool = True,
) -> TriageResult:
    t0 = time.monotonic()

    # Step 1: ESI (deterministic)
    esi_result = run_esi_algorithm(intake, estimated_resource_count=2)

    # Step 2: ESI-1 fast path — no LLM at all
    if esi_result.esi_level == 1:
        logger.warning(
            "ESI-1 patient %s — routing to Resus Bay, bypassing all agents",
            intake.patient_id[:8],
        )
        safe_score = TriageScoreResult(
            risk_score=95.0,
            esi_level=1,
            confidence=1.0,
            is_red=True,
            threshold_used=get_dynamic_threshold(patient_load),
            key_factors=["ESI-1 immediate"],
            reasoning="ESI-1: immediate life threat",
            pce_scope="excluded_resus",
        )
        safe_flag = RedFlagResult(
            is_emergency=True,
            esi1_immediate=True,
            layer_triggered="rules",
            reasoning="ESI-1 bypass",
        )
        return TriageResult(
            patient_id=intake.patient_id,
            intake=intake,
            esi_result=esi_result,
            triage_score=safe_score,
            red_flag=safe_flag,
            workup=None,
            total_latency_ms=(time.monotonic() - t0) * 1000,
            is_red=True,
            threshold_used=safe_score.threshold_used,
            parallel_confirmed=False,
        )

    # Step 3: Agents 1 + 2 in parallel
    llm = get_llm_client()

    score_task = run_triage_score(intake, esi_result, llm, patient_load)
    flag_task = run_red_flag_guardian(intake, llm)

    results = await asyncio.gather(score_task, flag_task, return_exceptions=True)
    parallel_confirmed = True

    triage_score_raw = results[0]
    red_flag = results[1]

    # Handle exceptions with safe defaults
    if isinstance(triage_score_raw, Exception):
        logger.error("Agent 1 failed: %s", triage_score_raw)
        triage_score_raw = TriageScoreResult(
            risk_score=50.0,
            esi_level=esi_result.esi_level,
            confidence=0.5,
            is_red=False,
            threshold_used=get_dynamic_threshold(patient_load),
            key_factors=["agent error"],
            reasoning="Agent 1 failed",
            pce_scope=esi_result.pce_scope,
        )
    if isinstance(red_flag, Exception):
        logger.error("Agent 2 failed: %s", red_flag)
        red_flag = RedFlagResult(
            is_emergency=False,
            layer_triggered="rules",
            reasoning="Agent 2 failed",
        )

    # Step 4: Set is_red — orchestrator, never LLM
    threshold = get_dynamic_threshold(patient_load)
    is_red = (triage_score_raw.risk_score >= threshold) or red_flag.is_emergency

    # Build final score with correct is_red
    triage_score = TriageScoreResult(
        risk_score=triage_score_raw.risk_score,
        esi_level=triage_score_raw.esi_level,
        confidence=triage_score_raw.confidence,
        is_red=is_red,
        threshold_used=threshold,
        key_factors=triage_score_raw.key_factors,
        reasoning=triage_score_raw.reasoning,
        pce_scope=triage_score_raw.pce_scope,
    )

    # Step 5: Red flag ESI-1 override — skip workup
    workup = None
    if red_flag.esi1_immediate:
        logger.warning(
            "RED FLAG ESI-1 | patient=%s flag=%s action=%s",
            intake.patient_id[:8],
            red_flag.flag_type,
            red_flag.immediate_action,
        )
    elif run_workup:
        # Step 6: Agent 3 workup
        whitelist = ProtocolWhitelist()
        workup = await run_preemptive_workup(intake, triage_score, llm, whitelist)

    total_latency_ms = (time.monotonic() - t0) * 1000

    # Step 7: Persist to DB (best-effort, don't let DB failure crash triage)
    try:
        tasks = [save_patient(intake, esi_result, triage_score)]
        if workup:
            tasks.append(save_investigation_orders(intake.patient_id, workup))
        tasks.append(
            log_agent_action(
                intake.patient_id, 1, "triage_score",
                f"esi={esi_result.esi_level}",
                f"score={triage_score.risk_score}",
                total_latency_ms, "gemini",
            )
        )
        tasks.append(
            log_agent_action(
                intake.patient_id, 2, "red_flag",
                "intake",
                f"emergency={red_flag.is_emergency}",
                total_latency_ms, "gemini",
            )
        )
        if workup:
            tasks.append(
                log_agent_action(
                    intake.patient_id, 3, "workup",
                    f"protocol={workup.protocol_key}",
                    f"orders={len(workup.orders)}",
                    total_latency_ms, "gemini",
                )
            )
        await asyncio.gather(*tasks)
    except Exception as exc:
        logger.error("DB persist failed: %s", exc)

    logger.info(
        "process_patient done | patient=%s score=%.1f is_red=%s latency=%.0fms",
        intake.patient_id[:8], triage_score.risk_score, is_red, total_latency_ms,
    )

    return TriageResult(
        patient_id=intake.patient_id,
        intake=intake,
        esi_result=esi_result,
        triage_score=triage_score,
        red_flag=red_flag,
        workup=workup,
        total_latency_ms=total_latency_ms,
        is_red=is_red,
        threshold_used=threshold,
        parallel_confirmed=parallel_confirmed,
    )
