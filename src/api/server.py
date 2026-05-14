from __future__ import annotations
import logging, os, time
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.core.patient import IntakeForm, OrderItem, TriageScoreResult, RedFlagResult
from src.core.esi_algorithm import ESIResult
from src.orchestrator.engine import process_patient, TriageResult
from src.orchestrator.queue import get_queue
from src.agents.batch import get_batch_coordinator
from src.database.db import init_db, get_patient
from src.orchestrator.score_engine import update_score_on_result, LabResult as LabResultModel
from src.agents.load_balancer import get_load_balancer
from src.agents.llm import get_llm_client

logger = logging.getLogger("pce.api")


# ── Response models ───────────────────────────────────────────────────────────

class WorkupOrderOut(BaseModel):
    test_name: str
    reason: str
    cost_tier: str
    timing: str


class TriageResponse(BaseModel):
    patient_id: str
    esi_level: int
    esi_scope: str
    risk_score: float
    is_red: bool
    threshold_used: float
    key_factors: list[str]
    reasoning: str
    is_emergency: bool
    flag_type: str | None
    immediate_action: str | None
    workup_orders: list[WorkupOrderOut]
    deferred_to_doctor: list[str]
    parallel_confirmed: bool
    total_latency_ms: float


class QueuePatientOut(BaseModel):
    patient_id: str
    position: int
    risk_score: float
    is_red: bool
    esi_level: int
    chief_complaint: str
    age: float
    gender: str
    wait_minutes: float
    status: str


class QueueResponse(BaseModel):
    patients: list[QueuePatientOut]
    total: int
    reds_count: int
    queue_depth: int
    batch_summary: dict[str, int]


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="Parallel Care Engine", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/triage", response_model=TriageResponse)
async def triage_patient(intake: IntakeForm):
    try:
        result: TriageResult = await process_patient(intake, patient_load=10)
    except Exception as exc:
        logger.error("Triage failed for patient %s: %s", intake.patient_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # Add to live queue (skip ESI-1 — they go straight to Resus Bay)
    if result.esi_result.esi_level > 1:
        await get_queue().add_patient(result)

    logger.info(
        "POST /triage | patient=%s latency=%.0fms",
        result.patient_id[:8],
        result.total_latency_ms,
    )

    workup_orders: list[WorkupOrderOut] = []
    deferred: list[str] = []
    if result.workup:
        workup_orders = [
            WorkupOrderOut(
                test_name=o.test_name,
                reason=o.reason,
                cost_tier=o.cost_tier,
                timing=o.timing,
            )
            for o in result.workup.orders
        ]
        deferred = result.workup.deferred_to_doctor

    return TriageResponse(
        patient_id=result.patient_id,
        esi_level=result.esi_result.esi_level,
        esi_scope=result.esi_result.pce_scope,
        risk_score=result.triage_score.risk_score,
        is_red=result.is_red,
        threshold_used=result.threshold_used,
        key_factors=result.triage_score.key_factors,
        reasoning=result.triage_score.reasoning,
        is_emergency=result.red_flag.is_emergency,
        flag_type=result.red_flag.flag_type,
        immediate_action=result.red_flag.immediate_action,
        workup_orders=workup_orders,
        deferred_to_doctor=deferred,
        parallel_confirmed=result.parallel_confirmed,
        total_latency_ms=result.total_latency_ms,
    )


class DirectQueueRequest(BaseModel):
    intake: IntakeForm
    esi_level: int
    pce_scope: str
    risk_score: float
    is_red: bool
    threshold_used: float
    confidence: float
    key_factors: list[str]
    reasoning: str
    is_emergency: bool
    esi1_immediate: bool
    flag_type: str | None = None
    immediate_action: str | None = None
    layer_triggered: str = "rules"
    flag_reasoning: str = ""


@app.post("/queue/add")
async def add_to_queue_direct(req: DirectQueueRequest):
    esi_result = ESIResult(
        esi_level=req.esi_level,
        pce_scope=req.pce_scope,
        decision_point_reached="C",
        rationale="Pre-computed by dashboard",
    )
    triage_score = TriageScoreResult(
        risk_score=req.risk_score,
        esi_level=req.esi_level,
        confidence=req.confidence,
        is_red=req.is_red,
        threshold_used=req.threshold_used,
        key_factors=req.key_factors,
        reasoning=req.reasoning,
        pce_scope=req.pce_scope,
    )
    red_flag = RedFlagResult(
        is_emergency=req.is_emergency,
        esi1_immediate=req.esi1_immediate,
        flag_type=req.flag_type,
        immediate_action=req.immediate_action,
        layer_triggered=req.layer_triggered,
        reasoning=req.flag_reasoning,
    )
    result = TriageResult(
        patient_id=req.intake.patient_id,
        intake=req.intake,
        esi_result=esi_result,
        triage_score=triage_score,
        red_flag=red_flag,
        workup=None,
        total_latency_ms=0,
        is_red=req.is_red,
        threshold_used=req.threshold_used,
        parallel_confirmed=True,
    )
    position = await get_queue().add_patient(result)
    return {"patient_id": req.intake.patient_id, "position": position}


@app.get("/queue", response_model=QueueResponse)
async def get_queue_status():
    queue = get_queue()
    sorted_patients = await queue.get_sorted_queue()
    now = time.time()

    patients_out: list[QueuePatientOut] = []
    for i, record in enumerate(sorted_patients, 1):
        score = record.triage_score.risk_score if record.triage_score else 0.0
        is_red = record.triage_score.is_red if record.triage_score else False
        esi_level = record.triage_score.esi_level if record.triage_score else 0
        arrival = record.intake.arrival_time
        wait_minutes = (now - arrival) / 60.0

        patients_out.append(
            QueuePatientOut(
                patient_id=record.intake.patient_id,
                position=i,
                risk_score=score,
                is_red=is_red,
                esi_level=esi_level,
                chief_complaint=record.intake.chief_complaint.free_text_en,
                age=record.intake.age_years,
                gender=record.intake.gender,
                wait_minutes=wait_minutes,
                status=record.status,
            )
        )

    reds_count = sum(1 for p in patients_out if p.is_red)
    bc = get_batch_coordinator()
    batch_summary = bc.get_batch_summary()

    return QueueResponse(
        patients=patients_out,
        total=len(patients_out),
        reds_count=reds_count,
        queue_depth=len(patients_out),
        batch_summary=batch_summary,
    )


@app.post("/queue/{patient_id}/score")
async def update_queue_score(patient_id: str, body: dict):
    new_score = float(body.get("new_score", 0))
    queue = get_queue()
    reordered = await queue.update_score(patient_id, new_score, new_score >= 85)

    sorted_q = await queue.get_sorted_queue()
    new_position = next(
        (i + 1 for i, p in enumerate(sorted_q) if p.intake.patient_id == patient_id),
        -1,
    )
    return {"reordered": reordered, "new_position": new_position}


@app.post("/result")
async def inject_lab_result(body: dict):
    lab = LabResultModel(
        patient_id=body["patient_id"],
        test_name=body["test_name"],
        result_value=body["result_value"],
        result_time=body.get("result_time", time.time()),
        is_critical=body.get("is_critical", False),
    )
    record = await get_patient(lab.patient_id)
    if record is None:
        return {"message": "patient not found"}
    event = await update_score_on_result(lab, record, patient_load=10)
    if event is None:
        return {"message": "patient not found or discharged"}
    return {
        "patient_id": event.patient_id,
        "test_name": event.test_name,
        "old_score": event.old_score,
        "new_score": event.new_score,
        "delta": event.delta,
        "threshold_crossed": event.threshold_crossed,
        "queue_reordered": event.queue_reordered,
        "latency_ms": event.latency_ms,
    }


@app.post("/assign/{patient_id}")
async def assign_doctor(patient_id: str, body: dict):
    esi_level = int(body.get("esi_level", 3))
    lb = get_load_balancer()
    result = lb.assign_patient(patient_id, esi_level)
    queue = get_queue()
    if result.assigned_doctor:
        await queue.assign_doctor(patient_id, result.assigned_doctor.name)
    return {
        "patient_id": patient_id,
        "assigned": result.assigned_doctor is not None,
        "doctor_name": result.assigned_doctor.name if result.assigned_doctor else None,
        "doctor_role": result.assigned_doctor.role.value if result.assigned_doctor else None,
        "reason": result.reason,
        "estimated_wait_min": result.estimated_wait_min,
        "load_balanced": result.load_balanced,
    }


@app.get("/doctors")
async def get_doctor_load():
    lb = get_load_balancer()
    return {"doctors": lb.get_load_report()}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "db": "ok",
        "llm_provider": os.getenv("LLM_PROVIDER", "gemini"),
        "agents": 4,
    }


@app.post("/exit/{patient_id}")
async def generate_exit_plan(patient_id: str, body: dict):
    from src.agents.exit_coord import run_exit_coordinator, ExitInput, DispositionType
    from src.orchestrator.score_engine import reconstruct_intake_from_record
    from src.database.db import update_patient_status

    record = await get_patient(patient_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    intake = reconstruct_intake_from_record(record)

    triage_score = TriageScoreResult(
        risk_score=float(record.get("risk_score", 50.0)),
        esi_level=int(record.get("esi_level", 3)),
        confidence=0.8,
        is_red=bool(record.get("is_red", False)),
        threshold_used=float(record.get("threshold_used", 85.0)),
        key_factors=["clinical assessment"],
        reasoning="Retrieved from DB",
        pce_scope=str(record.get("pce_scope", "pce_core")),
    )

    disposition_str = body.get("disposition", "discharge")
    try:
        disp_type = DispositionType(disposition_str)
    except ValueError:
        disp_type = DispositionType.DISCHARGE

    inp = ExitInput(
        intake=intake,
        triage_score=triage_score,
        disposition=disp_type,
        confirmed_diagnosis=body.get("confirmed_diagnosis", ""),
        results_summary=[],
        assigned_doctor=body.get("doctor_name", "Unknown"),
    )

    llm = get_llm_client()
    plan = await run_exit_coordinator(inp, llm)

    try:
        await update_patient_status(patient_id, "seen")
    except Exception:
        pass
    await get_queue().discharge(patient_id)

    return plan.model_dump()
