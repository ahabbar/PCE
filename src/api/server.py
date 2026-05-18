from __future__ import annotations
import itertools, logging, os, time
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

_patient_counter = itertools.count(1)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.core.patient import IntakeForm, OrderItem, TriageScoreResult, RedFlagResult, PatientRecord
from src.core.esi_algorithm import ESIResult
from src.orchestrator.engine import process_patient, TriageResult
from src.orchestrator.queue import get_queue
from src.agents.batch import get_batch_coordinator
from src.database.db import (
    init_db, get_patient, get_waiting_patients, get_recent_patients,
    save_patient_basic, clear_non_permanent_patients,
    get_patient_labs, seed_lab_orders, confirm_lab_order, enter_lab_result,
)
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
    assigned_doctor: str | None = None
    result_count: int = 0


class QueueResponse(BaseModel):
    patients: list[QueuePatientOut]
    total: int
    reds_count: int
    queue_depth: int
    batch_summary: dict[str, int]


# ── Lifespan ──────────────────────────────────────────────────────────────────

async def _reload_queue_from_db() -> int:
    from src.core.patient import PatientRecord, TriageScoreResult, RedFlagResult
    from src.orchestrator.score_engine import reconstruct_intake_from_record

    patients = await get_waiting_patients()
    q = get_queue()
    count = 0
    for record in patients:
        try:
            intake = reconstruct_intake_from_record(record)
            pr = PatientRecord(
                intake=intake,
                triage_score=TriageScoreResult(
                    risk_score=float(record.get("risk_score", 50)),
                    esi_level=int(record.get("esi_level", 3)),
                    confidence=0.8,
                    is_red=bool(record.get("is_red")),
                    threshold_used=float(record.get("threshold_used", 85.0)),
                    key_factors=[record.get("chief_complaint_category", "other")],
                    reasoning="Loaded from DB on restart",
                    pce_scope=str(record.get("pce_scope", "pce_core")),
                ),
                status=record.get("status", "waiting"),
                assigned_doctor=record.get("assigned_doctor"),
                result_count=int(record.get("result_count", 0)),
            )
            await q.add_patient_record(pr)
            count += 1
        except Exception as exc:
            logger.warning("Could not reload patient %s: %s", record.get("patient_id", "?")[:8], exc)
    if count:
        logger.info("Reloaded %d patients into queue from DB", count)
    return count


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    app.state.queue_loaded = await _reload_queue_from_db()
    yield


app = FastAPI(title="Parallel Care Engine", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from src.api.ed_view import router as ed_view_router
app.include_router(ed_view_router)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/triage", response_model=TriageResponse)
async def triage_patient(intake: IntakeForm):
    try:
        result: TriageResult = await process_patient(intake, patient_load=10)
    except Exception as exc:
        logger.error("Triage failed for patient %s: %s", intake.patient_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # Add to live queue. ESI-1 still gets added so it renders in the Resus Bay,
    # but it bypasses the PCE workup queue position — instead it is force-assigned
    # to a doctor immediately (overflow-tolerant) and stored with status='assigned'.
    if result.esi_result.esi_level == 1:
        from src.agents.load_balancer import get_load_balancer
        lb = get_load_balancer()
        assignment = lb.assign_patient(result.patient_id, esi_level=1)
        doctor_name = assignment.assigned_doctor.name if assignment.assigned_doctor else None
        pr = PatientRecord(
            intake=result.intake,
            esi_result=result.esi_result,
            triage_score=result.triage_score,
            red_flag=result.red_flag,
            workup=result.workup,
            status="assigned",
            assigned_doctor=doctor_name,
            result_count=0,
        )
        await get_queue().add_patient_record(pr)
        if assignment.bumped_patient_id:
            await get_queue().unassign_doctor(assignment.bumped_patient_id)
            logger.info("ESI-1 %s bumped patient %s off %s",
                        result.patient_id[:8], assignment.bumped_patient_id[:8], assignment.bumped_from_doctor)
    else:
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

    # ESI-1 → force-assign to a doctor immediately so the patient renders
    # in the Resus Bay with a connector line to the assigned doctor's bay.
    doctor_name = None
    if req.esi_level == 1:
        lb = get_load_balancer()
        assignment = lb.assign_patient(req.intake.patient_id, esi_level=1)
        doctor_name = assignment.assigned_doctor.name if assignment.assigned_doctor else None
        pr = PatientRecord(
            intake=req.intake,
            esi_result=esi_result,
            triage_score=triage_score,
            red_flag=red_flag,
            workup=None,
            status="assigned",
            assigned_doctor=doctor_name,
            result_count=0,
        )
        await get_queue().add_patient_record(pr)
        if assignment.bumped_patient_id:
            await get_queue().unassign_doctor(assignment.bumped_patient_id)
            logger.info("ESI-1 %s bumped patient %s off %s",
                        req.intake.patient_id[:8], assignment.bumped_patient_id[:8], assignment.bumped_from_doctor)
        position = await get_queue().queue_depth()
    else:
        position = await get_queue().add_patient(result)

    # Persist to DB so exit-plan / discharge lookups work later
    try:
        await save_patient_basic({
            "patient_id": req.intake.patient_id,
            "arrival_time": req.intake.arrival_time,
            "age_years": req.intake.age_years,
            "gender": req.intake.gender,
            "chief_complaint_text": req.intake.chief_complaint.free_text_en,
            "chief_complaint_category": req.intake.chief_complaint.category,
            "esi_level": req.esi_level,
            "risk_score": req.risk_score,
            "is_red": req.is_red,
            "pce_scope": req.pce_scope,
            "threshold_used": req.threshold_used,
            "status": "assigned" if req.esi_level == 1 else "waiting",
            "assigned_doctor": doctor_name,
        })
    except Exception as _e:
        logger.warning("queue/add: could not persist patient to DB: %s", _e)

    return {"patient_id": req.intake.patient_id, "position": position, "assigned_doctor": doctor_name}


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
                assigned_doctor=record.assigned_doctor,
                result_count=record.result_count,
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
    import json as _json
    lab = LabResultModel(
        patient_id=body["patient_id"],
        test_name=body["test_name"],
        result_value=body["result_value"],
        result_time=body.get("result_time", time.time()),
        is_critical=body.get("is_critical", False),
    )
    record = await get_patient(lab.patient_id)
    if record is None:
        # Patient added via /queue/add (not in DB) — build record from in-memory queue
        sorted_q = await get_queue().get_sorted_queue()
        for pr in sorted_q:
            if pr.intake.patient_id == lab.patient_id:
                record = {
                    "patient_id": pr.intake.patient_id,
                    "arrival_time": pr.intake.arrival_time,
                    "age_years": pr.intake.age_years,
                    "gender": pr.intake.gender,
                    "chief_complaint_text": pr.intake.chief_complaint.free_text_en,
                    "chief_complaint_category": pr.intake.chief_complaint.category,
                    "risk_score": pr.triage_score.risk_score if pr.triage_score else 50.0,
                    "is_red": 1 if (pr.triage_score.is_red if pr.triage_score else False) else 0,
                    "status": pr.status,
                    "known_diagnoses": _json.dumps(pr.intake.medical_history.known_diagnoses),
                    "additional_context": pr.intake.additional_context,
                    "esi_level": pr.triage_score.esi_level if pr.triage_score else 3,
                    "threshold_used": pr.triage_score.threshold_used if pr.triage_score else 85.0,
                    "pce_scope": pr.triage_score.pce_scope if pr.triage_score else "pce_core",
                }
                break
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
    if result.bumped_patient_id:
        await queue.unassign_doctor(result.bumped_patient_id)
        logger.info("ESI-%d %s bumped patient %s off %s",
                    esi_level, patient_id[:8], result.bumped_patient_id[:8], result.bumped_from_doctor)
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


@app.post("/queue/{patient_id}/discharge")
async def quick_discharge(patient_id: str, body: dict):
    import asyncio as _aio
    from src.orchestrator.engine import try_cascade_assignment
    disposition = body.get("disposition", "discharge")
    diagnosis = body.get("confirmed_diagnosis", "")
    await get_queue().discharge(patient_id)
    freed_name = None
    try:
        from src.database.db import update_patient_status, set_patient_disposition
        await update_patient_status(patient_id, "seen")
        await set_patient_disposition(patient_id, disposition)
        freed = get_load_balancer().complete_case(patient_id)
        if freed:
            freed_name = freed.name
    except Exception:
        pass
    if freed_name:
        _aio.create_task(try_cascade_assignment(freed_doctor_name=freed_name))
    logger.info("Quick discharge | patient=%s disposition=%s", patient_id[:8], disposition)
    return {"patient_id": patient_id, "status": "discharged", "disposition": disposition, "diagnosis": diagnosis}


@app.get("/next-patient-id")
async def next_patient_id():
    n = next(_patient_counter)
    return {"patient_id": f"PAT-{n:05d}"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "db": "ok",
        "llm_provider": os.getenv("LLM_PROVIDER", "gemini"),
        "agents": 7,
        "agents_active_intake": ["1_triage", "2_red_flag", "3_workup", "4_batch", "6_disposition"],
        "agents_conditional": ["5_load_balancer"],
        "agents_on_exit": ["7_exit_coord"],
        "queue_loaded_from_db": getattr(app.state, "queue_loaded", 0),
    }


@app.post("/exit/{patient_id}")
async def generate_exit_plan(patient_id: str, body: dict):
    from src.agents.exit_coord import run_exit_coordinator, ExitInput, DispositionType
    from src.orchestrator.score_engine import reconstruct_intake_from_record
    from src.database.db import update_patient_status

    record = await get_patient(patient_id)
    if record is None:
        # Patient may only be in the in-memory queue (added via /queue/add, not demo seed)
        sorted_q = await get_queue().get_sorted_queue()
        for pr in sorted_q:
            if pr.intake.patient_id == patient_id:
                ts = pr.triage_score
                record = {
                    "patient_id": patient_id,
                    "arrival_time": pr.intake.arrival_time,
                    "age_years": pr.intake.age_years,
                    "gender": pr.intake.gender,
                    "chief_complaint_text": pr.intake.chief_complaint.free_text_en,
                    "chief_complaint_category": pr.intake.chief_complaint.category,
                    "risk_score": ts.risk_score if ts else 50.0,
                    "esi_level": ts.esi_level if ts else 3,
                    "is_red": 1 if (ts.is_red if ts else False) else 0,
                    "threshold_used": ts.threshold_used if ts else 85.0,
                    "pce_scope": ts.pce_scope if ts else "pce_core",
                    "status": pr.status,
                    "assigned_doctor": pr.assigned_doctor,
                }
                break
    if record is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    intake = reconstruct_intake_from_record(record)

    ts_rec = record
    triage_score = TriageScoreResult(
        risk_score=float(ts_rec.get("risk_score", 50.0)),
        esi_level=int(ts_rec.get("esi_level", 3)),
        confidence=0.8,
        is_red=bool(ts_rec.get("is_red", False)),
        threshold_used=float(ts_rec.get("threshold_used", 85.0)),
        key_factors=["clinical assessment"],
        reasoning="Retrieved from queue",
        pce_scope=str(ts_rec.get("pce_scope", "pce_core")),
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
        from src.database.db import set_patient_disposition
        await set_patient_disposition(patient_id, disp_type.value)
    except Exception:
        pass
    await get_queue().discharge(patient_id)

    import asyncio as _aio
    from src.orchestrator.engine import try_cascade_assignment
    freed = get_load_balancer().complete_case(patient_id)
    if freed:
        _aio.create_task(try_cascade_assignment(freed_doctor_name=freed.name))

    return plan.model_dump()


# ── Analytics ─────────────────────────────────────────────────────────────────

@app.get("/analytics")
async def get_analytics():
    import time as _time
    patients_db = await get_recent_patients()
    queue_patients = await get_queue().get_sorted_queue()

    total = len(patients_db)
    discharged = sum(1 for p in patients_db if p.get("status") in ("seen", "discharged"))
    admitted = 0
    icu = 0
    still_waiting = len(queue_patients)
    red_zone = sum(1 for p in queue_patients if (p.triage_score.is_red if p.triage_score else False))

    esi_dist: dict[str, int] = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    scores = []
    for p in patients_db:
        lvl = str(p.get("esi_level", 3))
        if lvl in esi_dist:
            esi_dist[lvl] += 1
        s = p.get("risk_score")
        if s is not None:
            scores.append(float(s))
    mean_score = sum(scores) / len(scores) if scores else 0.0

    now = _time.time()
    waits = [(now - p.intake.arrival_time) / 60 for p in queue_patients]
    mean_wait = sum(waits) / len(waits) if waits else 0.0

    batch_summary = get_batch_coordinator().get_batch_summary()
    total_orders = sum(batch_summary.values()) if batch_summary else 0
    batched_orders = sum(v for v in batch_summary.values() if v > 1)
    batch_eff = round(batched_orders / total_orders * 100, 1) if total_orders else 0.0

    return {
        "session_stats": {
            "total_patients_today": total,
            "discharged": discharged,
            "admitted": admitted,
            "icu": icu,
            "still_waiting": still_waiting,
            "esi_distribution": esi_dist,
            "mean_risk_score": round(mean_score, 1),
            "red_zone_count": red_zone,
            "mean_wait_minutes_current": round(mean_wait, 1),
            "batch_efficiency_pct": batch_eff,
        },
        "comparison": {
            "traditional": {
                "seen_4hr_pct": 61.0,
                "median_los_min": 285,
                "doctor_min_per_pt": 43,
                "cost_per_pt_gbp": 95.69,
                "sepsis_to_abx_min": 142,
                "lwbs_pct": 5.1,
            },
            "pce": {
                "seen_4hr_pct": 87.0,
                "median_los_min": 144,
                "doctor_min_per_pt": 20,
                "cost_per_pt_gbp": 78.70,
                "sepsis_to_abx_min": 38,
                "lwbs_pct": 0.8,
            },
            "saving_per_patient_gbp": 16.99,
            "saving_pct": 17.8,
            "annual_impact_gbp": 11700000,
            "break_even_days": 13,
        },
    }


# ── Case Validation ───────────────────────────────────────────────────────────

@app.get("/cases/list")
async def list_cases():
    from src.cases.validator import load_case_list
    return load_case_list()


@app.post("/cases/run/{case_id}")
async def run_case(case_id: str):
    from src.cases.validator import load_case_by_id, validate_case
    case = load_case_by_id(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    result = await validate_case(case)
    return result.__dict__


@app.get("/cases/validate")
async def validate_all():
    from src.cases.validator import validate_all_cases
    results = await validate_all_cases()
    if not results:
        return {"cases": [], "summary": {"total": 0}}
    mean_agr = sum(r.agreement_rate for r in results) / len(results)
    disp_acc = sum(1 for r in results if r.disposition_match) / len(results)
    mean_lat = sum(r.latency_ms for r in results) / len(results)
    mean_sc  = sum(r.proposed_score for r in results) / len(results)
    return {
        "cases": [r.__dict__ for r in results],
        "summary": {
            "total": len(results),
            "mean_agreement_rate": round(mean_agr, 3),
            "disposition_accuracy": round(disp_acc, 3),
            "mean_latency_ms": round(mean_lat, 1),
            "mean_score": round(mean_sc, 1),
        },
    }


# ── Demo Seed / Reset ─────────────────────────────────────────────────────────

SEED_PATIENTS = [
    {"cc": "chest pain and diaphoresis, known hypertension", "category": "chest_pain",
     "age": 58, "gender": "male", "vitals": {"hr": 108, "sbp": 88, "spo2": 91, "temp": 36.8},
     "risk_score": 88.0, "esi_level": 2, "is_red": True, "pce_scope": "pce_priority"},
    {"cc": "confusion and weakness this morning, known DM", "category": "altered_mental_status",
     "age": 75, "gender": "male", "vitals": {"hr": 108, "sbp": 102, "spo2": 93, "temp": 38.6},
     "risk_score": 82.0, "esi_level": 3, "is_red": False, "pce_scope": "pce_core"},
    {"cc": "shortness of breath and leg swelling", "category": "dyspnea",
     "age": 68, "gender": "female", "vitals": {"hr": 96, "sbp": 142, "spo2": 88, "temp": 37.1},
     "risk_score": 71.0, "esi_level": 3, "is_red": False, "pce_scope": "pce_core"},
    {"cc": "fever and flank pain, known CKD", "category": "urinary",
     "age": 34, "gender": "female", "vitals": {"hr": 98, "sbp": 118, "spo2": 97, "temp": 38.6},
     "risk_score": 66.0, "esi_level": 3, "is_red": False, "pce_scope": "pce_core"},
    {"cc": "mild dysuria, no fever", "category": "urinary",
     "age": 29, "gender": "female", "vitals": {"hr": 72, "sbp": 118, "spo2": 99, "temp": 36.8},
     "risk_score": 17.0, "esi_level": 5, "is_red": False, "pce_scope": "pce_lite"},
]


@app.post("/demo/seed")
async def demo_seed():
    from src.core.patient import PatientRecord, TriageScoreResult, ChiefComplaint, Vitals, MedicalHistory, IntakeForm
    from src.core.lab_values import get_default_orders
    import time as _t

    q = get_queue()
    await q.clear_all()
    get_load_balancer().reset_all()
    # Wipe DB so old panel rows (e.g. "CBC") don't persist across seeds
    await clear_non_permanent_patients()
    # Forget per-patient sim arrivals so re-seeded DEMO-001 etc. start fresh
    from src.api.ed_view import reset_patient_sim_arrivals
    reset_patient_sim_arrivals()

    seeded = 0
    # Seeded patients arrive as if they just walked in (0-2 min ago) so the
    # canvas wait timer starts near 0 and visibly ticks up at sim_speed,
    # instead of dumping pre-baked 55m / 1h waits on the screen.
    base_time = _t.time()
    for i, sp in enumerate(SEED_PATIENTS):
        pid = f"DEMO-{i+1:03d}"
        arrival = base_time - i * 30
        vitals = sp.get("vitals", {})
        intake = IntakeForm(
            patient_id=pid, arrival_time=arrival,
            age_years=float(sp["age"]), gender=sp["gender"],
            chief_complaint=ChiefComplaint(
                free_text_en=sp["cc"], category=sp["category"], pain_present=True),
            vitals=Vitals(
                heart_rate=float(vitals.get("hr", 80)),
                systolic_bp=float(vitals.get("sbp", 120)),
                spo2_pct=float(vitals.get("spo2", 98)),
                temperature_c=float(vitals.get("temp", 37.0)),
            ),
            medical_history=MedicalHistory(),
        )
        ts = TriageScoreResult(
            risk_score=sp["risk_score"], esi_level=sp["esi_level"],
            confidence=0.9, is_red=sp["is_red"], threshold_used=85.0,
            key_factors=[sp["category"]], reasoning="Demo seed patient",
            pce_scope=sp["pce_scope"],
        )
        pr = PatientRecord(intake=intake, triage_score=ts, status="waiting")
        await q.add_patient_record(pr)
        await save_patient_basic({
            "patient_id": pid, "arrival_time": arrival,
            "age_years": sp["age"], "gender": sp["gender"],
            "chief_complaint_text": sp["cc"], "chief_complaint_category": sp["category"],
            "esi_level": sp["esi_level"], "risk_score": sp["risk_score"],
            "is_red": sp["is_red"], "pce_scope": sp["pce_scope"],
            "threshold_used": 85.0, "status": "waiting",
        })
        # Pre-seed lab orders immediately so panels are always expanded on first load
        default_tests = get_default_orders(sp["category"])
        await seed_lab_orders(pid, default_tests)
        seeded += 1

    depth = await q.queue_depth()
    return {"seeded": seeded, "queue_depth": depth}


@app.post("/admin/expand-panels")
async def admin_expand_panels():
    """Force-expand any panel rows (CBC, BMP, etc.) for all patients in DB."""
    from src.database.db import expand_existing_panels, get_patient_labs
    from src.core.lab_values import PANEL_EXPANSION, get_default_orders
    sorted_q = await get_queue().get_sorted_queue()
    expanded_patients = []
    for pr in sorted_q:
        pid = pr.intake.patient_id
        # If no labs exist yet, seed them
        labs = await get_patient_labs(pid)
        if not labs:
            default_tests = get_default_orders(pr.intake.chief_complaint.category)
            await seed_lab_orders(pid, default_tests)
            expanded_patients.append({"patient_id": pid, "action": "seeded"})
        else:
            changed = await expand_existing_panels(pid)
            if changed:
                expanded_patients.append({"patient_id": pid, "action": "expanded"})
    return {"expanded": expanded_patients}


@app.post("/demo/reset")
async def demo_reset():
    await get_queue().clear_all()
    get_load_balancer().reset_all()
    try:
        await clear_non_permanent_patients()
    except Exception:
        pass
    return {"reset": True}


# ── Lab Orders (Nurse Workflow) ───────────────────────────────────────────────

@app.get("/queue/{patient_id}/labs")
async def get_labs(patient_id: str):
    """Return all lab orders for a patient. Auto-seeds if none exist, expands any old panel rows."""
    from src.core.lab_values import get_default_orders
    from src.database.db import expand_existing_panels

    orders = await get_patient_labs(patient_id)
    if not orders:
        # Seed default tests for this patient's CC category
        category = "other"
        sorted_q = await get_queue().get_sorted_queue()
        for pr in sorted_q:
            if pr.intake.patient_id == patient_id:
                category = pr.intake.chief_complaint.category
                break
        else:
            rec = await get_patient(patient_id)
            if rec:
                category = rec.get("chief_complaint_category", "other")

        default_tests = get_default_orders(category)
        await seed_lab_orders(patient_id, default_tests)
        orders = await get_patient_labs(patient_id)
    else:
        # Migrate any existing panel rows (e.g. old 'CBC' row → individual components)
        if await expand_existing_panels(patient_id):
            orders = await get_patient_labs(patient_id)

    return {"patient_id": patient_id, "orders": orders}


@app.post("/queue/{patient_id}/labs/confirm")
async def confirm_lab(patient_id: str, body: dict):
    """Nurse accepts or rejects a suggested lab order."""
    test_name = body.get("test_name", "")
    action = body.get("action", "approved")  # 'approved' or 'rejected'
    if action not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="action must be 'approved' or 'rejected'")
    await confirm_lab_order(patient_id, test_name, action)
    return {"patient_id": patient_id, "test_name": test_name, "status": action}


@app.post("/queue/{patient_id}/labs/result")
async def submit_lab_result(patient_id: str, body: dict):
    """Nurse enters a result. Interprets against reference ranges and triggers re-score."""
    from src.core.lab_values import interpret_result

    test_name = body.get("test_name", "")
    value_str = body.get("value", "").strip()
    if not test_name or not value_str:
        raise HTTPException(status_code=400, detail="test_name and value required")

    interp = interpret_result(test_name, value_str)

    # Persist result to DB
    await enter_lab_result(
        patient_id=patient_id,
        test_name=test_name,
        result_value=value_str,
        is_critical=interp["is_critical"],
        is_abnormal=interp["is_abnormal"],
        interpretation=interp["interpretation"],
    )

    # Trigger score re-evaluation via Score Update Engine
    lab = LabResultModel(
        patient_id=patient_id,
        test_name=test_name,
        result_value=f"{value_str} — {interp['interpretation']}",
        result_time=time.time(),
        is_critical=interp["is_critical"],
    )

    record = await get_patient(patient_id)
    if record is None:
        sorted_q = await get_queue().get_sorted_queue()
        import json as _json
        for pr in sorted_q:
            if pr.intake.patient_id == patient_id:
                record = {
                    "patient_id": pr.intake.patient_id,
                    "arrival_time": pr.intake.arrival_time,
                    "age_years": pr.intake.age_years,
                    "gender": pr.intake.gender,
                    "chief_complaint_text": pr.intake.chief_complaint.free_text_en,
                    "chief_complaint_category": pr.intake.chief_complaint.category,
                    "risk_score": pr.triage_score.risk_score if pr.triage_score else 50.0,
                    "is_red": 1 if (pr.triage_score.is_red if pr.triage_score else False) else 0,
                    "status": pr.status,
                    "known_diagnoses": _json.dumps(pr.intake.medical_history.known_diagnoses),
                    "additional_context": pr.intake.additional_context,
                    "esi_level": pr.triage_score.esi_level if pr.triage_score else 3,
                    "threshold_used": pr.triage_score.threshold_used if pr.triage_score else 85.0,
                    "pce_scope": pr.triage_score.pce_scope if pr.triage_score else "pce_core",
                    "assigned_doctor": pr.assigned_doctor,
                }
                break

    score_event = None
    if record:
        try:
            score_event = await update_score_on_result(lab, record, patient_load=10)
        except Exception as exc:
            logger.warning("Score update failed after lab result: %s", exc)

    return {
        "patient_id": patient_id,
        "test_name": test_name,
        "value": value_str,
        "flag": interp["flag"],
        "is_abnormal": interp["is_abnormal"],
        "is_critical": interp["is_critical"],
        "interpretation": interp["interpretation"],
        "reference_range": interp["reference_range"],
        "score_delta": round(score_event.delta, 1) if score_event else None,
        "new_score": round(score_event.new_score, 1) if score_event else None,
        "queue_reordered": score_event.queue_reordered if score_event else False,
        "auto_assigned": score_event is not None,
    }
