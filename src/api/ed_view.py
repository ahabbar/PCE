from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import aiosqlite
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from src.agents.load_balancer import get_load_balancer
from src.database.db import get_recent_patients
from src.orchestrator.queue import get_queue

logger = logging.getLogger("pce.ed_view")
router = APIRouter()

_HTML_PATH = Path(__file__).parent / "ed_view.html"
_DB_PATH = os.getenv("PCE_DB_PATH", "data/pce_demo.db")

# Snapshot cadence (seconds). Lower = snappier, higher = lighter on DB.
_TICK_SECONDS = 1.5
# Agent-firings window the front-end pulses on
_AGENT_WINDOW_SEC = 5


def _patient_to_dict(p) -> dict:
    score = p.triage_score
    return {
        "id": p.intake.patient_id,
        "esi": int(score.esi_level) if score else 3,
        "status": p.status,
        "is_red": bool(score.is_red) if score else False,
        "doctor": p.assigned_doctor,
        "result_count": int(p.result_count or 0),
        "wait_min": round((time.time() - p.intake.arrival_time) / 60.0, 1),
    }


async def _agent_counts_in_window(window_sec: int) -> dict[str, int]:
    """Count audit_log entries per agent_id in the last `window_sec` seconds."""
    cutoff = time.time() - window_sec
    counts = {f"A{i}": 0 for i in range(1, 8)}
    try:
        async with aiosqlite.connect(_DB_PATH) as db:
            cursor = await db.execute(
                "SELECT agent_id, COUNT(*) FROM audit_log "
                "WHERE timestamp >= ? GROUP BY agent_id",
                (cutoff,),
            )
            rows = await cursor.fetchall()
        for agent_id, count in rows:
            if 1 <= int(agent_id) <= 7:
                counts[f"A{int(agent_id)}"] = int(count)
    except Exception as exc:
        logger.debug("agent_counts query failed: %s", exc)
    return counts


async def _lab_utilization_pct() -> int:
    """Active investigations / nominal capacity (50) → %."""
    try:
        async with aiosqlite.connect(_DB_PATH) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM investigations "
                "WHERE status IN ('approved','pending_approval')"
            )
            row = await cursor.fetchone()
        active = int(row[0]) if row else 0
        return min(100, int(active * 100 / 50))
    except Exception:
        return 0


async def _build_snapshot() -> dict:
    queue = get_queue()
    sorted_patients = await queue.get_sorted_queue()

    pts: list[dict] = []
    waiting = in_bed = with_doctor = 0
    wait_minutes: list[float] = []
    for p in sorted_patients:
        d = _patient_to_dict(p)
        pts.append(d)
        wait_minutes.append(d["wait_min"])
        if d["status"] == "waiting":
            waiting += 1
        elif d["status"] in ("workup", "results"):
            in_bed += 1
        elif d["status"] == "assigned":
            with_doctor += 1

    try:
        db_pts = await get_recent_patients()
        processed = sum(
            1 for r in db_pts if r.get("status") in ("seen", "discharged")
        )
    except Exception:
        processed = 0

    agents = await _agent_counts_in_window(_AGENT_WINDOW_SEC)
    lab_util = await _lab_utilization_pct()
    avg_wait = (
        round(sum(wait_minutes) / len(wait_minutes), 1) if wait_minutes else 0.0
    )

    lb = get_load_balancer()
    doctors_out: list[dict] = []
    try:
        for d in lb._doctors:  # type: ignore[attr-defined]
            doctors_out.append(
                {
                    "id": d.doctor_id,
                    "name": d.name,
                    "role": d.role.value,
                    "active": len(d.active_cases),
                    "max": d.max_cases,
                    "load_pct": round(d.load_pct, 1),
                    "patients": [c.patient_id for c in d.active_cases],
                }
            )
    except Exception as exc:
        logger.debug("doctor report failed: %s", exc)

    return {
        "patients": pts,
        "doctors": doctors_out,
        "agents": agents,
        "metrics": {
            "in_ed": len(pts),
            "waiting": waiting,
            "in_bed": in_bed,
            "with_doctor": with_doctor,
            "processed_total": processed,
            "avg_wait_pce_min": avg_wait,
            "avg_wait_trad_min": 85,
            "lab_util_pct": lab_util,
        },
        "ts": time.time(),
    }


@router.get("/ed-snapshot")
async def ed_snapshot():
    """One-shot JSON snapshot — used by the dashboard for server-side seeding
    when the browser cannot reach /ed-stream directly (e.g. private network)."""
    return await _build_snapshot()


@router.get("/ed-view", response_class=HTMLResponse)
async def ed_view():
    try:
        html = _HTML_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return HTMLResponse("<h1>ed_view.html missing</h1>", status_code=500)
    return HTMLResponse(html)


@router.get("/ed-stream")
async def ed_stream(request: Request):
    """Server-Sent Events stream of ED snapshots, one per ~1.5s."""

    async def event_generator():
        # Immediate first snapshot so the canvas isn't blank on open
        try:
            snap = await _build_snapshot()
            yield f"data: {json.dumps(snap)}\n\n"
        except Exception as exc:
            logger.warning("initial snapshot failed: %s", exc)

        while True:
            if await request.is_disconnected():
                break
            try:
                await asyncio.sleep(_TICK_SECONDS)
                snap = await _build_snapshot()
                yield f"data: {json.dumps(snap)}\n\n"
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("snapshot tick failed: %s", exc)
                # Send a keepalive comment so EventSource doesn't time out
                yield ": keepalive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx: disable buffering
        },
    )
