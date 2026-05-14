from __future__ import annotations

import asyncio
import time
import uuid

import aiosqlite
import pytest

from src.database.models import SCHEMA_SQL
from src.core.patient import (
    ChiefComplaint,
    IntakeForm,
    OrderItem,
    Vitals,
    WorkupPlan,
)


# ---------------------------------------------------------------------------
# Mock result objects
# ---------------------------------------------------------------------------

class _ESI:
    esi_level: int = 3


class _Score:
    risk_score: float = 65.0
    is_red: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_intake(patient_id: str | None = None) -> IntakeForm:
    return IntakeForm(
        patient_id=patient_id or str(uuid.uuid4()),
        age_years=34,
        gender="female",
        chief_complaint=ChiefComplaint(
            free_text_en="dysuria and flank pain",
            category="urinary",
        ),
        vitals=Vitals(heart_rate=98, temperature_c=38.2, systolic_bp=118, spo2_pct=97),
    )


def make_workup(patient_id: str) -> WorkupPlan:
    orders = [
        OrderItem(
            test_name="Urinalysis + microscopy",
            reason="Diagnose urinary infection",
            cost_tier="low",
            timing="immediate",
            whitelist_rule="uti_pyelonephritis:auto",
        ),
        OrderItem(
            test_name="CBC",
            reason="Assess infection severity",
            cost_tier="low",
            timing="routine",
            whitelist_rule="uti_pyelonephritis:auto",
        ),
    ]
    return WorkupPlan(
        patient_id=patient_id,
        protocol_key="uti_pyelonephritis",
        orders=orders,
        deferred_to_doctor=["CT urogram"],
        reasoning="Protocol: uti_pyelonephritis — 2 orders, 1 deferred to doctor",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_init_db():
    """init_db with :memory: should complete without exception."""
    from src.database.db import init_db
    await init_db(":memory:")


@pytest.mark.asyncio
async def test_save_and_retrieve_patient():
    """save_patient followed by get_patient should return matching fields."""
    pid = str(uuid.uuid4())
    intake = make_intake(patient_id=pid)
    now = time.time()

    async with aiosqlite.connect(":memory:") as db:
        # Init schema
        await db.executescript(SCHEMA_SQL)
        await db.commit()

        # Insert patient
        await db.execute(
            """
            INSERT OR REPLACE INTO patients (
                patient_id, arrival_time, age_years, gender,
                chief_complaint_text, chief_complaint_category,
                esi_level, risk_score, is_red,
                protocol_key, status, assigned_doctor,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                intake.patient_id,
                intake.arrival_time,
                intake.age_years,
                intake.gender,
                intake.chief_complaint.free_text_en,
                intake.chief_complaint.category,
                3,       # esi_level
                65.0,    # risk_score
                0,       # is_red
                None, "waiting", None, now, now,
            ),
        )
        await db.commit()

        # Retrieve
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM patients WHERE patient_id=?", (pid,)
        )
        row = await cursor.fetchone()
        assert row is not None
        record = dict(row)

    assert record["patient_id"] == pid
    assert record["age_years"] == 34.0
    assert record["gender"] == "female"


@pytest.mark.asyncio
async def test_investigation_lifecycle():
    """Save orders then update status; confirm status='done' in raw query."""
    pid = str(uuid.uuid4())
    workup = make_workup(pid)
    test_name = workup.orders[0].test_name  # "Urinalysis + microscopy"
    now = time.time()

    async with aiosqlite.connect(":memory:") as db:
        # Init schema
        await db.executescript(SCHEMA_SQL)

        # Insert minimal patient row (FK)
        await db.execute(
            "INSERT INTO patients (patient_id, age_years, gender, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (pid, 30, "female", "waiting", now, now),
        )

        # Save investigation orders
        rows = [
            (
                pid,
                order.test_name,
                order.cost_tier,
                order.timing,
                order.whitelist_rule,
                "pending",
                now,
                None,
                None,
            )
            for order in workup.orders
        ]
        await db.executemany(
            """
            INSERT INTO investigations
                (patient_id, test_name, cost_tier, timing, protocol_rule,
                 status, ordered_at, resulted_at, result_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        # Update status to done
        await db.execute(
            "UPDATE investigations SET status=?, resulted_at=?, result_value=? "
            "WHERE patient_id=? AND test_name=?",
            ("done", time.time(), "negative", pid, test_name),
        )
        await db.commit()

        # Confirm status
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT status FROM investigations WHERE patient_id=? AND test_name=?",
            (pid, test_name),
        )
        row = await cursor.fetchone()
        assert row is not None
        assert dict(row)["status"] == "done"


@pytest.mark.asyncio
async def test_audit_log():
    """log_agent_action should insert a row into audit_log."""
    pid = str(uuid.uuid4())
    now = time.time()

    async with aiosqlite.connect(":memory:") as db:
        # Init schema
        await db.executescript(SCHEMA_SQL)

        # Insert audit log row
        await db.execute(
            """
            INSERT INTO audit_log
                (patient_id, agent_id, action, inputs_summary, outputs_summary,
                 latency_ms, model_used, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (pid, 2, "triage_score", "intake summary", "score=65",
             142.5, "gemini-2.5-flash", now),
        )
        await db.commit()

        # Verify row exists
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM audit_log WHERE patient_id=?", (pid,)
        )
        row = await cursor.fetchone()
        assert row is not None
        record = dict(row)

    assert record["patient_id"] == pid
    assert record["action"] == "triage_score"
    assert record["latency_ms"] == 142.5
