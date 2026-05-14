from __future__ import annotations
import asyncio, logging, os, time
import aiosqlite
from src.database.models import SCHEMA_SQL
from src.core.patient import IntakeForm, WorkupPlan

logger = logging.getLogger("pce.db")
DB_PATH = os.getenv("PCE_DB_PATH", "data/pce_demo.db")


def _resolve_path(db_path: str | None) -> str:
    return db_path if db_path is not None else DB_PATH


async def init_db(db_path: str | None = None) -> None:
    """Initialise the database, creating tables if they do not exist."""
    path = _resolve_path(db_path)
    if path != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    async with aiosqlite.connect(path) as db:
        await db.executescript(SCHEMA_SQL)
        await db.commit()
    logger.info("init_db | path=%s", path)


async def save_patient(
    intake: IntakeForm,
    esi_result,
    score_result,
    db_path: str | None = None,
) -> None:
    """INSERT OR REPLACE a patient row from intake + triage results."""
    path = _resolve_path(db_path)
    now = time.time()
    async with aiosqlite.connect(path) as db:
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
                esi_result.esi_level,
                score_result.risk_score,
                1 if score_result.is_red else 0,
                None,       # protocol_key set later by workup
                "waiting",  # initial status
                None,       # assigned_doctor
                now,        # created_at
                now,        # updated_at
            ),
        )
        await db.commit()
    logger.debug("save_patient | patient=%s", intake.patient_id[:8])


async def update_patient_score(
    patient_id: str,
    risk_score: float,
    is_red: bool,
    status: str,
    db_path: str | None = None,
) -> None:
    path = _resolve_path(db_path)
    now = time.time()
    async with aiosqlite.connect(path) as db:
        await db.execute(
            """
            UPDATE patients
            SET risk_score=?, is_red=?, status=?, updated_at=?
            WHERE patient_id=?
            """,
            (risk_score, 1 if is_red else 0, status, now, patient_id),
        )
        await db.commit()
    logger.debug("update_patient_score | patient=%s risk=%.1f", patient_id[:8], risk_score)


async def save_investigation_orders(
    patient_id: str,
    workup: WorkupPlan,
    db_path: str | None = None,
) -> None:
    """Bulk INSERT investigation orders from a WorkupPlan."""
    path = _resolve_path(db_path)
    now = time.time()
    rows = [
        (
            patient_id,
            order.test_name,
            order.cost_tier,
            order.timing,
            order.whitelist_rule,
            "pending",
            now,
            None,   # resulted_at
            None,   # result_value
        )
        for order in workup.orders
    ]
    async with aiosqlite.connect(path) as db:
        await db.executemany(
            """
            INSERT INTO investigations
                (patient_id, test_name, cost_tier, timing, protocol_rule,
                 status, ordered_at, resulted_at, result_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        await db.commit()
    logger.debug(
        "save_investigation_orders | patient=%s orders=%d",
        patient_id[:8], len(rows),
    )


async def update_investigation_status(
    patient_id: str,
    test_name: str,
    status: str,
    result_value: str | None = None,
    db_path: str | None = None,
) -> None:
    path = _resolve_path(db_path)
    now = time.time()
    async with aiosqlite.connect(path) as db:
        await db.execute(
            """
            UPDATE investigations
            SET status=?, resulted_at=?, result_value=?
            WHERE patient_id=? AND test_name=?
            """,
            (status, now, result_value, patient_id, test_name),
        )
        await db.commit()
    logger.debug(
        "update_investigation_status | patient=%s test=%s status=%s",
        patient_id[:8], test_name, status,
    )


async def log_agent_action(
    patient_id: str,
    agent_id: int,
    action: str,
    inputs_summary: str,
    outputs_summary: str,
    latency_ms: float,
    model_used: str,
    db_path: str | None = None,
) -> None:
    path = _resolve_path(db_path)
    now = time.time()
    async with aiosqlite.connect(path) as db:
        await db.execute(
            """
            INSERT INTO audit_log
                (patient_id, agent_id, action, inputs_summary, outputs_summary,
                 latency_ms, model_used, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (patient_id, agent_id, action, inputs_summary, outputs_summary,
             latency_ms, model_used, now),
        )
        await db.commit()
    logger.debug("log_agent_action | patient=%s action=%s", patient_id[:8], action)


async def get_waiting_patients(db_path: str | None = None) -> list[dict]:
    """Return patients with status in ('waiting','workup','assigned'), priority ordered."""
    path = _resolve_path(db_path)
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT * FROM patients
            WHERE status IN ('waiting', 'workup', 'assigned')
            ORDER BY is_red DESC, risk_score DESC
            """
        )
        rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def update_patient_status(patient_id: str, status: str, db_path: str | None = None) -> None:
    path = _resolve_path(db_path)
    async with aiosqlite.connect(path) as db:
        await db.execute(
            "UPDATE patients SET status = ? WHERE patient_id = ?",
            (status, patient_id),
        )
        await db.commit()
    logger.debug("update_patient_status | patient=%s status=%s", patient_id[:8], status)


async def get_patient(patient_id: str, db_path: str | None = None) -> dict | None:
    path = _resolve_path(db_path)
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM patients WHERE patient_id=?",
            (patient_id,),
        )
        row = await cursor.fetchone()
    return dict(row) if row else None
