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


async def set_patient_disposition(patient_id: str, disposition: str, db_path: str | None = None) -> None:
    """Persist the final disposition (discharge/admit/icu/transfer) on the
    patient row. Stored in the existing `disposition_prediction` column so
    /ed-snapshot can count outflow per destination."""
    path = _resolve_path(db_path)
    async with aiosqlite.connect(path) as db:
        await db.execute(
            "UPDATE patients SET disposition_prediction = ? WHERE patient_id = ?",
            (disposition, patient_id),
        )
        await db.commit()
    logger.debug("set_patient_disposition | patient=%s dispo=%s", patient_id[:8], disposition)


async def count_dispositions(db_path: str | None = None) -> dict[str, int]:
    """Tally completed patients by disposition. Only counts rows already in a
    terminal status (seen/discharged) so in-flight patients aren't included."""
    path = _resolve_path(db_path)
    counts = {"discharge": 0, "admit": 0, "icu": 0, "transfer": 0}
    try:
        async with aiosqlite.connect(path) as db:
            cursor = await db.execute(
                "SELECT disposition_prediction, COUNT(*) FROM patients "
                "WHERE status IN ('seen','discharged') AND disposition_prediction IS NOT NULL "
                "GROUP BY disposition_prediction"
            )
            rows = await cursor.fetchall()
        for dispo, n in rows:
            key = str(dispo or "").lower()
            if key in counts:
                counts[key] = int(n)
    except Exception:
        pass
    return counts


async def get_recent_patients(db_path: str | None = None) -> list[dict]:
    """Return all patients from the current DB (for analytics)."""
    path = _resolve_path(db_path)
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM patients ORDER BY created_at DESC")
        rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def save_patient_basic(record: dict, db_path: str | None = None) -> None:
    """INSERT OR REPLACE a patient row from a plain dict (used by demo seed)."""
    path = _resolve_path(db_path)
    now = time.time()
    async with aiosqlite.connect(path) as db:
        await db.execute(
            """
            INSERT OR REPLACE INTO patients (
                patient_id, arrival_time, age_years, gender,
                chief_complaint_text, chief_complaint_category,
                esi_level, risk_score, is_red,
                pce_scope, threshold_used, status, assigned_doctor,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["patient_id"], record.get("arrival_time", now),
                record.get("age_years", 50), record.get("gender", "unknown"),
                record.get("chief_complaint_text", ""), record.get("chief_complaint_category", "other"),
                record.get("esi_level", 3), record.get("risk_score", 50.0),
                1 if record.get("is_red") else 0,
                record.get("pce_scope", "pce_core"), record.get("threshold_used", 85.0),
                record.get("status", "waiting"), record.get("assigned_doctor"),
                now, now,
            ),
        )
        await db.commit()


async def increment_result_count_db(patient_id: str, db_path: str | None = None) -> None:
    path = _resolve_path(db_path)
    async with aiosqlite.connect(path) as db:
        await db.execute(
            "UPDATE patients SET result_count = COALESCE(result_count, 0) + 1 WHERE patient_id = ?",
            (patient_id,),
        )
        await db.commit()


async def clear_non_permanent_patients(db_path: str | None = None) -> None:
    """Delete all patients for demo reset."""
    path = _resolve_path(db_path)
    async with aiosqlite.connect(path) as db:
        await db.execute("DELETE FROM investigations")
        await db.execute("DELETE FROM patients")
        await db.commit()


async def get_patient_labs(patient_id: str, db_path: str | None = None) -> list[dict]:
    """Return all investigation orders for a patient, sorted by ordered_at."""
    path = _resolve_path(db_path)
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM investigations WHERE patient_id=? ORDER BY ordered_at ASC",
            (patient_id,),
        )
        rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def seed_lab_orders(patient_id: str, test_names: list[str], db_path: str | None = None) -> None:
    """Insert suggested lab orders for a patient (status=pending_approval)."""
    from src.core.lab_values import PANEL_EXPANSION
    path = _resolve_path(db_path)
    now = time.time()
    # Expand panels (e.g. CBC → WBC, HGB, PLT)
    expanded: list[str] = []
    for name in test_names:
        expanded.extend(PANEL_EXPANSION.get(name, [name]))
    # Only insert tests that don't already exist for this patient
    async with aiosqlite.connect(path) as db:
        existing = await (await db.execute(
            "SELECT test_name FROM investigations WHERE patient_id=?", (patient_id,)
        )).fetchall()
        existing_names = {row[0] for row in existing}
        rows = [
            (patient_id, name, "low", "routine", "suggested", "pending_approval", now, None, None)
            for name in expanded if name not in existing_names
        ]
        if rows:
            await db.executemany(
                """INSERT INTO investigations
                   (patient_id, test_name, cost_tier, timing, protocol_rule,
                    status, ordered_at, resulted_at, result_value)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            await db.commit()


async def expand_existing_panels(patient_id: str, db_path: str | None = None) -> bool:
    """
    Replace any panel-name rows (e.g. 'CBC') with their individual components.
    Returns True if any expansion happened.
    """
    from src.core.lab_values import PANEL_EXPANSION
    path = _resolve_path(db_path)
    now = time.time()
    changed = False
    async with aiosqlite.connect(path) as db:
        rows = await (await db.execute(
            "SELECT test_name, status FROM investigations WHERE patient_id=?", (patient_id,)
        )).fetchall()
        existing_names = {r[0] for r in rows}
        for row in rows:
            panel_name, panel_status = row[0], row[1]
            components = PANEL_EXPANSION.get(panel_name)
            if not components:
                continue
            # Delete the old panel row
            await db.execute(
                "DELETE FROM investigations WHERE patient_id=? AND test_name=?",
                (patient_id, panel_name),
            )
            # Insert individual components that don't already exist
            new_rows = [
                (patient_id, comp, "low", "routine", "suggested", panel_status, now, None, None)
                for comp in components if comp not in existing_names
            ]
            if new_rows:
                await db.executemany(
                    """INSERT INTO investigations
                       (patient_id, test_name, cost_tier, timing, protocol_rule,
                        status, ordered_at, resulted_at, result_value)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    new_rows,
                )
            changed = True
        if changed:
            await db.commit()
    return changed


async def confirm_lab_order(
    patient_id: str, test_name: str, action: str, db_path: str | None = None
) -> None:
    """Set investigation status to 'approved' or 'rejected'."""
    path = _resolve_path(db_path)
    async with aiosqlite.connect(path) as db:
        await db.execute(
            "UPDATE investigations SET status=? WHERE patient_id=? AND test_name=?",
            (action, patient_id, test_name),
        )
        await db.commit()


async def enter_lab_result(
    patient_id: str,
    test_name: str,
    result_value: str,
    is_critical: bool,
    is_abnormal: bool,
    interpretation: str,
    db_path: str | None = None,
) -> None:
    """Record a lab result and mark investigation as resulted."""
    path = _resolve_path(db_path)
    now = time.time()
    full_value = f"{result_value} | {interpretation}"
    async with aiosqlite.connect(path) as db:
        await db.execute(
            """UPDATE investigations
               SET status='resulted', resulted_at=?, result_value=?
               WHERE patient_id=? AND test_name=?""",
            (now, full_value, patient_id, test_name),
        )
        await db.commit()


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
