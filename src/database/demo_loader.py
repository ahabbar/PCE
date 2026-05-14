import asyncio, json, logging, os
from src.database.db import save_patient, init_db

logger = logging.getLogger("pce.demo_loader")


async def load_demo_cases(
    json_path: str = "data/cases/demo_cases.json",
    db_path: str | None = None,
) -> None:
    """Load demo cases from JSON and insert them into the database."""
    if not os.path.exists(json_path):
        logger.warning("Demo cases file not found: %s", json_path)
        return

    with open(json_path, encoding="utf-8") as f:
        cases = json.load(f)

    await init_db(db_path=db_path)

    inserted = 0
    skipped = 0
    for case in cases:
        try:
            from src.core.patient import (
                IntakeForm,
                ChiefComplaint,
                Vitals,
                MedicalHistory,
                NurseObservation,
            )

            intake = IntakeForm(
                patient_id=case["patient_id"],
                arrival_time=case.get("arrival_time", 0.0),
                age_years=case["age_years"],
                gender=case.get("gender", "unknown"),
                chief_complaint=ChiefComplaint(
                    free_text_en=case["chief_complaint_text"],
                    category=case.get("chief_complaint_category", "other"),
                ),
                vitals=Vitals(**case.get("vitals", {})),
                medical_history=MedicalHistory(**case.get("medical_history", {})),
            )

            class _ESI:
                esi_level: int = case.get("esi_level", 3)

            class _Score:
                risk_score: float = case.get("risk_score", 50.0)
                is_red: bool = bool(case.get("is_red", False))

            await save_patient(intake, _ESI(), _Score(), db_path=db_path)
            inserted += 1
            logger.debug("Inserted demo case: %s", case["patient_id"])
        except Exception as exc:
            logger.error("Failed to insert demo case %s: %s", case.get("patient_id", "?"), exc)
            skipped += 1

    logger.info(
        "load_demo_cases | path=%s inserted=%d skipped=%d",
        json_path, inserted, skipped,
    )
