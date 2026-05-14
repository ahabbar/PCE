import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

import asyncio, uuid, time
from src.database.db import init_db
from src.orchestrator.engine import process_patient
from src.orchestrator.queue import get_queue
from src.agents.batch import get_batch_coordinator
from src.core.patient import IntakeForm, Vitals, ChiefComplaint, MedicalHistory, NurseObservation

PATIENTS = [
    {
        "cc": "chest pain and diaphoresis",
        "cat": "chest_pain",
        "age": 58,
        "gender": "male",
        "vitals": {"heart_rate": 108, "systolic_bp": 88, "spo2_pct": 91, "temperature_c": 36.8},
    },
    {
        "cc": "fever and flank pain",
        "cat": "urinary",
        "age": 34,
        "gender": "female",
        "vitals": {"heart_rate": 98, "systolic_bp": 118, "spo2_pct": 97, "temperature_c": 38.6},
        "diagnoses": ["CKD stage 2"],
    },
    {
        "cc": "confusion and weakness",
        "cat": "altered_mental_status",
        "age": 75,
        "gender": "male",
        "vitals": {"heart_rate": 108, "systolic_bp": 102, "spo2_pct": 93, "temperature_c": 38.6, "gcs": 13},
    },
    {
        "cc": "shortness of breath",
        "cat": "dyspnea",
        "age": 68,
        "gender": "female",
        "vitals": {"heart_rate": 96, "systolic_bp": 142, "spo2_pct": 88, "temperature_c": 37.1},
    },
    {
        "cc": "mild dysuria",
        "cat": "urinary",
        "age": 29,
        "gender": "female",
        "vitals": {"heart_rate": 72, "systolic_bp": 118, "spo2_pct": 99, "temperature_c": 36.8},
    },
]


async def main():
    await init_db()
    q = get_queue()
    bc = get_batch_coordinator()
    print("\nProcessing 5 patients...\n")

    for p in PATIENTS:
        intake = IntakeForm(
            patient_id=str(uuid.uuid4()),
            age_years=p["age"],
            gender=p["gender"],
            chief_complaint=ChiefComplaint(
                free_text_en=p["cc"],
                category=p["cat"],
                pain_present=True,
                pain_score=5,
            ),
            vitals=Vitals(**p.get("vitals", {})),
            medical_history=MedicalHistory(known_diagnoses=p.get("diagnoses", [])),
        )
        r = await process_patient(intake, patient_load=5)
        await q.add_patient(r)
        if r.workup:
            bc.add_orders(r.intake.patient_id, r.workup, r.esi_result.esi_level)
        print(
            f"  Pt {r.intake.age_years:.0f}{r.intake.gender[0].upper()}"
            f"  ESI-{r.esi_result.esi_level}"
            f"  Score:{r.triage_score.risk_score:.0f}%"
            f"  Red:{r.is_red}"
            f"  Orders:{len(r.workup.orders) if r.workup else 0}"
            f"  {r.total_latency_ms:.0f}ms"
        )

    print("\n── Sorted Queue ──")
    for i, pt in enumerate(await q.get_sorted_queue(), 1):
        score = pt.triage_score.risk_score if pt.triage_score else 0
        red = pt.triage_score.is_red if pt.triage_score else False
        print(
            f"  {i}. {pt.intake.chief_complaint.free_text_en[:30]:<30}"
            f"  Score:{score:.0f}%  Red:{red}"
        )

    print("\n── Batch Summary ──")
    for test, count in bc.get_batch_summary().items():
        print(f"  {test} x {count} patients")
    eff = bc.batch_efficiency_pct()
    print(f"  Efficiency: {eff:.0f}%")


if __name__ == "__main__":
    asyncio.run(main())
