#!/usr/bin/env python3
"""
PCE Full Journey Demo — 34F UTI+CKD through all 7 agents.
Usage:
    python scripts/full_journey_demo.py
    python scripts/full_journey_demo.py --slow   (0.5s pause between events for live demo)
"""
from __future__ import annotations
import argparse, sys, time, uuid
import requests

API = "http://localhost:8000"
SLOW = False  # set from --slow arg

def step(delay: float = 0):
    if SLOW:
        time.sleep(0.5)
    elif delay:
        time.sleep(delay)

def fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"T+{m:02d}:{s:02d}"

def print_event(tag: str, msg: str, t: float):
    print(f"{fmt_time(t)}  [{tag:<10}]  {msg}")

def check_server():
    try:
        r = requests.get(f"{API}/health", timeout=5)
        if r.status_code == 200:
            return True
    except Exception:
        pass
    print("ERROR: API server not running. Start with:")
    print("  venv\\Scripts\\python -m uvicorn src.api.server:app --port 8000")
    sys.exit(1)


def main():
    global SLOW
    parser = argparse.ArgumentParser()
    parser.add_argument("--slow", action="store_true")
    args = parser.parse_args()
    SLOW = args.slow

    check_server()
    start = time.time()

    print("=" * 60)
    print("PARALLEL CARE ENGINE — Full Journey Demo")
    print("Patient: 34F | UTI + CKD Stage 2")
    print("=" * 60)
    print()

    # ── Step 1: Patient arrival + triage ──
    patient_id = str(uuid.uuid4())
    intake_payload = {
        "patient_id": patient_id,
        "age_years": 34,
        "gender": "female",
        "chief_complaint": {
            "free_text_en": "Dysuria, flank pain, fever — known CKD stage 2",
            "category": "urinary",
            "pain_present": True,
            "pain_score": 5
        },
        "vitals": {
            "heart_rate": 98.0,
            "temperature_c": 38.6,
            "systolic_bp": 118.0,
            "diastolic_bp": 74.0,
            "spo2_pct": 97.0,
            "gcs": 15.0
        },
        "medical_history": {
            "known_diagnoses": ["CKD Stage 2", "Recurrent UTI"],
            "immunocompromised": False,
            "anticoagulated": False
        },
        "nurse_observation": {
            "general_appearance": "unwell",
            "work_of_breathing": "normal",
            "skin_assessment": "normal",
            "altered_mentation": False,
            "acute_distress": False,
            "seizure_active": False,
            "pulse_quality": "normal",
            "avpu": "A"
        }
    }

    t_arrival = time.time() - start
    print_event("ARRIVAL", f"34F UTI+CKD | Registering patient {patient_id[:8]}...", t_arrival)
    step()

    print_event("PARALLEL", "Running Agents 1 + 2 + 3 simultaneously via orchestrator...", time.time() - start)

    triage_resp = requests.post(f"{API}/triage", json=intake_payload, timeout=120)
    if triage_resp.status_code != 200:
        print(f"ERROR: Triage failed: {triage_resp.text[:200]}")
        sys.exit(1)
    tr = triage_resp.json()
    t_triage = time.time() - start

    print_event("AGENT 1", f"Risk Score: {tr['risk_score']:.0f}% | Threshold: {tr['threshold_used']:.0f}% | {'RED' if tr['is_red'] else 'NOT RED'}", t_triage)
    step()
    print_event("AGENT 2", f"Red Flag: {'EMERGENCY — ' + str(tr['flag_type']) if tr['is_emergency'] else 'CLEAR — no immediate emergency'}", t_triage)
    step()
    orders = [o['test_name'] for o in tr.get('workup_orders', [])]
    print_event("AGENT 3", f"Workup: {', '.join(orders) if orders else 'none'} ({len(orders)} orders)", t_triage)
    step()
    print_event("ESI", f"ESI-{tr['esi_level']} | Scope: {tr['esi_scope']} | Parallel: {tr['parallel_confirmed']}", t_triage)
    step()
    print()

    # Track score progression
    current_score = tr['risk_score']
    initial_score = current_score
    threshold = tr['threshold_used']
    threshold_crossed = tr['is_red']

    # ── Step 2: Lab results arrive ──
    lab_results = [
        ("UA POCT",     "Nitrites++, WBC >50, Bacteria +++",          True,  25),
        ("CBC",         "WBC 16,800 (leukocytosis)",                  False, 30),
        ("Creatinine",  "1.9 mmol/L (AKI on CKD, baseline 1.2)",     True,  40),
        ("CRP",         "CRP 185 mg/L (elevated)",                    False, 45),
    ]

    doctor_name = None

    for test_name, result_value, is_critical, sim_minutes in lab_results:
        t_lab = time.time() - start
        crit_tag = " [CRITICAL]" if is_critical else ""
        print_event("LAB", f"{test_name}: {result_value}{crit_tag}", t_lab)
        step()

        result_resp = requests.post(f"{API}/result", json={
            "patient_id": patient_id,
            "test_name": test_name,
            "result_value": result_value,
            "result_time": time.time(),
            "is_critical": is_critical,
        }, timeout=120)

        t_update = time.time() - start
        if result_resp.status_code == 200:
            ev = result_resp.json()
            if ev.get("message"):
                print_event("UPDATE", ev["message"], t_update)
            else:
                old = ev.get("old_score", current_score)
                new = ev.get("new_score", current_score)
                delta = ev.get("delta", 0)
                reordered = ev.get("queue_reordered", False)
                crossed = ev.get("threshold_crossed", False)
                current_score = new

                reorder_tag = " | Queue re-sorted → position 1" if reordered else ""
                print_event("UPDATE", f"Score: {old:.0f}% → {new:.0f}% (Δ{delta:+.0f}){reorder_tag}", t_update)

                if crossed and not threshold_crossed:
                    threshold_crossed = True
                    print_event("ALERT", f"Score >= {threshold:.0f}% threshold — patient moved to RED zone", t_update)
        step()

        # After 2nd result: assign doctor
        if test_name == "CBC":
            t_assign = time.time() - start
            print()
            assign_resp = requests.post(f"{API}/assign/{patient_id}",
                                        json={"esi_level": tr["esi_level"]}, timeout=10)
            if assign_resp.status_code == 200:
                ad = assign_resp.json()
                if ad.get("assigned"):
                    doctor_name = ad["doctor_name"]
                    print_event("AGENT 5", f"Doctor assigned: {doctor_name} ({ad['doctor_role']}) | {ad['reason']}", t_assign)
                else:
                    doctor_name = "On-call Registrar"
                    print_event("AGENT 5", "No preferred doctor available — escalating to next available", t_assign)
            step()
            print()

    print()

    # ── Step 3: Doctor encounter + exit ──
    t_doctor = time.time() - start
    print_event("DOCTOR", f"Encounter begins | Pre-computed results ready | Doctor: {doctor_name or 'Unknown'}", t_doctor)
    step()

    exit_resp = requests.post(f"{API}/exit/{patient_id}", json={
        "disposition": "admit",
        "confirmed_diagnosis": "Pyelonephritis with AKI stage 2 on CKD background",
        "doctor_name": doctor_name or "Dr. Patel",
    }, timeout=120)

    t_exit = time.time() - start
    if exit_resp.status_code == 200:
        plan = exit_resp.json()
        print_event("AGENT 7", "Disposition: ADMIT | Exit Coordinator activated", t_exit)
        step()
        if plan.get("handover_note"):
            preview = plan["handover_note"][:100].replace("\n", " ")
            print_event("EXIT", f"Handover note: {preview}...", t_exit)
        print_event("COMPLETE", f"Patient admitted | Doctor approves (one tap)", t_exit)
    else:
        print_event("EXIT", f"Exit plan error: {exit_resp.text[:100]}", t_exit)

    # ── Summary ──
    total_time = time.time() - start
    traditional_time = 189  # minutes (typical ED LOS)
    pce_time_est = 45 + int(total_time / 60)  # wait + actual agent time

    print()
    print("=" * 54)
    print("PARALLEL CARE ENGINE — Journey Summary")
    print("=" * 54)
    print(f"  Script runtime:        {total_time:.0f}s (real LLM calls)")
    print(f"  Score progression:     {initial_score:.0f}% → {current_score:.0f}%")
    print(f"  Threshold crossed:     {'Yes — moved to RED' if threshold_crossed else 'No'}")
    print(f"  Lab results injected:  {len(lab_results)}")
    print(f"  Doctor assigned:       {doctor_name or 'Unknown'}")
    print(f"  Workup orders issued:  {len(tr.get('workup_orders', []))}")
    print(f"  Parallel confirmed:    {tr['parallel_confirmed']}")
    print(f"  ESI level:             {tr['esi_level']}")
    print("=" * 54)


if __name__ == "__main__":
    main()
