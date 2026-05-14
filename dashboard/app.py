from __future__ import annotations
import asyncio, os, sys, time, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

import requests
import streamlit as st
os.environ.setdefault("LLM_PROVIDER", "gemini")

from src.core.patient import ChiefComplaint, IntakeForm, MedicalHistory, NurseObservation, Vitals
from src.core.esi_algorithm import run_esi_algorithm
from src.agents.llm import get_llm_client
from src.agents.triage_score import run_triage_score
from src.agents.red_flag import run_red_flag_guardian

API_URL = "http://localhost:8000"

st.set_page_config(page_title="PCE — Parallel Care Engine", layout="wide")
st.title("Parallel Care Engine")
st.caption("Emergency Triage — AI Risk Assessment")

# ── Constants ────────────────────────────────────────────────────────────────

HIS_DATA = {
    "chest_pain":           {"diagnoses": "Hypertension, Hyperlipidaemia",    "meds": "Amlodipine, Atorvastatin",       "cr": "0.9"},
    "fever_infection":      {"diagnoses": "Type 2 Diabetes",                  "meds": "Metformin, Insulin",             "cr": "1.1"},
    "urinary":              {"diagnoses": "CKD Stage 2, Recurrent UTI",       "meds": "Nitrofurantoin prn",             "cr": "1.6"},
    "dyspnea":              {"diagnoses": "COPD, Heart Failure",              "meds": "Furosemide, Salbutamol inhaler", "cr": "1.3"},
    "abdominal_pain":       {"diagnoses": "No significant history",           "meds": "None",                          "cr": "0.8"},
    "altered_mental_status":{"diagnoses": "Type 2 Diabetes, CKD",            "meds": "Metformin, Ramipril",            "cr": "1.4"},
    "other":                {"diagnoses": "Not retrieved",                    "meds": "Unknown",                       "cr": "Unknown"},
}
CC_OPTIONS = {
    "chest_pain": "Chest Pain", "fever_infection": "Fever / Infection",
    "urinary": "Dysuria / Flank Pain / Urinary", "dyspnea": "Shortness of Breath",
    "abdominal_pain": "Abdominal Pain", "altered_mental_status": "Confusion / Altered Mental Status",
    "other": "Other",
}
CC_TO_CATEGORY = {
    "chest_pain": "chest_pain", "fever_infection": "fever", "urinary": "urinary",
    "dyspnea": "dyspnea", "abdominal_pain": "abdominal_pain",
    "altered_mental_status": "altered_mental_status", "other": "other",
}
ASSOCIATED_SYMPTOMS = [
    "Nausea / Vomiting", "Diaphoresis (sweating)", "Palpitations", "Syncope / Near-syncope",
    "Cough", "Haemoptysis (coughing blood)", "Headache", "Dizziness / Vertigo", "Back / Flank pain",
    "Leg swelling / Oedema", "Diarrhoea", "Constipation", "Haematuria (blood in urine)",
    "Dysuria (painful urination)", "Jaundice", "Rash / Skin changes", "Visual disturbance",
    "Focal weakness / Numbness", "Seizure history", "Trauma / Fall", "Recent surgery",
    "Recent travel", "Immobility (>72h)", "Pregnancy / Post-partum",
]
COMORBIDITIES_LIST = [
    "Hypertension", "Type 2 Diabetes", "Type 1 Diabetes", "Coronary artery disease",
    "Heart failure", "Atrial fibrillation", "COPD", "Asthma", "CKD (Chronic Kidney Disease)",
    "End-stage renal disease", "Liver cirrhosis", "Active malignancy",
    "Immunocompromised / On chemotherapy", "Organ transplant recipient", "HIV / AIDS",
    "Stroke / TIA history", "Peripheral vascular disease", "Obesity (BMI >35)",
    "Anticoagulation therapy", "Chronic steroid use", "Dementia", "Epilepsy",
    "Thyroid disease", "Sickle cell disease",
]
ESI_COLORS = {1: "#ef4444", 2: "#f97316", 3: "#eab308", 4: "#22c55e", 5: "#3b82f6"}
ESI_LABELS = {
    1: "Immediate — Life Threat", 2: "Emergent — High Risk",
    3: "Urgent — Needs Workup", 4: "Less Urgent — Stable", 5: "Non-urgent — Minimal Risk",
}
PCE_SCOPE_LABELS = {
    "excluded_resus": "EXCLUDED — Resus Bay", "pce_priority": "PCE Priority",
    "pce_core": "PCE Core", "pce_efficient": "PCE Efficient", "pce_lite": "PCE Lite",
}

tab1, tab2 = st.tabs(["Triage Input", "Live Queue"])

# ── Tab 1: Triage Input ───────────────────────────────────────────────────────

with tab1:
    left, right = st.columns([4, 6])

    with left:
        st.subheader("Patient Intake")
        cc_key = st.selectbox("Primary Chief Complaint", options=list(CC_OPTIONS.keys()),
                              format_func=lambda k: CC_OPTIONS[k])
        his = HIS_DATA[cc_key]
        st.markdown(
            f"<div style='background:#f0f4f8;border-left:3px solid #3b82f6;padding:8px 12px;"
            f"border-radius:4px;font-size:13px;margin-bottom:8px'>"
            f"<b>HIS:</b> {his['diagnoses']} &nbsp;|&nbsp; <b>Meds:</b> {his['meds']}"
            f" &nbsp;|&nbsp; <b>Baseline Cr:</b> {his['cr']}</div>",
            unsafe_allow_html=True,
        )
        associated = st.multiselect("Associated Symptoms", options=ASSOCIATED_SYMPTOMS,
                                    placeholder="Select all that apply...")
        st.divider()
        demo_col1, demo_col2 = st.columns(2)
        with demo_col1:
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=34)
        with demo_col2:
            gender = st.selectbox("Gender", ["Female", "Male", "Other", "Unknown"])
        st.markdown("**Vitals**")
        v1, v2, v3 = st.columns(3)
        with v1:
            hr  = st.number_input("HR (bpm)",     min_value=0,   max_value=300, value=98)
            dbp = st.number_input("BP Diastolic", min_value=0,   max_value=200, value=74)
        with v2:
            temp = st.number_input("Temp (°C)", min_value=30.0, max_value=45.0, value=38.6, step=0.1)
            spo2 = st.number_input("SpO2 (%)",  min_value=50,   max_value=100,  value=97)
        with v3:
            sbp = st.number_input("BP Systolic", min_value=0, max_value=300, value=118)
            gcs = st.number_input("GCS (3-15)", min_value=3,  max_value=15,  value=15)
        st.divider()
        selected_comorbidities = st.multiselect("Known Comorbidities", options=COMORBIDITIES_LIST,
                                                placeholder="Select all that apply...")
        extra_comorbidities = st.text_input("Other comorbidities (free text)",
                                            placeholder="e.g. Gout, CKD stage 3")
        st.markdown("**Nurse Observation**")
        obs_col1, obs_col2 = st.columns(2)
        with obs_col1:
            appearance = st.selectbox("General Appearance",
                                      ["Well", "Unwell", "Distressed", "Critically ill", "Altered"])
        with obs_col2:
            wob = st.selectbox("Work of Breathing",
                               ["Normal", "Mildly increased", "Moderate distress", "Severe distress", "Apneic"])
        skin = st.selectbox("Skin Assessment",
                            ["Normal", "Pale", "Diaphoretic", "Mottled", "Cyanotic", "Flushed", "Jaundiced"])
        obs_flags = st.multiselect("Clinical Flags",
                                   ["Altered mentation", "Acute distress", "Active seizure", "Pulse absent / weak"],
                                   placeholder="Select if present...")
        st.divider()
        resource_choice = st.radio("Estimated Investigations / Resources",
                                   ["0 — None needed", "1 — One type", "2+ — Multiple types"])
        resource_count = int(resource_choice[0])
        patient_load = st.slider("Current ED Patient Load", 0, 40, 10)
        run_btn = st.button("Analyze — Run Agents 1 + 2 in Parallel", type="primary", use_container_width=True)

    with right:
        st.subheader("Agent Results")

        if run_btn:
            cc_text_parts = [CC_OPTIONS[cc_key]]
            if associated:
                cc_text_parts.append("with: " + ", ".join(associated))
            cc_free_text = ". ".join(cc_text_parts)
            try:
                baseline_cr = float(his["cr"])
            except Exception:
                baseline_cr = None
            his_diagnoses = [d.strip() for d in his["diagnoses"].split(",")
                             if d.strip() and d.strip() != "Not retrieved"]
            all_diagnoses = list(set(his_diagnoses + selected_comorbidities))
            if extra_comorbidities:
                all_diagnoses += [c.strip() for c in extra_comorbidities.split(",") if c.strip()]

            appearance_map = {"Well": "well", "Unwell": "unwell", "Distressed": "distressed",
                              "Critically ill": "critically_ill", "Altered": "altered"}
            wob_map = {"Normal": "normal", "Mildly increased": "mild_increased",
                       "Moderate distress": "moderate_distress", "Severe distress": "severe_distress", "Apneic": "apneic"}
            skin_map = {"Normal": "normal", "Pale": "pale", "Diaphoretic": "diaphoretic",
                        "Mottled": "mottled", "Cyanotic": "cyanotic", "Flushed": "flushed", "Jaundiced": "jaundiced"}
            immunocompromised = any(x in selected_comorbidities for x in [
                "Immunocompromised / On chemotherapy", "Organ transplant recipient", "HIV / AIDS", "Active malignancy"])
            anticoagulated = "Anticoagulation therapy" in selected_comorbidities

            intake = IntakeForm(
                patient_id=str(uuid.uuid4()), age_years=float(age), gender=gender.lower(),
                chief_complaint=ChiefComplaint(free_text_en=cc_free_text,
                    category=CC_TO_CATEGORY[cc_key],
                    pain_present="pain" in cc_free_text.lower() or "chest" in cc_key),
                vitals=Vitals(heart_rate=float(hr), temperature_c=float(temp),
                    systolic_bp=float(sbp), diastolic_bp=float(dbp), spo2_pct=float(spo2), gcs=float(gcs)),
                medical_history=MedicalHistory(known_diagnoses=all_diagnoses,
                    baseline_cr_abnormal=baseline_cr, immunocompromised=immunocompromised,
                    anticoagulated=anticoagulated),
                nurse_observation=NurseObservation(general_appearance=appearance_map[appearance],
                    work_of_breathing=wob_map[wob], skin_assessment=skin_map[skin],
                    altered_mentation="Altered mentation" in obs_flags,
                    acute_distress="Acute distress" in obs_flags,
                    seizure_active="Active seizure" in obs_flags,
                    pulse_quality="absent" if "Pulse absent / weak" in obs_flags else "normal"),
                additional_context=", ".join(associated) if associated else "",
            )

            esi = run_esi_algorithm(intake, estimated_resource_count=resource_count)

            with st.spinner("Running AI agents in parallel..."):
                llm = get_llm_client()
                async def _run_parallel(intake, esi, llm, load):
                    return await asyncio.gather(
                        run_triage_score(intake, esi, llm, patient_load=load),
                        run_red_flag_guardian(intake, llm),
                    )
                t0 = time.time()
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                score, flag = loop.run_until_complete(_run_parallel(intake, esi, llm, patient_load))
                elapsed = time.time() - t0

            rs = score.risk_score
            score_color = "#22c55e" if rs < 50 else "#f97316" if rs < 75 else "#ea580c" if rs < 90 else "#ef4444"

            # Store everything in session state so buttons in subsequent re-runs can access it
            st.session_state["result"] = {
                "intake_json": intake.model_dump(mode="json"),
                "esi_level": esi.esi_level,
                "esi_color": ESI_COLORS[esi.esi_level],
                "pce_scope_raw": esi.pce_scope,
                "pce_scope": PCE_SCOPE_LABELS.get(esi.pce_scope, esi.pce_scope),
                "decision_point": esi.decision_point_reached,
                "rationale": esi.rationale,
                "rs": rs,
                "score_color": score_color,
                "threshold": score.threshold_used,
                "is_red": score.is_red,
                "esi_level_score": score.esi_level,
                "confidence": score.confidence,
                "key_factors": score.key_factors,
                "reasoning": score.reasoning,
                "is_emergency": flag.is_emergency,
                "esi1_immediate": flag.esi1_immediate,
                "flag_type": flag.flag_type,
                "immediate_action": flag.immediate_action,
                "layer_triggered": flag.layer_triggered,
                "flag_reasoning": flag.reasoning or "",
                "elapsed": elapsed,
                "queued": False,
            }

        if "result" not in st.session_state:
            st.info("Complete the patient intake form on the left and click Analyze.")
        else:
            d = st.session_state["result"]

            st.markdown(
                f"<div style='background:{d['esi_color']};color:white;padding:12px 16px;border-radius:8px;"
                f"font-size:20px;font-weight:bold;margin-bottom:4px'>"
                f"ESI-{d['esi_level']} &nbsp; {ESI_LABELS[d['esi_level']]}</div>",
                unsafe_allow_html=True,
            )
            col_scope, col_dp = st.columns(2)
            col_scope.caption(f"Scope: {d['pce_scope']}")
            col_dp.caption(f"Decision Point: {d['decision_point']}")
            st.caption(d["rationale"])
            st.divider()

            rs = d["rs"]
            score_col, meta_col = st.columns([1, 2])
            with score_col:
                st.markdown(
                    f"<div style='font-size:64px;font-weight:900;color:{d['score_color']};line-height:1'>"
                    f"{rs:.0f}<span style='font-size:28px'>%</span></div>"
                    f"<div style='font-size:13px;color:#64748b'>Risk Score</div>",
                    unsafe_allow_html=True,
                )
            with meta_col:
                st.markdown(f"**Threshold:** {d['threshold']:.0f}% &nbsp;(ED load: {patient_load})")
                red_badge = (
                    "<span style='background:#ef4444;color:white;padding:2px 10px;border-radius:12px;"
                    "font-weight:bold;font-size:13px'>RED — Escalate</span>" if d["is_red"] else
                    "<span style='background:#22c55e;color:white;padding:2px 10px;border-radius:12px;"
                    "font-size:13px'>Not Red</span>"
                )
                st.markdown(f"**Status:** {red_badge}", unsafe_allow_html=True)
                st.markdown(f"**ESI Level:** {d['esi_level_score']} &nbsp;|&nbsp; **Confidence:** {d['confidence']:.0%}")
            st.progress(int(rs) / 100)
            factors_html = " ".join(
                f"<span style='background:#1e3a5f;color:white;padding:3px 10px;"
                f"border-radius:12px;font-size:12px;margin:2px'>{f}</span>"
                for f in d["key_factors"]
            )
            st.markdown(f"**Key Factors:** &nbsp; {factors_html}", unsafe_allow_html=True)
            st.caption(f"Reasoning: {d['reasoning']}")
            st.divider()

            st.markdown("**Red Flag Guardian**")
            if d["is_emergency"]:
                level = "ESI-1 IMMEDIATE" if d["esi1_immediate"] else "ESI-2 HIGH RISK"
                st.error(
                    f"**{level} — {(d['flag_type'] or 'EMERGENCY').replace('_', ' ').upper()}**\n\n"
                    f"{d['immediate_action'] or ''}\n\nDetected by: `{d['layer_triggered']}` layer"
                    + (f"\n\n_{d['flag_reasoning']}_" if d["flag_reasoning"] else "")
                )
            else:
                st.success(
                    f"No immediate emergency detected\n\nScreened by: `{d['layer_triggered']}` layer"
                    + (f"\n\n_{d['flag_reasoning']}_" if d["flag_reasoning"] else "")
                )
            st.caption(f"Analyzed in {d['elapsed']:.1f}s — both agents ran in parallel")
            st.divider()

            if d.get("queued"):
                st.success(f"Patient added to queue — position {d.get('queue_position', '?')}")
            elif st.button("Add to Queue", use_container_width=True):
                payload = {
                    "intake": d["intake_json"],
                    "esi_level": d["esi_level"],
                    "pce_scope": d["pce_scope_raw"],
                    "risk_score": d["rs"],
                    "is_red": d["is_red"],
                    "threshold_used": d["threshold"],
                    "confidence": d["confidence"],
                    "key_factors": d["key_factors"],
                    "reasoning": d["reasoning"],
                    "is_emergency": d["is_emergency"],
                    "esi1_immediate": d["esi1_immediate"],
                    "flag_type": d["flag_type"],
                    "immediate_action": d["immediate_action"],
                    "layer_triggered": d["layer_triggered"],
                    "flag_reasoning": d["flag_reasoning"],
                }
                with st.spinner("Adding to queue..."):
                    try:
                        resp = requests.post(f"{API_URL}/queue/add", json=payload, timeout=15)
                        if resp.status_code == 200:
                            data = resp.json()
                            st.session_state["result"]["queued"] = True
                            st.session_state["result"]["queue_position"] = data.get("position", "?")
                            st.rerun()
                        else:
                            st.warning(f"Server returned {resp.status_code}: {resp.text[:200]}")
                    except requests.exceptions.ConnectionError:
                        st.warning("API server not running. Start with: uvicorn src.api.server:app --port 8000")
                    except Exception as exc:
                        st.warning(f"Could not add to queue: {exc}")

# ── Tab 2: Live Queue ─────────────────────────────────────────────────────────

with tab2:
    st.subheader("Live Patient Queue")
    st.info("ESI-1 patients are directed to Resus Bay and excluded from this queue.")
    if st.button("Refresh", key="queue_refresh"):
        st.rerun()

    try:
        resp = requests.get(f"{API_URL}/queue", timeout=10)
        if resp.status_code != 200:
            st.warning(f"Queue API returned {resp.status_code}")
            st.stop()
        queue_data = resp.json()
    except requests.exceptions.ConnectionError:
        st.warning("API server not running. Start with: uvicorn src.api.server:app --port 8000")
        st.stop()
    except Exception as exc:
        st.warning(f"Could not fetch queue: {exc}")
        st.stop()

    stat1, stat2, stat3 = st.columns(3)
    stat1.metric("Total Patients", queue_data["total"])
    stat2.metric("Red Alerts", queue_data["reds_count"])
    stat3.metric("Queue Depth", queue_data["queue_depth"])
    st.divider()

    patients = queue_data.get("patients", [])
    if not patients:
        st.info("No patients in queue.")
    else:
        for pt in patients:
            score = pt["risk_score"]
            bg = "#ef4444" if score >= 90 else "#f97316" if score >= 75 else "#eab308" if score >= 50 else "#22c55e"
            wait = pt["wait_minutes"]
            wait_str = f"{wait:.0f} min" if wait < 60 else f"{wait/60:.1f} hr"
            st.markdown(
                f"<div style='background:{bg};color:white;padding:12px 16px;"
                f"border-radius:8px;margin-bottom:8px'>"
                f"<b>#{pt['position']} &nbsp; Score: {score:.0f}% &nbsp; ESI-{pt['esi_level']}</b>"
                f" &nbsp;|&nbsp; {pt['age']:.0f}{pt['gender'][0].upper()}"
                f"<br><span style='font-size:14px'>{pt['chief_complaint'][:60]}</span>"
                f"<br><span style='font-size:12px;opacity:0.85'>Wait: {wait_str}"
                f" &nbsp;|&nbsp; Status: {pt['status']}</span></div>",
                unsafe_allow_html=True,
            )
            patient_id = pt["patient_id"]
            with st.expander("Actions", expanded=False):
                col_assign, col_exit = st.columns(2)

                with col_assign:
                    st.markdown("**Assign Doctor**")
                    assigned = pt.get("status") == "assigned"
                    if assigned:
                        st.success(f"Doctor assigned")
                    elif st.button("Assign Doctor", key=f"assign_{patient_id}"):
                        try:
                            ar = requests.post(f"{API_URL}/assign/{patient_id}",
                                               json={"esi_level": pt["esi_level"]}, timeout=10)
                            if ar.status_code == 200:
                                d = ar.json()
                                if d.get("assigned"):
                                    st.success(f"Assigned: {d['doctor_name']} ({d['doctor_role']})")
                                    st.rerun()
                                else:
                                    st.warning("No doctor available.")
                            else:
                                st.warning(f"Error {ar.status_code}")
                        except Exception as exc:
                            st.warning(f"Error: {exc}")

                with col_exit:
                    st.markdown("**Generate Exit Plan**")
                    exit_key = f"exit_plan_{patient_id}"
                    if exit_key not in st.session_state:
                        disposition = st.selectbox(
                            "Disposition", ["discharge", "admit", "icu", "transfer"],
                            key=f"disp_{patient_id}"
                        )
                        diagnosis = st.text_input(
                            "Confirmed diagnosis", key=f"diag_{patient_id}",
                            placeholder="e.g. Pyelonephritis + AKI stage 2"
                        )
                        doctor_name = pt.get("assigned_doctor") or "Unknown"
                        if st.button("Generate Exit Plan", key=f"exit_{patient_id}"):
                            if diagnosis:
                                try:
                                    with st.spinner("Generating..."):
                                        er = requests.post(f"{API_URL}/exit/{patient_id}", json={
                                            "disposition": disposition,
                                            "confirmed_diagnosis": diagnosis,
                                            "doctor_name": doctor_name,
                                        }, timeout=60)
                                    if er.status_code == 200:
                                        st.session_state[exit_key] = er.json()
                                        st.rerun()
                                    else:
                                        st.warning(f"Error {er.status_code}: {er.text[:100]}")
                                except Exception as exc:
                                    st.warning(f"Error: {exc}")
                            else:
                                st.warning("Enter confirmed diagnosis first.")
                    else:
                        plan = st.session_state[exit_key]
                        if plan.get("instructions"):
                            st.text_area("Discharge Instructions", plan["instructions"],
                                         height=120, key=f"inst_{patient_id}", disabled=True)
                        if plan.get("handover_note"):
                            st.text_area("Handover Note", plan["handover_note"],
                                         height=120, key=f"hn_{patient_id}", disabled=True)
                        if plan.get("gp_letter_draft"):
                            st.text_area("GP Letter", plan["gp_letter_draft"],
                                         height=100, key=f"gpl_{patient_id}", disabled=True)
                        if plan.get("prescription_notes"):
                            st.caption(f"Prescription notes: {plan['prescription_notes']}")
                        if st.button("Clear Exit Plan", key=f"clear_{patient_id}"):
                            del st.session_state[exit_key]
                            st.rerun()

    batch_summary = queue_data.get("batch_summary", {})
    if batch_summary:
        st.divider()
        st.markdown("**Batch Summary** (tests shared by 2+ patients)")
        for test, count in batch_summary.items():
            st.markdown(f"- {test}: {count} patients")

    st.divider()
    st.subheader("Doctor Load")
    try:
        dr_resp = requests.get(f"{API_URL}/doctors", timeout=5)
        if dr_resp.status_code == 200:
            doctors = dr_resp.json().get("doctors", [])
            if doctors:
                for d in doctors:
                    role_label = d["role"].replace("_", " ").title()
                    cases = d["active_cases"]
                    remaining = d["total_remaining_min"]
                    st.markdown(f"**{d['name']}** ({role_label})")
                    st.progress(d["load_pct"] / 100)
                    st.caption(f"{cases} case{'s' if cases != 1 else ''} · {remaining:.0f} min remaining")
            else:
                st.info("No active doctor load.")
        else:
            st.warning("Could not fetch doctor data.")
    except requests.exceptions.ConnectionError:
        st.warning("API server offline — doctor load unavailable.")
    except Exception as exc:
        st.warning(f"Doctor load error: {exc}")

    st.divider()
    st.subheader("Inject Lab Result")
    st.caption("Simulate a result arriving from the lab system")

    if patients:
        with st.form("lab_result_form"):
            pt_options = {p["patient_id"][:8]: p["patient_id"] for p in patients}
            selected_short = st.selectbox("Patient (first 8 chars of ID)", options=list(pt_options.keys()))
            test_name = st.text_input("Test name", value="Creatinine")
            result_val = st.text_input("Result value", value="1.9 mmol/L (elevated)")
            is_critical = st.checkbox("Mark as critical value", value=False)
            inject_btn = st.form_submit_button("Inject Result")

        if inject_btn:
            full_id = pt_options[selected_short]
            try:
                resp = requests.post(f"{API_URL}/result", json={
                    "patient_id": full_id,
                    "test_name": test_name,
                    "result_value": result_val,
                    "result_time": time.time(),
                    "is_critical": is_critical,
                }, timeout=60)
                if resp.status_code == 200:
                    ev = resp.json()
                    if ev.get("message"):
                        st.info(ev["message"])
                    elif ev.get("queue_reordered"):
                        st.warning(f"Queue re-sorted: {ev['old_score']:.0f}% → {ev['new_score']:.0f}% (Δ{ev['delta']:+.0f})")
                    else:
                        st.info(f"Score updated: {ev.get('old_score', 0):.0f}% → {ev.get('new_score', 0):.0f}%")
                    st.rerun()
                else:
                    st.warning(f"Server returned {resp.status_code}: {resp.text[:200]}")
            except requests.exceptions.ConnectionError:
                st.warning("API server not running.")
            except Exception as exc:
                st.warning(f"Error: {exc}")
    else:
        st.info("No patients in queue to inject results for.")
