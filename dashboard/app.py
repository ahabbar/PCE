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
from src.core.lab_values import LAB_REFERENCE, ORDER_TO_KEY
from src.agents.llm import get_llm_client
from src.agents.triage_score import run_triage_score
from src.agents.red_flag import run_red_flag_guardian

API_URL = os.environ.get("PCE_API_URL", "http://localhost:8000")
# Browser-facing API URL — used for iframes, must be reachable from the user's
# browser. On Railway, set this to the API service's public URL when the
# dashboard talks to the API over the private network (PCE_API_URL).
PUBLIC_API_URL = os.environ.get("PCE_PUBLIC_API_URL", API_URL)

st.set_page_config(page_title="PCE — Parallel Care Engine", layout="wide")


@st.cache_data(ttl=2, show_spinner=False)
def _api_get_json(path: str, timeout: int = 10):
    """Cached GET to dedupe identical requests within a 2s window across reruns."""
    try:
        r = requests.get(f"{API_URL}{path}", timeout=timeout)
        return r.status_code, r.json() if r.status_code == 200 else None
    except Exception as exc:
        return 0, {"error": str(exc)}

# ── Dashboard styling: compact, with per-tab internal scroll containers ──────
st.markdown("""
<style>
  .block-container,
  [data-testid="block-container"],
  [data-testid="stMainBlockContainer"] {
      padding-top: 0.6rem !important;
      padding-bottom: 0.6rem !important;
      max-width: 100% !important;
  }
  h1 { font-size: 1.5rem !important; margin: 0 0 4px 0 !important; }
  h2 { font-size: 1.2rem !important; margin: 6px 0 !important; }
  h3 { font-size: 1.0rem !important; margin: 6px 0 !important; }
  [data-testid="stMetricValue"] { font-size: 1.15rem !important; }
  [data-testid="stMetricLabel"] { font-size: 0.75rem !important; }
  hr { margin: 0.4rem 0 !important; }
  .stTabs [data-baseweb="tab-list"] { gap: 4px !important; margin-bottom: 6px !important; }
  .stTabs [data-baseweb="tab"] { padding: 4px 12px !important; }
</style>
""", unsafe_allow_html=True)

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

def _refetch_labs(pid: str) -> None:
    try:
        r = requests.get(f"{API_URL}/queue/{pid}/labs", timeout=10)
        if r.status_code == 200:
            st.session_state[f"labs_{pid}"] = r.json().get("orders", [])
    except Exception:
        pass


@st.fragment
def _render_patient_card(pt: dict) -> None:
    """Live Queue card — Actions + read-only lab results only (no data entry)."""
    patient_id = pt["patient_id"]

    # ── Read-only lab results summary ────────────────────────────────
    if f"labs_{patient_id}" not in st.session_state:
        _refetch_labs(patient_id)
    orders = st.session_state.get(f"labs_{patient_id}", [])
    resulted = [o for o in orders if o.get("status") == "resulted"]
    approved_count = sum(1 for o in orders if o.get("status") == "approved")
    pending_count = sum(1 for o in orders if o.get("status") == "pending_approval")

    col_ref, col_btn = st.columns([6, 1])
    with col_ref:
        if resulted:
            parts = []
            for o in resulted:
                rv = (o.get("result_value") or "").split(" | ")[0]
                is_crit = "CRITICAL" in (o.get("result_value") or "").upper()
                is_abn  = any(w in (o.get("result_value") or "").lower()
                               for w in ("high", "low", "abnormal", "elevated"))
                icon = "🔴" if is_crit else "🟡" if is_abn else "🟢"
                parts.append(f"{icon} **{o['test_name']}**: {rv}")
            st.markdown("  ·  ".join(parts))
        else:
            tags = []
            if pending_count:
                tags.append(f"⏳ {pending_count} awaiting nurse approval")
            if approved_count:
                tags.append(f"🔬 {approved_count} in lab")
            if not tags:
                tags = ["No investigations yet"]
            st.caption("  ·  ".join(tags))
    if col_btn.button("↻", key=f"ref_q_{patient_id}", help="Refresh labs"):
        _refetch_labs(patient_id)
        st.rerun()

    with st.expander("Actions", expanded=False):
        col_assign, col_exit = st.columns(2)

        with col_assign:
            st.markdown("**Doctor Assignment**")
            doc = pt.get("assigned_doctor")
            # Derive count from actual resulted labs, not stale in-memory counter
            actual_resulted = len(resulted)
            has_critical = any("CRITICAL" in (o.get("result_value") or "").upper() for o in resulted)
            if doc:
                st.success(f"👨‍⚕️ **{doc}** — auto-assigned by Agent 5")
            else:
                if actual_resulted == 0:
                    st.caption("Waiting for first lab result before assignment")
                elif actual_resulted >= 2 or has_critical:
                    # Should already be assigned — auto-trigger to recover from sync issues
                    st.warning(
                        f"{actual_resulted} result(s){' (CRITICAL)' if has_critical else ''}"
                        " — assignment overdue. Click below to assign now.")
                else:
                    st.info(f"Agent 5 auto-assigns after 2nd result ({actual_resulted} received)")
                if st.button("Assign Doctor Now", key=f"manual_assign_{patient_id}",
                             type="primary" if actual_resulted >= 2 or has_critical else "secondary",
                             use_container_width=True):
                    try:
                        ar = requests.post(f"{API_URL}/assign/{patient_id}",
                                           json={"esi_level": pt["esi_level"]}, timeout=10)
                        if ar.status_code == 200:
                            d = ar.json()
                            if d.get("doctor_name"):
                                st.success(f"Assigned: {d.get('doctor_name')}")
                            else:
                                st.warning("All doctors at capacity — try again shortly.")
                            st.rerun(scope="app")
                        else:
                            st.warning(f"Error {ar.status_code}")
                    except Exception as exc:
                        st.warning(f"Error: {exc}")

            st.divider()
            st.markdown("**Quick Discharge**")
            qd_disp = st.selectbox("Outcome", ["discharge", "admit", "icu", "transfer"],
                                   key=f"qd_disp_{patient_id}")
            qd_diag = st.text_input("Diagnosis", key=f"qd_diag_{patient_id}",
                                    placeholder="e.g. UTI — treated")
            if st.button("Discharge / Complete", key=f"qd_{patient_id}",
                         type="primary", use_container_width=True):
                if qd_diag:
                    try:
                        dr = requests.post(
                            f"{API_URL}/queue/{patient_id}/discharge",
                            json={"disposition": qd_disp,
                                  "confirmed_diagnosis": qd_diag,
                                  "doctor_name": pt.get("assigned_doctor", "")},
                            timeout=10,
                        )
                        if dr.status_code == 200:
                            st.success(f"Patient discharged ({qd_disp})")
                            st.rerun(scope="app")
                        else:
                            st.warning(f"Error {dr.status_code}")
                    except Exception as exc:
                        st.warning(f"Error: {exc}")
                else:
                    st.warning("Enter diagnosis before discharging.")

        with col_exit:
            st.markdown("**Generate Exit Plan (AI)**")
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
                st.caption("Generates instructions/handover note then discharges patient.")
                if st.button("Generate Exit Plan", key=f"exit_{patient_id}",
                             use_container_width=True):
                    if diagnosis:
                        try:
                            with st.spinner("Agent 7 generating exit plan..."):
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
                st.success("Exit plan generated — patient discharged from queue.")
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

        # ── Nurse: Approve / Reject AI-suggested orders ──────────────
        st.divider()
        st.markdown("**AI-Suggested Investigations — Nurse Approval**")
        labs_key = f"labs_{patient_id}"
        if labs_key not in st.session_state:
            _refetch_labs(patient_id)
        pending_orders = [o for o in st.session_state.get(labs_key, [])
                          if o.get("status") == "pending_approval"]
        if pending_orders:
            pending_names = [o["test_name"] for o in pending_orders]
            selected = st.multiselect(
                "Investigations",
                options=pending_names,
                default=pending_names,
                key=f"sel_{patient_id}",
                label_visibility="collapsed",
            )
            c_approve, c_reject = st.columns(2)
            if c_approve.button("✓ Approve Selected", key=f"bapprove_{patient_id}",
                                type="primary", use_container_width=True):
                if selected:
                    for tname in selected:
                        try:
                            requests.post(f"{API_URL}/queue/{patient_id}/labs/confirm",
                                          json={"test_name": tname, "action": "approved"}, timeout=5)
                        except Exception as e:
                            st.warning(str(e))
                    _refetch_labs(patient_id)
                    st.rerun()
            if c_reject.button("✗ Reject Selected", key=f"breject_{patient_id}",
                               use_container_width=True):
                if selected:
                    for tname in selected:
                        try:
                            requests.post(f"{API_URL}/queue/{patient_id}/labs/confirm",
                                          json={"test_name": tname, "action": "rejected"}, timeout=5)
                        except Exception as e:
                            st.warning(str(e))
                    _refetch_labs(patient_id)
                    st.rerun()
        else:
            st.caption("✓ All investigations approved — see Lab Results tab for entry.")


_PANEL_LABELS = {
    "CBC":                   "CBC — Full Blood Count",
    "BMP":                   "BMP — Basic Metabolic Panel",
    "LFTs":                  "LFTs — Liver Function Tests",
    "Coagulation (PT/APTT)": "Coagulation Screen (PT/APTT)",
    "Blood gas (ABG/VBG)":   "Blood Gas (ABG/VBG)",
}


@st.fragment
def _render_lab_tech_card(pt: dict) -> None:
    """Lab Results tab — all ordered tests with result entry fields."""
    from src.core.lab_values import PANEL_EXPANSION as _PE

    patient_id = pt["patient_id"]
    labs_key = f"labtech_{patient_id}"

    if labs_key not in st.session_state:
        try:
            r = requests.get(f"{API_URL}/queue/{patient_id}/labs", timeout=10)
            if r.status_code == 200:
                st.session_state[labs_key] = r.json().get("orders", [])
        except Exception:
            st.session_state[labs_key] = []

    hdr_col, btn_col = st.columns([6, 1])
    hdr_col.markdown(
        f"**{patient_id}** — {pt['chief_complaint'][:55]} "
        f"| {pt['age']:.0f}{pt['gender'][0].upper()} | ESI-{pt['esi_level']}"
    )
    if btn_col.button("↻", key=f"reflt_{patient_id}", help="Refresh"):
        del st.session_state[labs_key]
        st.rerun()

    orders = st.session_state.get(labs_key, [])
    # Only show nurse-approved orders (and already-resulted ones, read-only).
    # Pending and rejected are excluded — labs are entered after approval in Live Queue.
    active = [o for o in orders if o.get("status") in ("approved", "resulted")]
    if not active:
        st.caption("No approved orders yet — nurse must confirm investigations in Live Queue first.")
        return

    # Build panel reverse map
    _comp_to_panel: dict[str, str] = {}
    for _pname, _comps in _PE.items():
        for _c in _comps:
            _comp_to_panel[_c] = _pname

    # Group by panel
    _pg: dict[str, list] = {}
    _po: list[str] = []
    _ug: list = []
    for _o in active:
        _p = _comp_to_panel.get(_o["test_name"])
        if _p:
            if _p not in _pg:
                _pg[_p] = []
                _po.append(_p)
            _pg[_p].append(_o)
        else:
            _ug.append(_o)

    def _result_row(order):
        tname = order["test_name"]
        ostatus = order.get("status", "approved")
        result_val = order.get("result_value", "") or ""
        sk = tname.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")

        if ostatus == "resulted":
            flag = "critical" if "CRITICAL" in result_val.upper() else \
                   "abnormal" if any(w in result_val.lower() for w in
                                     ("high", "low", "abnormal", "elevated")) else "normal"
            col = {"normal": "#22c55e", "abnormal": "#f97316", "critical": "#ef4444"}.get(flag, "#64748b")
            disp = result_val.split(" | ")[0] if " | " in result_val else result_val
            note = result_val.split(" | ")[1] if " | " in result_val else ""
            st.markdown(
                f"<div style='padding:4px 8px;border-left:3px solid {col};margin:2px 0'>"
                f"<b style='color:{col}'>{tname}</b> — {disp}"
                + (f"<br><span style='font-size:11px;color:{col}'>{note}</span>" if note else "")
                + "</div>", unsafe_allow_html=True)
        else:
            # approved or pending_approval — show input field
            _ref_info = LAB_REFERENCE.get(tname.lower())
            if _ref_info is None:
                _keys = ORDER_TO_KEY.get(tname, [])
                _ref_info = LAB_REFERENCE.get(_keys[0]) if _keys else None
            _ref_hint = f"ref: {_ref_info['range']} {_ref_info['unit']}" if _ref_info else ""
            _label = f"{tname}  —  {_ref_hint}" if _ref_hint else tname
            c1, c2 = st.columns([4, 1])
            val = c1.text_input(_label, key=f"lt_{patient_id}_{sk}",
                                placeholder="Enter result value")
            if c2.button("Submit", key=f"ltsub_{patient_id}_{sk}", use_container_width=True):
                if val:
                    try:
                        rr = requests.post(f"{API_URL}/queue/{patient_id}/labs/result",
                                           json={"test_name": tname, "value": val}, timeout=60)
                        if rr.status_code == 200:
                            rv = rr.json()
                            if rv.get("is_critical"):
                                st.error(f"CRITICAL — {rv.get('interpretation', '')}")
                            elif rv.get("is_abnormal"):
                                st.warning(f"Abnormal — {rv.get('interpretation', '')}")
                            else:
                                st.success(f"✓ {rv.get('interpretation', 'Result saved')}")
                            if rv.get("score_delta") is not None:
                                st.info(f"Risk → {rv['new_score']:.0f}% (Δ{rv['score_delta']:+.1f})")
                            # Invalidate BOTH caches so Live Queue picks up new result
                            st.session_state.pop(labs_key, None)
                            st.session_state.pop(f"labs_{patient_id}", None)
                            st.rerun(scope="app")
                        else:
                            st.warning(f"Error {rr.status_code}: {rr.text[:80]}")
                    except Exception as e:
                        st.warning(str(e))
                else:
                    st.warning("Enter a value first.")

    # Render panel sections
    for _pname in _po:
        _plabel = _PANEL_LABELS.get(_pname, _pname)
        _porders = _pg[_pname]
        _done = sum(1 for o in _porders if o.get("status") == "resulted")
        _badge = f"{_done}/{len(_porders)} resulted" if _done else f"{len(_porders)} tests"
        st.markdown(
            f"<div style='background:#1e293b;color:#94a3b8;padding:4px 10px;"
            f"border-radius:4px;margin:10px 0 4px;font-size:13px;font-weight:600'>"
            f"📋 {_plabel} &nbsp;<span style='font-weight:400'>({_badge})</span></div>",
            unsafe_allow_html=True)
        for _o in _porders:
            st.markdown(
                "<div style='margin-left:14px;border-left:2px solid #334155;padding-left:8px'>",
                unsafe_allow_html=True)
            _result_row(_o)
            st.markdown("</div>", unsafe_allow_html=True)

    # Standalone tests (ECG, CXR, BNP, etc.)
    if _ug:
        st.markdown(
            "<div style='background:#1e293b;color:#94a3b8;padding:4px 10px;"
            "border-radius:4px;margin:10px 0 4px;font-size:13px;font-weight:600'>"
            "🔬 Other Investigations</div>",
            unsafe_allow_html=True)
        for _o in _ug:
            _result_row(_o)


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Triage Input", "Live Queue", "Lab Results", "Demo Mode", "Analytics", "ED Flow"])

with st.sidebar:
    with st.expander("Demo Controls"):
        if st.button("Seed Demo Patients", key="seed_btn"):
            try:
                requests.post(f"{API_URL}/demo/seed", timeout=10)
            except Exception as exc:
                st.warning(f"Seed error: {exc}")
            # Seed protocol-specific labs from whitelist (bypasses stale server cache)
            try:
                import asyncio as _aio
                from src.core.whitelist import ProtocolWhitelist as _WL
                from src.database.db import seed_lab_orders as _slo, _resolve_path
                import aiosqlite as _asql
                _DEMO_CCS = {
                    "DEMO-001": "chest pain and diaphoresis",
                    "DEMO-002": "confusion and weakness",
                    "DEMO-003": "shortness of breath",
                    "DEMO-004": "fever and flank pain",
                    "DEMO-005": "mild dysuria",
                }
                async def _seed_whitelist():
                    _wl = _WL.from_env()
                    _path = _resolve_path(None)
                    for _pid, _cc in _DEMO_CCS.items():
                        _proto = _wl.match(_cc) or "general_medical"
                        _tests = [o["test"] for o in _wl.get_auto_orders(_proto)]
                        async with _asql.connect(_path) as _db:
                            await _db.execute(
                                "DELETE FROM investigations WHERE patient_id=?", (_pid,))
                            await _db.commit()
                        await _slo(_pid, _tests)
                _aio.run(_seed_whitelist())
                st.success("5 demo patients loaded — protocol-specific labs seeded")
            except Exception as exc:
                st.warning(f"Lab seed error: {exc}")
            st.rerun()
        if st.button("Reset Demo", key="reset_btn"):
            try:
                requests.post(f"{API_URL}/demo/reset", timeout=10)
                st.success("Demo reset")
                st.rerun()
            except Exception as exc:
                st.warning(f"Error: {exc}")

# ── Tab 1: Triage Input ───────────────────────────────────────────────────────

with tab1:
    left, right = st.columns([4, 6])

    with left, st.container(height=560, border=False):
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

    with right, st.container(height=560, border=False):
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

            try:
                pid_resp = requests.get(f"{API_URL}/next-patient-id", timeout=5)
                new_pid = pid_resp.json()["patient_id"] if pid_resp.status_code == 200 else f"PAT-{uuid.uuid4().hex[:5].upper()}"
            except Exception:
                new_pid = f"PAT-{uuid.uuid4().hex[:5].upper()}"

            intake = IntakeForm(
                patient_id=new_pid, age_years=float(age), gender=gender.lower(),
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
                            # Seed protocol-specific labs directly from workup result
                            _pid = d.get("intake_json", {}).get("patient_id", "")
                            _workup = d.get("workup_orders", [])
                            if _pid and _workup:
                                import asyncio as _aio
                                from src.database.db import seed_lab_orders as _slo, _resolve_path
                                import aiosqlite as _asql
                                async def _seed():
                                    _path = _resolve_path(None)
                                    async with _asql.connect(_path) as _db:
                                        await _db.execute(
                                            "DELETE FROM investigations WHERE patient_id=?", (_pid,))
                                        await _db.commit()
                                    await _slo(_pid, [o["test_name"] for o in _workup])
                                _aio.run(_seed())
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

def _render_live_queue_tab() -> None:
    st.subheader("Live Patient Queue")
    st.error(
        "ESI-1 (Immediate) patients are directed to the Resuscitation Bay and EXCLUDED from this queue. "
        "PCE does not delay emergency resuscitation.",
        icon="🚨",
    )
    if st.button("Refresh", key="queue_refresh"):
        st.cache_data.clear()
        st.rerun()

    status, queue_data = _api_get_json("/queue", timeout=10)
    if status != 200 or queue_data is None:
        if status == 0:
            st.warning(f"API unreachable: {(queue_data or {}).get('error', 'connection error')}")
        else:
            st.warning(f"Queue API returned {status}")
        return

    stat1, stat2, stat3 = st.columns(3)
    stat1.metric("Total Patients", queue_data["total"])
    stat2.metric("Red Alerts", queue_data["reds_count"])
    stat3.metric("Queue Depth", queue_data["queue_depth"])
    st.divider()

    patients = queue_data.get("patients", [])
    if not patients:
        st.info("No patients in queue.")
    else:
        pid_list = [pt["patient_id"] for pt in patients]
        sel_key = "queue_selected_pid"
        if st.session_state.get(sel_key) not in pid_list:
            st.session_state[sel_key] = pid_list[0]

        list_col, detail_col = st.columns([1, 3])
        with list_col:
            st.markdown("**Patients**")
            with st.container(height=520, border=False):
                for pt in patients:
                    score = pt["risk_score"]
                    bg = "#ef4444" if score >= 90 else "#f97316" if score >= 75 else "#eab308" if score >= 50 else "#22c55e"
                    is_sel = st.session_state[sel_key] == pt["patient_id"]
                    border = "3px solid #fff" if is_sel else "1px solid transparent"
                    st.markdown(
                        f"<div style='background:{bg};color:white;padding:8px 10px;"
                        f"border-radius:6px;margin-bottom:4px;border:{border};font-size:12px'>"
                        f"<b>#{pt['position']} ESI-{pt['esi_level']} · {score:.0f}%</b><br>"
                        f"{pt['age']:.0f}{pt['gender'][0].upper()} · {pt['chief_complaint'][:32]}</div>",
                        unsafe_allow_html=True,
                    )
                    if st.button("Open" if not is_sel else "● Selected",
                                 key=f"qsel_{pt['patient_id']}",
                                 use_container_width=True,
                                 disabled=is_sel):
                        st.session_state[sel_key] = pt["patient_id"]
                        st.rerun()

        with detail_col:
            pt = next((p for p in patients if p["patient_id"] == st.session_state[sel_key]), patients[0])
            score = pt["risk_score"]
            bg = "#ef4444" if score >= 90 else "#f97316" if score >= 75 else "#eab308" if score >= 50 else "#22c55e"
            wait = pt["wait_minutes"]
            wait_str = f"{wait:.0f} min" if wait < 60 else f"{wait/60:.1f} hr"
            st.markdown(
                f"<div style='background:{bg};color:white;padding:12px 16px;"
                f"border-radius:8px;margin-bottom:8px'>"
                f"<b>#{pt['position']} &nbsp; {pt['patient_id']} &nbsp; Score: {score:.0f}% &nbsp; ESI-{pt['esi_level']}</b>"
                f" &nbsp;|&nbsp; {pt['age']:.0f}{pt['gender'][0].upper()}"
                f"<br><span style='font-size:14px'>{pt['chief_complaint']}</span>"
                f"<br><span style='font-size:12px;opacity:0.85'>Wait: {wait_str}"
                f" &nbsp;|&nbsp; Status: {pt['status']}</span></div>",
                unsafe_allow_html=True,
            )
            with st.container(height=520, border=False):
                _render_patient_card(pt)

    batch_summary = queue_data.get("batch_summary", {})
    if batch_summary:
        st.divider()
        st.markdown("**Batch Summary** (tests shared by 2+ patients)")
        for test, count in batch_summary.items():
            st.markdown(f"- {test}: {count} patients")

    with st.expander("Doctor Availability", expanded=False):
        dr_status, dr_data = _api_get_json("/doctors", timeout=5)
        if dr_status == 200 and dr_data:
            doctors = dr_data.get("doctors", [])
            if doctors:
                assigned_doctors = {
                    pt["assigned_doctor"]
                    for pt in patients
                    if pt.get("assigned_doctor")
                }
                cols = st.columns(min(len(doctors), 5))
                for i, d in enumerate(doctors):
                    busy = d["name"] in assigned_doctors
                    role_label = d["role"].replace("_", " ").title()
                    with cols[i % 5]:
                        if busy:
                            pt_name = next(
                                (pt["patient_id"] for pt in patients
                                 if pt.get("assigned_doctor") == d["name"]), ""
                            )
                            st.error(f"🔴 **{d['name']}**\n\n{pt_name}")
                        else:
                            st.success(f"🟢 **{d['name']}**\n\nAvailable")
                        st.caption(role_label)
            else:
                st.info("No doctors configured.")
        elif dr_status == 0:
            st.warning("API server offline — doctor load unavailable.")
        else:
            st.warning(f"Could not fetch doctor data ({dr_status}).")


with tab2:
    _render_live_queue_tab()


# ── Tab 3: Lab Results ───────────────────────────────────────────────────────

def _render_lab_results_tab() -> None:
    st.subheader("Lab Results Entry")
    st.caption("Laboratory technician workspace — enter results for approved investigations")

    if st.button("Refresh All", key="lab_refresh_all"):
        for k in list(st.session_state.keys()):
            if k.startswith("labtech_"):
                del st.session_state[k]
        st.cache_data.clear()
        st.rerun()

    lq_status, lq_data = _api_get_json("/queue", timeout=10)
    lab_patients = lq_data.get("patients", []) if (lq_status == 200 and lq_data) else []

    if not lab_patients:
        st.info("No patients in queue.")
        return

    pid_list = [pt["patient_id"] for pt in lab_patients]
    sel_key = "lab_selected_pid"
    if st.session_state.get(sel_key) not in pid_list:
        st.session_state[sel_key] = pid_list[0]

    list_col, detail_col = st.columns([1, 3])
    with list_col:
        st.markdown("**Patients**")
        with st.container(height=580, border=False):
            for lpt in lab_patients:
                score = lpt["risk_score"]
                bg = "#ef4444" if score >= 90 else "#f97316" if score >= 75 else "#eab308" if score >= 50 else "#22c55e"
                is_sel = st.session_state[sel_key] == lpt["patient_id"]
                border = "3px solid #fff" if is_sel else "1px solid transparent"
                st.markdown(
                    f"<div style='background:{bg};color:white;padding:8px 10px;"
                    f"border-radius:6px;margin-bottom:4px;border:{border};font-size:12px'>"
                    f"<b>#{lpt['position']} ESI-{lpt['esi_level']} · {score:.0f}%</b><br>"
                    f"{lpt['age']:.0f}{lpt['gender'][0].upper()} · {lpt['chief_complaint'][:32]}</div>",
                    unsafe_allow_html=True,
                )
                if st.button("Open" if not is_sel else "● Selected",
                             key=f"lsel_{lpt['patient_id']}",
                             use_container_width=True,
                             disabled=is_sel):
                    st.session_state[sel_key] = lpt["patient_id"]
                    st.rerun()

    with detail_col:
        lpt = next((p for p in lab_patients if p["patient_id"] == st.session_state[sel_key]), lab_patients[0])
        score = lpt["risk_score"]
        bg = "#ef4444" if score >= 90 else "#f97316" if score >= 75 else "#eab308" if score >= 50 else "#22c55e"
        st.markdown(
            f"<div style='background:{bg};color:white;padding:8px 14px;"
            f"border-radius:6px;margin-bottom:8px'>"
            f"<b>#{lpt['position']} &nbsp; {lpt['patient_id']} &nbsp; ESI-{lpt['esi_level']} &nbsp; Score: {score:.0f}%</b>"
            f" &nbsp;|&nbsp; {lpt['age']:.0f}{lpt['gender'][0].upper()}"
            f" &nbsp;|&nbsp; {lpt['chief_complaint']}</div>",
            unsafe_allow_html=True)
        with st.container(height=580, border=False):
            _render_lab_tech_card(lpt)


with tab3:
    _render_lab_results_tab()


# ── Tab 4: Demo Mode ──────────────────────────────────────────────────────────

with tab4:
    st.subheader("Live Patient Triage")
    st.caption("Enter any chief complaint — all 7 agents respond in parallel")

    left3, right3 = st.columns([4, 6])

    with left3, st.container(height=540, border=False):
        demo_cc = st.text_input("Chief Complaint",
            placeholder="e.g. chest pain and diaphoresis, confusion, fever and flank pain...",
            key="demo_cc")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            demo_age    = st.number_input("Age", 1, 120, 55, key="demo_age")
            demo_gender = st.selectbox("Gender", ["male", "female", "other"], key="demo_gender")
        with col_b:
            demo_hr   = st.number_input("HR",      40, 200, 88, key="demo_hr")
            demo_temp = st.number_input("Temp °C", 35.0, 42.0, 38.2, step=0.1, key="demo_temp")
        with col_c:
            demo_sbp  = st.number_input("BP Sys",  60, 220, 118, key="demo_sbp")
            demo_spo2 = st.number_input("SpO2 %",  70, 100, 97,  key="demo_spo2")
        demo_comorbidities = st.text_input("Comorbidities (optional)",
            placeholder="e.g. CKD stage 2, DM, immunosuppressed...", key="demo_comorbidities")
        demo_analyze = st.button("Analyze — Run All 7 Agents", type="primary",
                                  key="demo_analyze", use_container_width=True)

    with right3, st.container(height=540, border=False):
        if demo_analyze and demo_cc:
            comorbidities = [c.strip() for c in demo_comorbidities.split(",") if c.strip()] if demo_comorbidities else []
            try:
                pid_r = requests.get(f"{API_URL}/next-patient-id", timeout=5)
                demo_pid = pid_r.json()["patient_id"] if pid_r.status_code == 200 else f"DEMO-{uuid.uuid4().hex[:6].upper()}"
            except Exception:
                demo_pid = f"DEMO-{uuid.uuid4().hex[:6].upper()}"

            payload = {
                "patient_id": demo_pid, "age_years": float(demo_age), "gender": demo_gender,
                "chief_complaint": {"free_text_en": demo_cc, "category": "other",
                                    "pain_present": "pain" in demo_cc.lower()},
                "vitals": {"heart_rate": float(demo_hr), "temperature_c": float(demo_temp),
                           "systolic_bp": float(demo_sbp), "spo2_pct": float(demo_spo2)},
                "medical_history": {"known_diagnoses": comorbidities},
            }
            t0 = time.time()
            with st.spinner("Agents 1+2 running in parallel..."):
                try:
                    resp = requests.post(f"{API_URL}/triage", json=payload, timeout=30)
                    elapsed = time.time() - t0
                    if resp.status_code == 200:
                        st.session_state["demo_result"] = resp.json()
                        st.session_state["demo_result"]["_elapsed"] = elapsed
                        st.session_state["demo_result"]["_pid"] = demo_pid
                    elif resp.status_code == 422:
                        st.error(f"Payload error: {resp.json()}")
                    else:
                        st.error(f"Server error {resp.status_code}: {resp.text[:200]}")
                except requests.exceptions.ConnectionError:
                    st.error("API server not running. Start with: uvicorn src.api.server:app --port 8000")
                except Exception as exc:
                    elapsed = time.time() - t0
                    if elapsed > 20:
                        st.warning("LLM response slow — check API keys")
                    else:
                        st.error(f"Error: {exc}")
        elif demo_analyze and not demo_cc:
            st.warning("Enter a chief complaint first.")

        if "demo_result" in st.session_state:
            dr = st.session_state["demo_result"]
            rs = dr.get("risk_score", 0)
            score_color = "#ef4444" if rs >= 90 else "#f97316" if rs >= 75 else "#eab308" if rs >= 50 else "#22c55e"
            is_red = dr.get("is_red", False)
            zone_label = "🔴 RED ZONE" if is_red else "✓ Not Red"

            st.markdown(
                f"<div style='background:{score_color};color:white;padding:20px;border-radius:10px;text-align:center'>"
                f"<div style='font-size:72px;font-weight:900;line-height:1'>{rs:.0f}<span style='font-size:32px'>%</span></div>"
                f"<div style='font-size:16px;margin-top:4px'>ESI-{dr.get('esi_level')} &nbsp;|&nbsp; "
                f"Threshold: {dr.get('threshold_used', 85):.0f}% &nbsp;|&nbsp; {zone_label}</div>"
                f"</div>", unsafe_allow_html=True)
            st.markdown("")

            # Key factors
            factors = dr.get("key_factors", [])
            if factors:
                pills = " ".join(f"`{f}`" for f in factors)
                st.markdown(f"**Key Factors:** {pills}")

            # Agent status row
            orders = dr.get("workup_orders", [])
            flag_type = dr.get("flag_type") or ("CLEAR" if not dr.get("is_emergency") else "EMERGENCY")
            c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
            c1.metric("Agent 1", f"{rs:.0f}%", "Risk Score")
            c2.metric("Agent 2", flag_type[:8], "Red Flag")
            c3.metric("Agent 3", f"{len(orders)} orders", "Workup")
            c4.metric("Agent 4", "pending", "Batch")
            c5.metric("Agent 5", "deferred", "Load Bal.")
            c6.metric("Agent 6", "deferred", "Disposition")
            c7.metric("Agent 7", "deferred", "Exit Coord.")
            st.caption("Agents 5–7 fire automatically after lab results arrive.")

            # Workup table
            if orders:
                st.markdown("**Workup Plan** — validated against Protocol Whitelist")
                tbl = {"Test": [], "Reason": [], "Cost": [], "Timing": []}
                for o in orders:
                    tbl["Test"].append(o["test_name"])
                    tbl["Reason"].append(o["reason"])
                    tbl["Cost"].append(o["cost_tier"])
                    tbl["Timing"].append(o["timing"])
                st.dataframe(tbl, use_container_width=True)

            deferred = dr.get("deferred_to_doctor", [])
            if deferred:
                with st.expander("Deferred to doctor"):
                    for item in deferred:
                        st.markdown(f"- {item}")

            st.info(dr.get("reasoning", ""))

            if dr.get("parallel_confirmed"):
                st.success("Agents 1 and 2 ran simultaneously (asyncio.gather confirmed)")
            else:
                st.warning("Agents ran sequentially — check orchestrator")

            elapsed = dr.get("_elapsed", dr.get("total_latency_ms", 0) / 1000)
            st.caption(f"Analyzed in {elapsed:.1f}s | Model: Gemini 2.5 Flash")

            act1, act2 = st.columns(2)
            with act1:
                if not dr.get("_queued"):
                    if st.button("Add to Queue", key="demo_add_queue"):
                        pid = dr.get("_pid", dr.get("patient_id", ""))
                        add_payload = {
                            "intake": {
                                "patient_id": pid,
                                "age_years": float(demo_age),
                                "gender": demo_gender,
                                "chief_complaint": {
                                    "free_text_en": dr.get("_cc", demo_cc or ""),
                                    "category": "other",
                                    "pain_present": False,
                                },
                                "vitals": {},
                                "medical_history": {},
                            },
                            "esi_level": dr.get("esi_level", 3),
                            "pce_scope": dr.get("esi_scope", "pce_core"),
                            "risk_score": rs,
                            "is_red": is_red,
                            "threshold_used": dr.get("threshold_used", 85.0),
                            "confidence": 0.85,
                            "key_factors": factors,
                            "reasoning": dr.get("reasoning", ""),
                            "is_emergency": dr.get("is_emergency", False),
                            "esi1_immediate": False,
                            "flag_type": dr.get("flag_type"),
                            "immediate_action": dr.get("immediate_action"),
                        }
                        try:
                            r = requests.post(f"{API_URL}/queue/add", json=add_payload, timeout=10)
                            if r.status_code == 200:
                                pos = r.json().get("position", "?")
                                st.session_state["demo_result"]["_queued"] = True
                                st.success(f"Added — Queue position: {pos}")
                                st.rerun()
                        except Exception as exc:
                            st.warning(f"Error: {exc}")
                else:
                    st.success("Added to queue")
            with act2:
                if st.button("Clear", key="demo_clear"):
                    del st.session_state["demo_result"]
                    st.rerun()

    # ── Real Case Validation ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Real Case Validation")
    st.caption("Real anonymised cases from our internal medicine ED")

    cl_status, cl_data = _api_get_json("/cases/list", timeout=5)
    cases_list = cl_data if (cl_status == 200 and isinstance(cl_data, list)) else []

    if not cases_list:
        st.info("No demo cases loaded. Add cases to data/cases/demo_cases.json")
    else:
        vc1, vc2 = st.columns([2, 1])
        with vc1:
            case_options = {c["chief_complaint"][:60]: c["id"] for c in cases_list}
            selected_cc_val = st.selectbox("Select a real case:", list(case_options.keys()), key="val_case_select")
            if st.button("Run PCE on this case", key="val_run"):
                case_id = case_options[selected_cc_val]
                with st.spinner("Running agents..."):
                    try:
                        vr = requests.post(f"{API_URL}/cases/run/{case_id}", timeout=60)
                        if vr.status_code == 200:
                            st.session_state["val_result"] = vr.json()
                        else:
                            st.warning(f"Error {vr.status_code}")
                    except Exception as exc:
                        st.warning(f"Error: {exc}")
        with vc2:
            if st.button("Validate All Cases", key="val_all_btn"):
                with st.spinner("Running all cases..."):
                    try:
                        va = requests.get(f"{API_URL}/cases/validate", timeout=300)
                        if va.status_code == 200:
                            st.session_state["val_all"] = va.json()
                    except Exception as exc:
                        st.warning(f"Error: {exc}")

        if "val_result" in st.session_state:
            vr = st.session_state["val_result"]
            vm1, vm2, vm3 = st.columns(3)
            vm1.metric("Agreement Rate", f"{vr['agreement_rate']*100:.0f}%")
            vm2.metric("Risk Score", f"{vr['proposed_score']:.0f}%")
            vm3.metric("ESI Level", vr["proposed_esi"])
            vd1, vd2 = st.columns(2)
            with vd1:
                st.markdown("**PCE Proposed**")
                for t in vr["proposed_workup"]:
                    icon = "✅" if t in vr["matched_tests"] else "➕"
                    st.markdown(f"{icon} {t}")
            with vd2:
                st.markdown("**Actually Ordered**")
                for t in vr["actual_workup"]:
                    icon = "✅" if t in vr["matched_tests"] else "⭕"
                    st.markdown(f"{icon} {t}")
            if vr.get("missed_tests"):
                st.warning(f"Missed by PCE: {', '.join(vr['missed_tests'])}")
            if vr.get("extra_tests"):
                st.info(f"PCE added (not in actual): {', '.join(vr['extra_tests'])}")
            disp_icon = "✅" if vr["disposition_match"] else "⚠️"
            st.markdown(f"**Diagnosis:** {vr['actual_diagnosis']}")
            st.markdown(f"**Actual disposition:** {vr['actual_disposition']} {disp_icon}")

        if "val_all" in st.session_state and isinstance(st.session_state["val_all"], dict):
            summary = st.session_state["val_all"].get("summary", {})
            if summary.get("total", 0) > 0:
                st.markdown("---")
                va1, va2, va3, va4 = st.columns(4)
                va1.metric("Cases Validated", summary["total"])
                va2.metric("Mean Agreement", f"{summary['mean_agreement_rate']*100:.0f}%")
                va3.metric("Disposition Accuracy", f"{summary['disposition_accuracy']*100:.0f}%")
                va4.metric("Avg Latency", f"{summary['mean_latency_ms']:.0f}ms")

# ── Tab 5: Analytics ──────────────────────────────────────────────────────────

with tab5, st.container(height=600, border=False):
    st.subheader("Session Analytics")
    if st.button("Refresh Analytics", key="analytics_refresh"):
        st.rerun()

    an_status, an = _api_get_json("/analytics", timeout=10)
    if an_status != 200:
        an = None

    if an:
        stats = an.get("session_stats", {})
        comp  = an.get("comparison", {})

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Patients Today", stats.get("total_patients_today", 0))
        s2.metric("Still Waiting",  stats.get("still_waiting", 0))
        red_n = stats.get("red_zone_count", 0)
        s3.metric("Red Zone", red_n, delta="⚠️" if red_n > 0 else None)
        s4.metric("Avg Wait", f"{stats.get('mean_wait_minutes_current', 0):.0f} min")
        s5.metric("Batch Efficiency", f"{stats.get('batch_efficiency_pct', 0):.0f}%")

        esi_dist = stats.get("esi_distribution", {})
        if any(v > 0 for v in esi_dist.values()):
            st.markdown("**ESI Distribution**")
            st.bar_chart({f"ESI-{k}": v for k, v in esi_dist.items()})

    st.divider()
    st.subheader("Traditional vs Parallel Care Engine")
    st.caption("Source: NHS England 2024-25 statistics + PCE projections based on nurse-initiated protocol literature")

    trad = {"seen_4hr_pct": 61.0, "median_los_min": 285, "doctor_min_per_pt": 43,
            "cost_per_pt_gbp": 95.69, "sepsis_to_abx_min": 142, "lwbs_pct": 5.1}
    pce_v = {"seen_4hr_pct": 87.0, "median_los_min": 144, "doctor_min_per_pt": 20,
             "cost_per_pt_gbp": 78.70, "sepsis_to_abx_min": 38, "lwbs_pct": 0.8}

    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**Traditional ED**")
        st.metric("Seen within 4 hours", f"{trad['seen_4hr_pct']:.0f}%")
        st.metric("Median ED stay",  f"{trad['median_los_min']//60}h {trad['median_los_min']%60}min")
        st.metric("Doctor time / patient", f"{trad['doctor_min_per_pt']} min")
        st.metric("Sepsis → antibiotics",  f"{trad['sepsis_to_abx_min']} min")
        st.metric("Left without being seen", f"{trad['lwbs_pct']}%")
        st.metric("Cost per patient", f"£{trad['cost_per_pt_gbp']:.2f}")
    with m2:
        st.markdown("**With PCE**")
        st.metric("Seen within 4 hours",   f"{pce_v['seen_4hr_pct']:.0f}%",   delta="+26%")
        st.metric("Median ED stay",  f"{pce_v['median_los_min']//60}h {pce_v['median_los_min']%60}min", delta="-2h 21min")
        st.metric("Doctor time / patient", f"{pce_v['doctor_min_per_pt']} min", delta="-23 min")
        st.metric("Sepsis → antibiotics",  f"{pce_v['sepsis_to_abx_min']} min", delta="-104 min")
        st.metric("Left without being seen", f"{pce_v['lwbs_pct']}%",           delta="-4.3%")
        st.metric("Cost per patient", f"£{pce_v['cost_per_pt_gbp']:.2f}",       delta="-£16.99")

    st.divider()
    st.subheader("Annual Economic Impact (250 patients/day)")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Direct savings",    "£1.55M",  delta="per year")
    e2.metric("Throughput gains",  "£3.56M",  delta="per year")
    e3.metric("Fewer admissions",  "£5.84M",  delta="per year")
    e4.metric("LWBS prevention",   "£0.78M",  delta="per year")
    st.metric("Total Annual Impact", "~£11.7M", delta="Break-even: 13 days")
    st.info(
        "NHS 2024-25 data. 61% = annual average Type 1 EDs. "
        "34% = January 2025 peak (worst month on record). "
        "PCE figures are projections based on nurse-initiated protocol literature."
    )

    st.divider()
    st.subheader("Time Savings per Encounter")
    svc_data = {
        "ESI Level":          ["ESI-2", "ESI-3", "ESI-4", "ESI-5"],
        "Traditional (min)":  [40,       28,       18,       12],
        "With PCE (min)":     [22,       15,       10,        7],
        "Saved (min)":        [18,       13,        8,        5],
    }
    st.dataframe(svc_data, use_container_width=True)
    st.caption("PCE savings come from results already available when doctor arrives")


with tab6:
    from pathlib import Path as _Path
    _ED_HTML_PATH = _Path(__file__).parent.parent / "src" / "api" / "ed_view.html"
    _ed_url = f"{PUBLIC_API_URL}/ed-view"
    _stream_url = f"{PUBLIC_API_URL}/ed-stream"

    st.markdown(
        "Live ED workflow — patients and AI agent activity stream from FastAPI via SSE. "
        "Particles fire on real status transitions; dots show actual occupancy in each zone."
    )

    try:
        _ed_html = _ED_HTML_PATH.read_text(encoding="utf-8")
        # Inject the absolute SSE URL so the embedded page knows where to
        # connect (it would otherwise default to a relative /ed-stream that
        # resolves against Streamlit's origin, where no such endpoint exists).
        _inject = f"<script>window.PCE_STREAM_URL='{_stream_url}';</script>"
        _ed_html = _ed_html.replace("</head>", _inject + "</head>", 1)
        st.components.v1.html(_ed_html, height=720, scrolling=False)
    except FileNotFoundError:
        st.error(f"ED view HTML not found at {_ED_HTML_PATH}")

    st.caption(
        f"Live API: [{_ed_url}]({_ed_url}) · stream: `{_stream_url}`. "
        "On Railway, set `PCE_PUBLIC_API_URL` on the dashboard service to the API's public HTTPS URL."
    )
