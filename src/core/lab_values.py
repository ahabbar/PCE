"""
Reference ranges and result interpretation for common ED laboratory tests.
Sources: PSAP (Pharmacotherapy Self-Assessment Program) + ACP Laboratory Values.
"""
from __future__ import annotations
import re


# ── Reference ranges ─────────────────────────────────────────────────────────
# Each entry: (low_normal, high_normal, critical_low, critical_high, unit, display_range)
# None = no critical threshold defined for that direction

LAB_REFERENCE: dict[str, dict] = {
    # Haematology — WBC total
    "wbc": dict(lo=4.5, hi=11.0, clo=2.0, chi=20.0, unit="×10³/µL",
                range="4.5–11.0", full="WBC (White Blood Cell Count)"),
    # WBC differential (absolute)
    "neutrophils": dict(lo=1.8, hi=7.7, clo=0.5, chi=None, unit="×10³/µL",
                        range="1.8–7.7", full="Neutrophils (Absolute)"),
    "neutro": dict(lo=1.8, hi=7.7, clo=0.5, chi=None, unit="×10³/µL",
                   range="1.8–7.7", full="Neutrophils (Absolute)"),
    "lymphocytes": dict(lo=1.0, hi=4.8, clo=0.5, chi=None, unit="×10³/µL",
                        range="1.0–4.8", full="Lymphocytes (Absolute)"),
    "lympho": dict(lo=1.0, hi=4.8, clo=0.5, chi=None, unit="×10³/µL",
                   range="1.0–4.8", full="Lymphocytes (Absolute)"),
    "monocytes": dict(lo=0.2, hi=0.8, clo=None, chi=None, unit="×10³/µL",
                      range="0.2–0.8", full="Monocytes (Absolute)"),
    "mono": dict(lo=0.2, hi=0.8, clo=None, chi=None, unit="×10³/µL",
                 range="0.2–0.8", full="Monocytes (Absolute)"),
    "eosinophils": dict(lo=0.0, hi=0.45, clo=None, chi=None, unit="×10³/µL",
                        range="0.0–0.45", full="Eosinophils (Absolute)"),
    "eos": dict(lo=0.0, hi=0.45, clo=None, chi=None, unit="×10³/µL",
                range="0.0–0.45", full="Eosinophils (Absolute)"),
    "basophils": dict(lo=0.0, hi=0.1, clo=None, chi=None, unit="×10³/µL",
                      range="0.0–0.10", full="Basophils (Absolute)"),
    "baso": dict(lo=0.0, hi=0.1, clo=None, chi=None, unit="×10³/µL",
                 range="0.0–0.10", full="Basophils (Absolute)"),
    # Red cell indices
    "rbc": dict(lo=4.0, hi=5.5, clo=None, chi=None, unit="×10⁶/µL",
                range="4.0–5.5", full="Red Blood Cell Count"),
    "hemoglobin": dict(lo=12.0, hi=17.0, clo=7.0, chi=None, unit="g/dL",
                       range="12–17", full="Hemoglobin"),
    "hgb": dict(lo=12.0, hi=17.0, clo=7.0, chi=None, unit="g/dL",
                range="12–17", full="Hemoglobin"),
    "hematocrit": dict(lo=36.0, hi=51.0, clo=21.0, chi=None, unit="%",
                       range="36–51", full="Hematocrit"),
    "hct": dict(lo=36.0, hi=51.0, clo=21.0, chi=None, unit="%",
                range="36–51", full="Hematocrit"),
    "mcv": dict(lo=80.0, hi=100.0, clo=None, chi=None, unit="fL",
                range="80–100", full="Mean Corpuscular Volume"),
    "mch": dict(lo=27.0, hi=33.0, clo=None, chi=None, unit="pg",
                range="27–33", full="Mean Corpuscular Haemoglobin"),
    "mchc": dict(lo=32.0, hi=36.0, clo=None, chi=None, unit="g/dL",
                 range="32–36", full="Mean Corpuscular Haemoglobin Concentration"),
    "rdw": dict(lo=11.5, hi=14.5, clo=None, chi=None, unit="%",
                range="11.5–14.5", full="Red Cell Distribution Width"),
    # Platelets
    "platelets": dict(lo=150.0, hi=350.0, clo=50.0, chi=None, unit="×10³/µL",
                      range="150–350", full="Platelet Count"),
    "plt": dict(lo=150.0, hi=350.0, clo=50.0, chi=None, unit="×10³/µL",
                range="150–350", full="Platelet Count"),
    "inr": dict(lo=0.9, hi=1.1, clo=None, chi=4.0, unit="",
                range="0.9–1.1", full="INR"),

    # Electrolytes & Renal
    "sodium": dict(lo=136.0, hi=145.0, clo=120.0, chi=160.0, unit="mEq/L",
                   range="136–145 mEq/L", full="Sodium"),
    "na": dict(lo=136.0, hi=145.0, clo=120.0, chi=160.0, unit="mEq/L",
               range="136–145 mEq/L", full="Sodium"),
    "potassium": dict(lo=3.5, hi=5.0, clo=2.5, chi=6.5, unit="mEq/L",
                      range="3.5–5.0 mEq/L", full="Potassium"),
    "k": dict(lo=3.5, hi=5.0, clo=2.5, chi=6.5, unit="mEq/L",
              range="3.5–5.0 mEq/L", full="Potassium"),
    "chloride": dict(lo=98.0, hi=106.0, clo=None, chi=None, unit="mEq/L",
                     range="98–106 mEq/L", full="Chloride"),
    "cl": dict(lo=98.0, hi=106.0, clo=None, chi=None, unit="mEq/L",
               range="98–106 mEq/L", full="Chloride"),
    "bicarbonate": dict(lo=23.0, hi=28.0, clo=10.0, chi=None, unit="mEq/L",
                        range="23–28 mEq/L", full="Bicarbonate"),
    "hco3": dict(lo=23.0, hi=28.0, clo=10.0, chi=None, unit="mEq/L",
                 range="23–28 mEq/L", full="Bicarbonate"),
    "bun": dict(lo=8.0, hi=20.0, clo=None, chi=None, unit="mg/dL",
                range="8–20 mg/dL", full="Blood Urea Nitrogen"),
    "creatinine": dict(lo=0.6, hi=1.3, clo=None, chi=3.0, unit="mg/dL",
                       range="0.6–1.3 mg/dL", full="Creatinine"),
    "scr": dict(lo=0.6, hi=1.3, clo=None, chi=3.0, unit="mg/dL",
                range="0.6–1.3 mg/dL", full="Creatinine"),
    "glucose": dict(lo=70.0, hi=110.0, clo=40.0, chi=500.0, unit="mg/dL",
                    range="70–110 mg/dL", full="Glucose"),

    # Cardiac markers
    "troponin i": dict(lo=0.0, hi=0.04, clo=None, chi=0.5, unit="ng/mL",
                       range="<0.04 ng/mL", full="Troponin I"),
    "troponin t": dict(lo=0.0, hi=0.01, clo=None, chi=0.1, unit="ng/mL",
                       range="<0.01 ng/mL", full="Troponin T"),
    "troponin": dict(lo=0.0, hi=0.04, clo=None, chi=0.5, unit="ng/mL",
                     range="<0.04 ng/mL", full="Troponin"),
    "bnp": dict(lo=0.0, hi=100.0, clo=None, chi=500.0, unit="pg/mL",
                range="<100 pg/mL", full="BNP (B-type Natriuretic Peptide)"),
    "ck": dict(lo=40.0, hi=170.0, clo=None, chi=None, unit="U/L",
               range="40–170 U/L", full="Creatine Kinase"),

    # Liver
    "alt": dict(lo=0.0, hi=35.0, clo=None, chi=None, unit="U/L",
                range="0–35 U/L", full="Alanine Aminotransferase (ALT)"),
    "ast": dict(lo=0.0, hi=35.0, clo=None, chi=None, unit="U/L",
                range="0–35 U/L", full="Aspartate Aminotransferase (AST)"),
    "bilirubin": dict(lo=0.3, hi=1.2, clo=None, chi=None, unit="mg/dL",
                      range="0.3–1.2 mg/dL", full="Total Bilirubin"),
    "lipase": dict(lo=0.0, hi=95.0, clo=None, chi=None, unit="U/L",
                   range="<95 U/L", full="Lipase"),
    "amylase": dict(lo=27.0, hi=131.0, clo=None, chi=None, unit="U/L",
                    range="27–131 U/L", full="Amylase"),

    # Infection markers
    "crp": dict(lo=0.0, hi=8.0, clo=None, chi=None, unit="mg/L",
                range="<8 mg/L", full="C-Reactive Protein"),
    "lactate": dict(lo=0.5, hi=2.0, clo=None, chi=4.0, unit="mmol/L",
                    range="0.5–2.0 mmol/L", full="Lactate"),
    "procalcitonin": dict(lo=0.0, hi=0.25, clo=None, chi=None, unit="ng/mL",
                          range="<0.25 ng/mL", full="Procalcitonin"),

    # Blood gas
    "ph": dict(lo=7.35, hi=7.45, clo=7.2, chi=7.6, unit="",
               range="7.35–7.45", full="Arterial pH"),
    "pco2": dict(lo=35.0, hi=45.0, clo=20.0, chi=70.0, unit="mmHg",
                 range="35–45 mmHg", full="pCO₂"),
    "po2": dict(lo=80.0, hi=100.0, clo=60.0, chi=None, unit="mmHg",
                range="80–100 mmHg", full="pO₂"),

    # Coagulation
    "pt": dict(lo=10.0, hi=13.0, clo=None, chi=None, unit="s",
               range="10–13 s", full="Prothrombin Time"),
    "ptt": dict(lo=25.0, hi=40.0, clo=None, chi=None, unit="s",
                range="25–40 s", full="Partial Thromboplastin Time"),
    "d-dimer": dict(lo=0.0, hi=0.5, clo=None, chi=None, unit="µg/mL",
                    range="<0.5 µg/mL", full="D-Dimer"),

    # Other
    "magnesium": dict(lo=1.5, hi=2.4, clo=None, chi=None, unit="mg/dL",
                      range="1.5–2.4 mg/dL", full="Magnesium"),
    "calcium": dict(lo=9.0, hi=10.5, clo=6.5, chi=13.0, unit="mg/dL",
                    range="9.0–10.5 mg/dL", full="Calcium"),
    "phosphorus": dict(lo=3.0, hi=4.5, clo=None, chi=None, unit="mg/dL",
                       range="3.0–4.5 mg/dL", full="Phosphorus"),
    "uric acid": dict(lo=2.5, hi=8.0, clo=None, chi=None, unit="mg/dL",
                      range="2.5–8.0 mg/dL", full="Uric Acid"),
    "albumin": dict(lo=3.5, hi=5.5, clo=None, chi=None, unit="g/dL",
                    range="3.5–5.5 g/dL", full="Albumin"),
}

# ── Protocol-based default orders by CC category ─────────────────────────────
CATEGORY_DEFAULT_ORDERS: dict[str, list[str]] = {
    "chest_pain":             ["ECG 12-lead", "Troponin POCT", "WBC", "HGB", "PLT", "BMP", "CXR"],
    "dyspnea":                ["SpO2 continuous", "ECG 12-lead", "CXR", "WBC", "HGB", "PLT", "BMP"],
    "fever_infection":        ["WBC", "HGB", "PLT", "CRP", "Urinalysis POCT", "Blood glucose"],
    "fever_infectious":       ["WBC", "HGB", "PLT", "CRP", "Urinalysis POCT", "Blood glucose"],
    "urinary":                ["Urinalysis POCT", "Urine culture", "WBC", "HGB", "PLT", "BMP", "CRP"],
    "uti_pyelonephritis":     ["Urinalysis POCT", "Urine culture", "WBC", "HGB", "PLT", "BMP", "CRP"],
    "abdominal_pain":         ["WBC", "HGB", "PLT", "BMP", "LFTs", "Lipase", "Urinalysis POCT", "CRP"],
    "altered_mental_status":  ["Glucose POCT", "ECG 12-lead", "WBC", "HGB", "PLT", "BMP", "Blood gas (ABG/VBG)", "Urinalysis POCT"],
    "altered_consciousness":  ["Glucose POCT", "ECG 12-lead", "WBC", "HGB", "PLT", "BMP", "Blood gas (ABG/VBG)", "Urinalysis POCT"],
    "syncope":                ["ECG 12-lead", "Glucose POCT", "WBC", "HGB", "PLT", "BMP"],
    "palpitations":           ["ECG 12-lead", "Glucose POCT", "WBC", "HGB", "PLT", "TFTs", "BMP"],
    "headache":               ["WBC", "HGB", "PLT", "BMP", "Glucose POCT"],
    "weakness_fatigue":       ["WBC", "HGB", "PLT", "BMP", "TFTs", "LFTs", "Glucose POCT", "CRP"],
    "gi_bleeding":            ["WBC", "HGB", "PLT", "Coagulation (PT/APTT)", "BMP", "LFTs"],
    "leg_swelling_dvt":       ["D-dimer", "WBC", "HGB", "PLT", "BMP", "BNP / NT-proBNP"],
    "hypertensive_emergency": ["ECG 12-lead", "WBC", "HGB", "PLT", "BMP", "Urinalysis POCT"],
    "diabetic_emergency":     ["Glucose POCT", "Blood gas (ABG/VBG)", "BMP", "WBC", "HGB", "PLT", "Urinalysis POCT"],
    "renal_colic":            ["Urinalysis POCT", "BMP", "WBC", "HGB", "PLT", "CRP"],
    "vomiting_nausea":        ["BMP", "WBC", "HGB", "PLT", "Glucose POCT", "LFTs", "Lipase"],
    "rash":                   ["WBC", "HGB", "PLT", "CRP", "Glucose POCT"],
    "joint_pain":             ["WBC", "HGB", "PLT", "CRP", "BMP", "Glucose POCT"],
    "psychiatric_emergency":  ["Glucose POCT", "WBC", "HGB", "PLT", "BMP", "Urinalysis POCT"],
    "other":                  ["WBC", "HGB", "PLT", "BMP", "Glucose POCT", "CRP"],
}

# Expand panel orders into individual component tests
# Note: CBC is NOT a panel — WBC, HGB, PLT are ordered as 3 separate standalone tests.
PANEL_EXPANSION: dict[str, list[str]] = {
    "BMP":                   ["Sodium", "Potassium", "Creatinine", "BUN", "Glucose", "Bicarbonate"],
    "LFTs":                  ["ALT", "AST", "Bilirubin"],
    "Coagulation (PT/APTT)": ["PT/INR", "APTT"],
    "Blood gas (ABG/VBG)":   ["pH", "pCO2", "pO2", "Bicarbonate", "Lactate"],
}

# Mapping from order names to lab_reference keys for interpretation
ORDER_TO_KEY: dict[str, list[str]] = {
    "CBC":              ["wbc", "neutrophils", "lymphocytes", "monocytes", "eosinophils",
                        "rbc", "hemoglobin", "hematocrit", "mcv", "mch", "mchc", "platelets", "rdw"],
    "BMP":              ["sodium", "potassium", "creatinine", "glucose", "bun", "bicarbonate"],
    "Troponin POCT":    ["troponin"],
    "Troponin":         ["troponin"],
    "Lactate":          ["lactate"],
    "BNP":              ["bnp"],
    "CRP":              ["crp"],
    "Blood gas":        ["ph", "pco2", "po2", "bicarbonate"],
    "Glucose POCT":     ["glucose"],
    "LFTs":             ["alt", "ast", "bilirubin"],
    "Lipase":           ["lipase"],
    "Coagulation screen": ["inr", "ptt"],
    "D-Dimer":          ["d-dimer"],
}


def _extract_number(value_str: str) -> float | None:
    """Extract first numeric value from a result string."""
    m = re.search(r"[-+]?\d*\.?\d+", value_str.replace(",", "."))
    return float(m.group()) if m else None


def interpret_result(test_name: str, value_str: str) -> dict:
    """
    Interpret a lab result against reference ranges.
    Returns dict with is_abnormal, is_critical, flag, reference_range, interpretation.
    """
    # Normalise test name to look up reference key
    test_key = test_name.strip().lower()
    # Try direct lookup first, then ORDER_TO_KEY
    ref = LAB_REFERENCE.get(test_key)
    if ref is None:
        # Try matching via ORDER_TO_KEY
        for order_name, keys in ORDER_TO_KEY.items():
            if order_name.lower() == test_key or test_key in order_name.lower():
                ref = LAB_REFERENCE.get(keys[0]) if keys else None
                break

    # If we have a reference range, try numeric comparison
    if ref is not None:
        num = _extract_number(value_str)
        if num is not None:
            lo, hi = ref["lo"], ref["hi"]
            clo, chi = ref.get("clo"), ref.get("chi")
            is_critical = (clo is not None and num < clo) or (chi is not None and num > chi)
            is_abnormal = num < lo or num > hi or is_critical

            if is_critical and clo is not None and num < clo:
                flag = "critical_low"
                interpretation = f"CRITICAL LOW — {num} {ref['unit']} (ref: {ref['range']})"
            elif is_critical and chi is not None and num > chi:
                flag = "critical_high"
                interpretation = f"CRITICAL HIGH — {num} {ref['unit']} (ref: {ref['range']})"
            elif num < lo:
                flag = "low"
                interpretation = f"Low — {num} {ref['unit']} (ref: {ref['range']})"
            elif num > hi:
                flag = "high"
                interpretation = f"High — {num} {ref['unit']} (ref: {ref['range']})"
            else:
                flag = "normal"
                interpretation = f"Normal — {num} {ref['unit']} (ref: {ref['range']})"

            return dict(
                is_abnormal=is_abnormal,
                is_critical=is_critical,
                flag=flag,
                reference_range=ref["range"],
                interpretation=interpretation,
                numeric_value=num,
                unit=ref["unit"],
            )

    # Qualitative result or unknown test — look for keywords
    lower_val = value_str.lower()
    is_positive = any(k in lower_val for k in ("positive", "elevated", "abnormal", "high", "detected", "+"))
    is_critical_kw = any(k in lower_val for k in ("critical", "panic", "life-threatening"))
    is_negative = any(k in lower_val for k in ("negative", "normal", "clear", "not detected", "within"))

    if is_critical_kw:
        return dict(is_abnormal=True, is_critical=True, flag="critical_high",
                    reference_range="Qualitative", interpretation=f"CRITICAL: {value_str}",
                    numeric_value=None, unit="")
    if is_positive:
        return dict(is_abnormal=True, is_critical=False, flag="high",
                    reference_range="Qualitative", interpretation=f"Abnormal: {value_str}",
                    numeric_value=None, unit="")
    if is_negative:
        return dict(is_abnormal=False, is_critical=False, flag="normal",
                    reference_range="Qualitative", interpretation=f"Normal: {value_str}",
                    numeric_value=None, unit="")

    # Unknown — return as uninterpretable
    return dict(is_abnormal=False, is_critical=False, flag="unknown",
                reference_range="N/A", interpretation=f"Result recorded: {value_str}",
                numeric_value=None, unit="")


def get_default_orders(category: str) -> list[str]:
    """Return default suggested lab orders for a chief complaint category."""
    return CATEGORY_DEFAULT_ORDERS.get(category, CATEGORY_DEFAULT_ORDERS["other"])
