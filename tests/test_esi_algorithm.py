from __future__ import annotations

import pytest

from src.core.esi_algorithm import ESIResult, check_high_risk_vitals, run_esi_algorithm
from src.core.patient import (
    AgeGroup,
    ChiefComplaint,
    IntakeForm,
    MedicalHistory,
    NurseObservation,
    Vitals,
)


def make_intake(
    age=45,
    obs: NurseObservation = None,
    vitals: Vitals = None,
    hx: MedicalHistory = None,
    cc_category="other",
    cc_text="test",
    pain_score=None,
):
    return IntakeForm(
        patient_id="t001",
        age_years=age,
        chief_complaint=ChiefComplaint(
            free_text_en=cc_text,
            category=cc_category,
            pain_score=pain_score,
        ),
        vitals=vitals or Vitals(),
        nurse_observation=obs or NurseObservation(),
        medical_history=hx or MedicalHistory(),
    )


def test_esi1_respiratory_arrest():
    intake = make_intake(
        obs=NurseObservation(work_of_breathing="apneic", avpu="P", general_appearance="critically_ill"),
        vitals=Vitals(spo2_pct=72, heart_rate=140),
    )
    r = run_esi_algorithm(intake, 0)
    assert r.esi_level == 1
    assert r.decision_point_reached == "A"
    assert r.pce_scope == "excluded_resus"


def test_esi2_chest_pain_diaphoresis():
    intake = make_intake(
        obs=NurseObservation(
            skin_assessment="diaphoretic",
            general_appearance="distressed",
            acute_distress=True,
            work_of_breathing="moderate_distress",
        ),
        vitals=Vitals(heart_rate=110, systolic_bp=90),
    )
    r = run_esi_algorithm(intake, 2)
    assert r.esi_level == 2
    assert r.pce_scope == "pce_priority"


def test_esi2_immunocompromised_fever():
    intake = make_intake(
        hx=MedicalHistory(immunocompromised=True),
        vitals=Vitals(temperature_c=39.2),
    )
    r = run_esi_algorithm(intake, 1)
    assert r.esi_level == 2


def test_esi3_resources_2():
    intake = make_intake(vitals=Vitals(heart_rate=70, spo2_pct=98))
    r = run_esi_algorithm(intake, 2)
    assert r.esi_level == 3


def test_esi4_resources_1():
    intake = make_intake(vitals=Vitals(heart_rate=70, spo2_pct=98))
    r = run_esi_algorithm(intake, 1)
    assert r.esi_level == 4


def test_esi5_resources_0():
    intake = make_intake(vitals=Vitals(heart_rate=70, spo2_pct=99))
    r = run_esi_algorithm(intake, 0)
    assert r.esi_level == 5
    assert r.pce_scope == "pce_lite"


def test_dp_d_upgrades():
    intake = make_intake(
        age=35,
        vitals=Vitals(heart_rate=115, spo2_pct=98),
    )
    r = run_esi_algorithm(intake, 1)
    assert r.esi_level <= 3
    assert r.decision_point_reached == "D"


def test_pce_scope_mapping():
    intake1 = make_intake(
        obs=NurseObservation(work_of_breathing="apneic"),
    )
    r1 = run_esi_algorithm(intake1, 0)
    assert r1.pce_scope == "excluded_resus"

    intake3 = make_intake(vitals=Vitals(heart_rate=70, spo2_pct=99))
    r3 = run_esi_algorithm(intake3, 2)
    assert r3.pce_scope == "pce_core"

    intake5 = make_intake(vitals=Vitals(heart_rate=70, spo2_pct=99))
    r5 = run_esi_algorithm(intake5, 0)
    assert r5.pce_scope == "pce_lite"
