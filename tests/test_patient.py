from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.patient import (
    AgeGroup,
    ChiefComplaint,
    IntakeForm,
    MedicalHistory,
    NurseObservation,
    OrderItem,
    TriageScoreResult,
    Vitals,
)


def make_intake(age=45, **kwargs):
    defaults = dict(
        patient_id="test-001",
        age_years=age,
        chief_complaint=ChiefComplaint(free_text_en="test complaint"),
        vitals=Vitals(),
    )
    defaults.update(kwargs)
    return IntakeForm(**defaults)


def test_intake_age_group_adult():
    p = make_intake(age=45)
    assert p.age_group == AgeGroup.ADULT


def test_intake_age_group_neonate():
    p = make_intake(age=0.04)
    assert p.age_group == AgeGroup.NEONATE


def test_intake_age_group_child():
    p = make_intake(age=8)
    assert p.age_group == AgeGroup.CHILD


def test_triage_score_valid():
    t = TriageScoreResult(
        risk_score=73.5,
        esi_level=3,
        confidence=0.8,
        is_red=False,
        key_factors=["stable vitals"],
        reasoning="Looks stable",
    )
    assert t.risk_score == 73.5
    assert not t.is_red


def test_triage_score_over_100():
    with pytest.raises(ValidationError):
        TriageScoreResult(
            risk_score=110,
            esi_level=3,
            confidence=0.8,
            key_factors=["x"],
            reasoning="test",
        )


def test_vitals_all_optional():
    v = Vitals()
    assert v.heart_rate is None
    assert v.spo2_pct is None


def test_order_item_timing():
    o = OrderItem(
        test_name="CBC",
        reason="baseline",
        cost_tier="low",
        timing="routine",
        whitelist_rule="fever_infectious",
    )
    assert o.timing == "routine"
