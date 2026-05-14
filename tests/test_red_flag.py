from __future__ import annotations

import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.red_flag import check_red_flags_fast, run_red_flag_guardian
from src.core.patient import (
    ChiefComplaint,
    IntakeForm,
    MedicalHistory,
    NurseObservation,
    Vitals,
)


def make_intake(
    age=45,
    gender="male",
    cc_text="test",
    cc_category="other",
    vitals: Vitals = None,
    obs: NurseObservation = None,
    hx: MedicalHistory = None,
):
    return IntakeForm(
        patient_id=str(uuid.uuid4()),
        age_years=age,
        gender=gender,
        chief_complaint=ChiefComplaint(free_text_en=cc_text, category=cc_category),
        vitals=vitals or Vitals(),
        nurse_observation=obs or NurseObservation(),
        medical_history=hx or MedicalHistory(),
    )


def test_layer1_cardiac_arrest():
    intake = make_intake(obs=NurseObservation(pulse_quality="absent"))
    result = check_red_flags_fast(intake)
    assert result is not None
    assert result.is_emergency
    assert result.layer_triggered == "rules"


def test_layer1_hypoglycaemia():
    intake = make_intake(
        vitals=Vitals(blood_glucose=1.8),
        obs=NurseObservation(altered_mentation=True),
    )
    result = check_red_flags_fast(intake)
    assert result is not None
    assert result.is_emergency


def test_layer1_sepsis():
    intake = make_intake(
        vitals=Vitals(temperature_c=39.1, heart_rate=115, systolic_bp=88),
    )
    result = check_red_flags_fast(intake)
    assert result is not None
    assert result.is_emergency
    assert result.esi2_high_risk


def test_layer1_stemi_presentation():
    intake = make_intake(
        cc_category="chest_pain",
        obs=NurseObservation(skin_assessment="diaphoretic", general_appearance="distressed"),
    )
    result = check_red_flags_fast(intake)
    assert result is not None
    assert result.is_emergency


def test_layer1_clear():
    intake = make_intake(
        vitals=Vitals(heart_rate=72, systolic_bp=120, spo2_pct=99, temperature_c=36.8),
    )
    result = check_red_flags_fast(intake)
    assert result is None


def test_layer1_no_llm_needed():
    intake = make_intake(obs=NurseObservation(pulse_quality="absent"))
    mock_llm = MagicMock()
    mock_llm.call = AsyncMock()

    import asyncio
    result = asyncio.run(run_red_flag_guardian(intake, mock_llm))
    assert result.is_emergency
    mock_llm.call.assert_not_called()


@pytest.mark.real_llm
@pytest.mark.asyncio
async def test_llm_layer_validates_schema():
    os.environ["LLM_PROVIDER"] = "gemini"
    import src.agents.llm as llm_mod
    llm_mod._singleton = None
    from src.agents.llm import get_llm_client
    from src.core.patient import RedFlagResult

    intake = make_intake(
        cc_text="chest pain and shortness of breath",
        cc_category="chest_pain",
        vitals=Vitals(heart_rate=95, systolic_bp=140, spo2_pct=96, temperature_c=37.0),
    )
    llm = get_llm_client()
    result = await run_red_flag_guardian(intake, llm)
    assert isinstance(result, RedFlagResult)
    assert result.layer_triggered in ("rules", "llm")


def test_shock_triggers_immediately():
    intake = make_intake(
        vitals=Vitals(systolic_bp=74),
        obs=NurseObservation(skin_assessment="mottled"),
    )
    result = check_red_flags_fast(intake)
    assert result is not None
    assert result.is_emergency
    assert result.esi1_immediate
