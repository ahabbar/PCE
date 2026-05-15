from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, field_validator, model_validator


class AgeGroup(str, Enum):
    NEONATE = "neonate"
    INFANT = "infant"
    TODDLER = "toddler"
    PRESCHOOL = "preschool"
    CHILD = "child"
    ADOLESCENT = "adolescent"
    ADULT = "adult"


class ArrivalMode(str, Enum):
    WALK_IN = "walk_in"
    AMBULANCE = "ambulance"
    POLICE = "police"
    TRANSFER = "transfer"
    OTHER = "other"


class Vitals(BaseModel):
    heart_rate: Optional[float] = None
    respiratory_rate: Optional[float] = None
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    temperature_c: Optional[float] = None
    spo2_pct: Optional[float] = None
    gcs: Optional[float] = None
    blood_glucose: Optional[float] = None
    weight_kg: Optional[float] = None

    @field_validator("gcs")
    @classmethod
    def gcs_range(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (3 <= v <= 15):
            raise ValueError("GCS must be between 3 and 15")
        return v


class ChiefComplaint(BaseModel):
    free_text_en: str
    free_text_ar: str = ""
    category: Literal[
        "chest_pain",
        "dyspnea",
        "abdominal_pain",
        "altered_mental_status",
        "neurological",
        "urinary",
        "fever",
        "musculoskeletal",
        "skin",
        "eye",
        "ear_nose_throat",
        "psychiatric",
        "obstetric_gynaecological",
        "trauma",
        "poisoning_overdose",
        "allergic",
        "endocrine",
        "haematological",
        "gastrointestinal",
        "cardiovascular",
        "other",
    ] = "other"
    pain_present: bool = False
    pain_score: Optional[int] = None
    onset_minutes: Optional[int] = None
    onset_type: Optional[Literal["sudden", "gradual", "unknown"]] = None

    @field_validator("pain_score")
    @classmethod
    def pain_score_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (0 <= v <= 10):
            raise ValueError("pain_score must be 0-10")
        return v


class MedicalHistory(BaseModel):
    known_diagnoses: list[str] = []
    current_medications: list[str] = []
    allergies: list[str] = []
    last_er_visit_days: Optional[int] = None
    immunocompromised: bool = False
    anticoagulated: bool = False
    baseline_spo2_low: bool = False
    baseline_cr_abnormal: Optional[float] = None


class NurseObservation(BaseModel):
    general_appearance: Literal["well", "unwell", "distressed", "critically_ill", "altered"] = "well"
    work_of_breathing: Literal[
        "normal", "mild_increased", "moderate_distress", "severe_distress", "apneic"
    ] = "normal"
    skin_assessment: Literal[
        "normal", "pale", "diaphoretic", "mottled", "cyanotic", "flushed", "jaundiced"
    ] = "normal"
    pulse_quality: Literal["normal", "weak", "bounding", "irregular", "absent"] = "normal"
    avpu: Optional[Literal["A", "V", "P", "U"]] = None
    acute_distress: bool = False
    altered_mentation: bool = False
    seizure_active: bool = False


def _derive_age_group(age_years: float) -> AgeGroup:
    if age_years < (28 / 365):
        return AgeGroup.NEONATE
    elif age_years < 1:
        return AgeGroup.INFANT
    elif age_years < 3:
        return AgeGroup.TODDLER
    elif age_years < 6:
        return AgeGroup.PRESCHOOL
    elif age_years < 12:
        return AgeGroup.CHILD
    elif age_years < 18:
        return AgeGroup.ADOLESCENT
    else:
        return AgeGroup.ADULT


class IntakeForm(BaseModel):
    patient_id: str
    arrival_time: float = 0.0
    arrival_mode: ArrivalMode = ArrivalMode.WALK_IN
    age_years: float
    age_group: AgeGroup = AgeGroup.ADULT
    gender: Literal["male", "female", "other", "unknown"] = "unknown"
    pregnant: Optional[bool] = None
    chief_complaint: ChiefComplaint
    vitals: Vitals
    medical_history: MedicalHistory = None  # type: ignore[assignment]
    nurse_observation: NurseObservation = None  # type: ignore[assignment]
    additional_context: str = ""

    def model_post_init(self, __context: Any) -> None:
        if self.arrival_time == 0.0:
            object.__setattr__(self, "arrival_time", time.time())
        if self.medical_history is None:
            object.__setattr__(self, "medical_history", MedicalHistory())
        if self.nurse_observation is None:
            object.__setattr__(self, "nurse_observation", NurseObservation())

    @field_validator("age_years")
    @classmethod
    def age_range(cls, v: float) -> float:
        if not (0 <= v <= 120):
            raise ValueError("age_years must be 0-120")
        return v

    @model_validator(mode="after")
    def derive_age_group(self) -> IntakeForm:
        object.__setattr__(self, "age_group", _derive_age_group(self.age_years))
        return self


class TriageScoreResult(BaseModel):
    risk_score: float
    esi_level: int
    confidence: float
    is_red: bool = False
    threshold_used: float = 85.0
    key_factors: list[str]
    reasoning: str
    pce_scope: str = ""

    @field_validator("risk_score")
    @classmethod
    def score_range(cls, v: float) -> float:
        if not (0 <= v <= 100):
            raise ValueError("risk_score must be 0-100")
        return v

    @field_validator("key_factors")
    @classmethod
    def factors_not_empty(cls, v: list[str]) -> list[str]:
        if len(v) < 1:
            raise ValueError("key_factors must have at least 1 entry")
        if len(v) > 6:
            raise ValueError("key_factors must have at most 6 entries")
        return v

    @field_validator("reasoning")
    @classmethod
    def reasoning_length(cls, v: str) -> str:
        if len(v) > 200:
            raise ValueError("reasoning must be <= 200 chars")
        return v


class RedFlagResult(BaseModel):
    is_emergency: bool
    esi1_immediate: bool = False
    esi2_high_risk: bool = False
    flag_type: Optional[str] = None
    immediate_action: Optional[str] = None
    override_all_agents: bool = False
    layer_triggered: Literal["rules", "llm", "none"] = "none"
    reasoning: str = ""


class OrderItem(BaseModel):
    test_name: str
    reason: str
    cost_tier: Literal["low", "medium", "high"]
    timing: Literal["immediate", "urgent", "routine"] = "routine"
    whitelist_rule: str
    conditional_reason: Optional[str] = None


class WorkupPlan(BaseModel):
    patient_id: str
    protocol_key: str
    orders: list[OrderItem]
    deferred_to_doctor: list[str]
    reasoning: str


class DispositionPrediction(BaseModel):
    discharge_pct: float = 0.0
    admit_pct: float = 0.0
    icu_pct: float = 0.0
    transfer_pct: float = 0.0
    predicted_destination: Literal["discharge", "admit", "icu", "transfer"] = "discharge"
    reasoning: str = ""
    bed_reservation_sent: bool = False


class ExitPlan(BaseModel):
    disposition: Literal["discharge", "admit", "icu", "transfer"]
    instructions: Optional[str] = None
    follow_up: Optional[str] = None
    handover_note: Optional[str] = None
    transport_needed: bool = False
    gp_letter_draft: Optional[str] = None
    prescription_notes: Optional[str] = None


class PatientRecord(BaseModel):
    intake: IntakeForm
    esi_result: Optional[Any] = None
    triage_score: Optional[TriageScoreResult] = None
    red_flag: Optional[RedFlagResult] = None
    workup: Optional[WorkupPlan] = None
    status: Literal["waiting", "workup", "results", "assigned", "discharged"] = "waiting"
    arrival_time: float = 0.0
    assigned_doctor: Optional[str] = None
    result_count: int = 0

    def model_post_init(self, __context: Any) -> None:
        if self.arrival_time == 0.0:
            object.__setattr__(self, "arrival_time", time.time())
