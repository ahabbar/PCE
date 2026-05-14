from __future__ import annotations
import pytest
import uuid
from httpx import AsyncClient, ASGITransport
from src.api.server import app


@pytest.mark.asyncio
async def test_health():
    """GET /health should return 200 with status ok."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
@pytest.mark.real_llm
async def test_triage_returns_valid_response():
    """POST /triage with a UTI intake should return valid triage response."""
    payload = {
        "patient_id": str(uuid.uuid4()),
        "age_years": 34,
        "gender": "female",
        "chief_complaint": {
            "free_text_en": "dysuria and flank pain",
            "category": "urinary",
            "pain_present": True,
            "pain_score": 5,
        },
        "vitals": {
            "heart_rate": 98,
            "temperature_c": 38.6,
            "systolic_bp": 118,
            "spo2_pct": 97,
        },
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/triage", json=payload)

    assert r.status_code == 200
    data = r.json()
    assert 35 <= data["risk_score"] <= 85
    assert len(data["workup_orders"]) > 0
