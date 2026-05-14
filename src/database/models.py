from __future__ import annotations

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    arrival_time REAL,
    age_years REAL,
    gender TEXT,
    chief_complaint_text TEXT,
    chief_complaint_category TEXT,
    esi_level INTEGER,
    risk_score REAL,
    is_red INTEGER,
    protocol_key TEXT,
    status TEXT,
    assigned_doctor TEXT,
    created_at REAL,
    updated_at REAL,
    disposition_prediction TEXT,
    bed_reservation_sent INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS investigations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT,
    test_name TEXT,
    cost_tier TEXT,
    timing TEXT,
    protocol_rule TEXT,
    status TEXT,
    ordered_at REAL,
    resulted_at REAL,
    result_value TEXT,
    FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT,
    agent_id INTEGER,
    action TEXT,
    inputs_summary TEXT,
    outputs_summary TEXT,
    latency_ms REAL,
    model_used TEXT,
    timestamp REAL
);

CREATE TABLE IF NOT EXISTS nurse_confirmations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT,
    test_name TEXT,
    action TEXT,
    reason_code TEXT,
    nurse_id TEXT,
    timestamp REAL
);
"""
