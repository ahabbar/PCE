from __future__ import annotations

import time
import uuid

import pytest

from src.agents.batch import BatchCoordinator, PendingOrder
from src.core.patient import OrderItem, WorkupPlan


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_plan(patient_id: str, tests: list[str]) -> WorkupPlan:
    orders = [
        OrderItem(
            test_name=t,
            reason="test",
            cost_tier="low",
            timing="routine",
            whitelist_rule="test",
        )
        for t in tests
    ]
    return WorkupPlan(
        patient_id=patient_id,
        protocol_key="test",
        orders=orders,
        deferred_to_doctor=[],
        reasoning="test",
    )


def make_patient_id() -> str:
    return f"pt-{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_esi2_bypasses_batch():
    """ESI-2 patient should appear in a round with scheduled_offset_min=0."""
    bc = BatchCoordinator()
    pid = "pt-esi2"
    plan = make_plan(pid, ["CBC"])
    bc.add_orders(pid, plan, esi_level=2)

    rounds = bc.get_collection_rounds()
    assert len(rounds) >= 1
    immediate_rounds = [r for r in rounds if r.scheduled_offset_min == 0 and pid in r.patient_ids]
    assert len(immediate_rounds) == 1, f"Expected 1 immediate round for ESI-2, got: {rounds}"


def test_batch_grouping():
    """4 ESI-3 patients each needing CBC should produce one round with 4 patient_ids."""
    bc = BatchCoordinator()
    pids = [f"pt-{i}" for i in range(4)]
    for pid in pids:
        bc.add_orders(pid, make_plan(pid, ["CBC"]), esi_level=3)

    rounds = bc.get_collection_rounds()
    cbc_rounds = [r for r in rounds if "CBC" in r.tests]
    assert len(cbc_rounds) == 1
    assert len(cbc_rounds[0].patient_ids) == 4


def test_min_batch_size():
    """1 ESI-3 patient ordered just now should NOT appear in any round (held)."""
    bc = BatchCoordinator()
    pid = "pt-solo-new"
    plan = make_plan(pid, ["CBC"])
    bc.add_orders(pid, plan, esi_level=3)

    rounds = bc.get_collection_rounds()
    # No round should contain this patient (it's held, ordered just now)
    all_pids = [p for r in rounds for p in r.patient_ids]
    assert pid not in all_pids, f"Expected patient to be held, but found in rounds: {rounds}"


def test_max_wait_solo():
    """A single patient whose order is older than max_wait should appear in a solo round."""
    bc = BatchCoordinator(max_wait_min=8.0)
    pid = "pt-old"
    # Manually inject an old PendingOrder (ordered 10 minutes ago)
    old_order = PendingOrder(
        patient_id=pid,
        test_name="CBC",
        esi_level=3,
        ordered_at=time.time() - 600,  # 10 min ago
        cost_tier="low",
        timing="routine",
    )
    bc._pending.append(old_order)

    rounds = bc.get_collection_rounds()
    solo_rounds = [r for r in rounds if pid in r.patient_ids]
    assert len(solo_rounds) == 1, f"Expected solo round for old patient, got: {rounds}"
    assert solo_rounds[0].efficiency_note == "solo — max wait exceeded"


def test_efficiency_calculation():
    """4 patients needing CBC + 1 patient needing ECG → efficiency > 60%."""
    bc = BatchCoordinator()

    # 4 patients with CBC (4 out of 5 non-immediate orders are batchable)
    for i in range(4):
        pid = f"pt-cbc-{i}"
        bc.add_orders(pid, make_plan(pid, ["CBC"]), esi_level=3)

    # 1 patient with ECG (solo, not batchable)
    pid_ecg = "pt-ecg"
    bc.add_orders(pid_ecg, make_plan(pid_ecg, ["ECG 12-lead"]), esi_level=3)

    eff = bc.batch_efficiency_pct()
    assert eff > 60.0, f"Expected efficiency > 60%, got {eff:.1f}%"


def test_confirm_dispatched():
    """After dispatching 2 of 3 patients' CBC, pending_count should be 1."""
    bc = BatchCoordinator()
    pids = ["pt-0", "pt-1", "pt-2"]
    for pid in pids:
        bc.add_orders(pid, make_plan(pid, ["CBC"]), esi_level=3)

    assert bc.pending_count() == 3
    bc.confirm_dispatched(["pt-0", "pt-1"], "CBC")
    assert bc.pending_count() == 1


def test_empty_pool():
    """Fresh BatchCoordinator should return empty list from get_collection_rounds()."""
    bc = BatchCoordinator()
    assert bc.get_collection_rounds() == []
