from __future__ import annotations
from dataclasses import dataclass, field
import time, logging
from src.core.patient import WorkupPlan

logger = logging.getLogger("pce.batch")


@dataclass
class PendingOrder:
    patient_id: str
    test_name: str
    esi_level: int
    ordered_at: float
    cost_tier: str
    timing: str


@dataclass
class CollectionRound:
    round_number: int
    tests: list[str]
    patient_ids: list[str]
    scheduled_offset_min: float
    efficiency_note: str


class BatchCoordinator:
    def __init__(self, batch_window_min: float = 5.0, max_wait_min: float = 8.0) -> None:
        self._batch_window_sec = batch_window_min * 60
        self._max_wait_sec = max_wait_min * 60
        self._pending: list[PendingOrder] = []

    def add_orders(
        self,
        patient_id: str,
        workup: WorkupPlan,
        esi_level: int,
    ) -> None:
        """Add orders from a WorkupPlan to the pending pool."""
        now = time.time()
        for order in workup.orders:
            timing = "immediate" if esi_level <= 2 else order.timing
            self._pending.append(PendingOrder(
                patient_id=patient_id,
                test_name=order.test_name,
                esi_level=esi_level,
                ordered_at=now,
                cost_tier=order.cost_tier,
                timing=timing,
            ))
        logger.debug(
            "add_orders | patient=%s esi=%d orders=%d total_pending=%d",
            patient_id[:8], esi_level, len(workup.orders), len(self._pending),
        )

    def get_collection_rounds(self, current_time: float | None = None) -> list[CollectionRound]:
        """
        Build collection rounds from the current pending pool.

        - Immediate orders → one CollectionRound per unique patient (round 0, offset 0)
        - Non-immediate orders:
            - Groups with 2+ patients sharing the same test → batched round
            - Single patient past max_wait → solo round
            - Single patient within max_wait → held (skipped)
        - Immediate rounds first, then non-immediate sorted by patient_count desc
        """
        ct = current_time if current_time is not None else time.time()

        immediate = [o for o in self._pending if o.timing == "immediate"]
        non_immediate = [o for o in self._pending if o.timing != "immediate"]

        # --- Immediate rounds (one per unique patient_id) ---
        immediate_by_patient: dict[str, list[PendingOrder]] = {}
        for o in immediate:
            immediate_by_patient.setdefault(o.patient_id, []).append(o)

        immediate_rounds: list[CollectionRound] = [
            CollectionRound(
                round_number=0,
                tests=list({o.test_name for o in orders}),
                patient_ids=[pid],
                scheduled_offset_min=0.0,
                efficiency_note="immediate — ESI-2 or urgent",
            )
            for pid, orders in immediate_by_patient.items()
        ]

        # --- Non-immediate rounds (group by test_name) ---
        by_test: dict[str, list[PendingOrder]] = {}
        for o in non_immediate:
            by_test.setdefault(o.test_name, []).append(o)

        batched_rounds: list[CollectionRound] = []
        for test_name, orders in by_test.items():
            patient_ids = list({o.patient_id for o in orders})

            if len(patient_ids) >= 2:
                # Batchable group
                batched_rounds.append(CollectionRound(
                    round_number=0,  # assigned below
                    tests=[test_name],
                    patient_ids=patient_ids,
                    scheduled_offset_min=0.0,
                    efficiency_note=f"batched {len(patient_ids)} patients",
                ))
            else:
                # Single patient — check max_wait
                oldest = min(o.ordered_at for o in orders)
                elapsed = ct - oldest
                if elapsed >= self._max_wait_sec:
                    batched_rounds.append(CollectionRound(
                        round_number=0,  # assigned below
                        tests=[test_name],
                        patient_ids=patient_ids,
                        scheduled_offset_min=0.0,
                        efficiency_note="solo — max wait exceeded",
                    ))
                # else: hold, skip

        # Sort non-immediate by patient count desc, then assign round numbers
        batched_rounds.sort(key=lambda r: len(r.patient_ids), reverse=True)
        for idx, rnd in enumerate(batched_rounds, start=1):
            rnd.round_number = idx

        return immediate_rounds + batched_rounds

    def confirm_dispatched(self, patient_ids: list[str], test_name: str) -> None:
        """Remove matching PendingOrders that have been dispatched."""
        pid_set = set(patient_ids)
        before = len(self._pending)
        self._pending = [
            o for o in self._pending
            if not (o.patient_id in pid_set and o.test_name == test_name)
        ]
        removed = before - len(self._pending)
        logger.debug("confirm_dispatched | test=%s removed=%d", test_name, removed)

    def pending_count(self) -> int:
        return len(self._pending)

    def batch_efficiency_pct(self) -> float:
        """
        Of all non-immediate orders, what % are in groups of 2+ patients
        sharing the same test_name?
        """
        non_immediate = [o for o in self._pending if o.timing != "immediate"]
        if not non_immediate:
            return 0.0

        by_test: dict[str, set[str]] = {}
        for o in non_immediate:
            by_test.setdefault(o.test_name, set()).add(o.patient_id)

        batched_count = sum(
            len(patient_ids)
            for patient_ids in by_test.values()
            if len(patient_ids) >= 2
        )
        return (batched_count / len(non_immediate)) * 100.0

    def get_batch_summary(self) -> dict[str, int]:
        """Return {test_name: patient_count} for tests with 2+ patients."""
        by_test: dict[str, set[str]] = {}
        for o in self._pending:
            by_test.setdefault(o.test_name, set()).add(o.patient_id)
        return {
            test: len(pids)
            for test, pids in by_test.items()
            if len(pids) >= 2
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_batch: BatchCoordinator | None = None


def get_batch_coordinator() -> BatchCoordinator:
    global _batch
    if _batch is None:
        _batch = BatchCoordinator()
    return _batch
