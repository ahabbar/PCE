from __future__ import annotations
import logging, re, time
from typing import Optional
from pydantic import BaseModel
from src.agents.llm import LLMClient
from src.core.whitelist import ProtocolWhitelist
from src.core.patient import IntakeForm, TriageScoreResult, WorkupPlan, OrderItem

logger = logging.getLogger("pce.workup")


def evaluate_conditional(condition_str: str, intake: IntakeForm, whitelist: ProtocolWhitelist) -> bool:
    """Evaluate a condition string against an IntakeForm. Returns True if condition is met."""
    # Handle OR (case-insensitive)
    upper = condition_str.upper()
    if " OR " in upper:
        parts = re.split(r"\s+OR\s+", condition_str, flags=re.IGNORECASE)
        return any(evaluate_conditional(p.strip(), intake, whitelist) for p in parts)

    # Handle AND (case-insensitive)
    if " AND " in upper:
        parts = re.split(r"\s+AND\s+", condition_str, flags=re.IGNORECASE)
        return all(evaluate_conditional(p.strip(), intake, whitelist) for p in parts)

    # Single condition evaluation
    text = condition_str.strip().lower()
    v = intake.vitals
    hx = intake.medical_history

    # temperature > X or temp > X
    m = re.match(r"(?:temperature|temp)\s*>\s*([0-9.]+)", text)
    if m:
        threshold = float(m.group(1))
        if v.temperature_c is not None:
            return v.temperature_c > threshold
        return False

    # HR > X or heart_rate > X
    m = re.match(r"(?:hr|heart_rate)\s*>\s*([0-9.]+)", text)
    if m:
        threshold = float(m.group(1))
        if v.heart_rate is not None:
            return v.heart_rate > threshold
        return False

    # SBP < X or systolic < X
    m = re.match(r"(?:sbp|systolic)\s*<\s*([0-9.]+)", text)
    if m:
        threshold = float(m.group(1))
        if v.systolic_bp is not None:
            return v.systolic_bp < threshold
        return False

    # SpO2 < X
    m = re.match(r"spo2\s*<\s*([0-9.]+)", text)
    if m:
        threshold = float(m.group(1))
        if v.spo2_pct is not None:
            return v.spo2_pct < threshold
        return False

    # Gender: female or women
    if "female" in text or "women" in text:
        return intake.gender == "female"

    # Age: elderly or > 65 or age > 65
    if "elderly" in text or "> 65" in text or "age > 65" in text:
        return intake.age_years > 65

    # Immunocompromised
    if "immunocompromised" in text:
        return hx.immunocompromised

    # CKD / renal / kidney
    if "ckd" in text or "renal" in text or "kidney" in text:
        for dx in hx.known_diagnoses:
            dx_lower = dx.lower()
            if "ckd" in dx_lower or "renal" in dx_lower or "kidney" in dx_lower:
                return True
        return False

    # No match found — conservative: return True
    logger.debug("evaluate_conditional: no match for %r — defaulting True", condition_str)
    return True


async def run_preemptive_workup(
    intake: IntakeForm,
    triage_score: TriageScoreResult,
    llm: LLMClient,
    whitelist: ProtocolWhitelist,
) -> WorkupPlan:
    t0 = time.time()

    # ESI-1 fast path
    if triage_score.esi_level == 1:
        logger.info("patient=%s ESI-1 fast path — skipping workup", intake.patient_id[:8])
        return WorkupPlan(
            patient_id=intake.patient_id,
            protocol_key="none",
            orders=[],
            deferred_to_doctor=[],
            reasoning="ESI-1: Resus team manages workup",
        )

    # Match protocol
    protocol_key = whitelist.match(intake.chief_complaint.free_text_en)
    if protocol_key is None:
        # Fallback to category
        category = intake.chief_complaint.category
        protocol_key = whitelist.match(category)

    if protocol_key is None:
        logger.info("patient=%s no matching protocol", intake.patient_id[:8])
        return WorkupPlan(
            patient_id=intake.patient_id,
            protocol_key="unknown",
            orders=[],
            deferred_to_doctor=[],
            reasoning="No matching protocol found",
        )

    orders: list[OrderItem] = []

    # Auto orders
    for entry in whitelist.get_auto_orders(protocol_key):
        test_name = entry["test"]
        if not whitelist.is_allowed(protocol_key, test_name):
            logger.warning(
                "patient=%s auto order %r not allowed by whitelist for protocol %r — skipping",
                intake.patient_id[:8], test_name, protocol_key,
            )
            continue
        orders.append(OrderItem(
            test_name=test_name,
            reason=entry.get("reason", "Protocol auto-order"),
            cost_tier=entry.get("cost_tier", "low"),
            timing=entry.get("timing", "routine"),
            whitelist_rule=f"{protocol_key}:auto",
        ))

    # Conditional orders
    for entry in whitelist.get_conditional_orders(protocol_key):
        test_name = entry["test"]
        condition_str = entry.get("condition", "")
        if not evaluate_conditional(condition_str, intake, whitelist):
            continue
        if not whitelist.is_allowed(protocol_key, test_name):
            logger.warning(
                "patient=%s conditional order %r not allowed by whitelist for protocol %r — skipping",
                intake.patient_id[:8], test_name, protocol_key,
            )
            continue
        orders.append(OrderItem(
            test_name=test_name,
            reason=entry.get("reason", "Conditional protocol order"),
            cost_tier=entry.get("cost_tier", "low"),
            timing=entry.get("timing", "routine"),
            whitelist_rule=f"{protocol_key}:conditional",
            conditional_reason=condition_str if condition_str else None,
        ))

    # Deferred to doctor
    deferred = whitelist.get_requires_doctor(protocol_key)

    # LLM refinement — only for ESI <= 3
    if triage_score.esi_level <= 3:
        class _LLMWorkupSuggestion(BaseModel):
            additional_tests: list[dict] = []

        system_prompt = (
            "You are an emergency medicine physician reviewing a preemptive workup plan. "
            "Suggest any additional tests that are clearly indicated but not yet ordered. "
            "Only suggest tests that are safe to order before a doctor sees the patient. "
            "Return a JSON object with additional_tests: a list of {test_name, reason, cost_tier, timing} dicts. "
            "cost_tier must be 'low', 'medium', or 'high'. "
            "timing must be 'immediate', 'urgent', or 'routine'. "
            "If no additional tests are needed, return additional_tests: []."
        )

        user_prompt = (
            f"Patient: {intake.age_years}y {intake.gender}\n"
            f"Chief complaint: {intake.chief_complaint.free_text_en}\n"
            f"Protocol: {protocol_key}\n"
            f"ESI level: {triage_score.esi_level}\n"
            f"Orders already placed: {[o.test_name for o in orders]}\n"
            f"Deferred to doctor: {deferred}\n"
            f"Medical history: {intake.medical_history.known_diagnoses}\n"
            "Suggest any additional tests. Be conservative — only add if clearly indicated."
        )

        try:
            raw = await llm.call(
                system=system_prompt,
                user=user_prompt,
                response_schema=_LLMWorkupSuggestion,
                temperature=0.1,
                max_tokens=2000,
            )
            for suggestion in raw.additional_tests:
                test_name = suggestion.get("test_name", "")
                if not test_name:
                    continue
                if not whitelist.is_allowed(protocol_key, test_name):
                    logger.debug(
                        "patient=%s LLM suggested %r not in whitelist — skipping",
                        intake.patient_id[:8], test_name,
                    )
                    continue
                orders.append(OrderItem(
                    test_name=test_name,
                    reason=suggestion.get("reason", "LLM suggestion"),
                    cost_tier=suggestion.get("cost_tier", "low"),
                    timing=suggestion.get("timing", "routine"),
                    whitelist_rule=f"{protocol_key}:llm",
                ))
        except Exception as exc:
            logger.warning("patient=%s LLM workup refinement failed: %s", intake.patient_id[:8], exc)

    reasoning = (
        f"Protocol: {protocol_key} — {len(orders)} orders, {len(deferred)} deferred to doctor"
    )

    latency_ms = int((time.time() - t0) * 1000)
    logger.info(
        "Workup | patient=%s protocol=%s orders=%d deferred=%d latency=%dms",
        intake.patient_id[:8],
        protocol_key,
        len(orders),
        len(deferred),
        latency_ms,
    )

    return WorkupPlan(
        patient_id=intake.patient_id,
        protocol_key=protocol_key,
        orders=orders,
        deferred_to_doctor=deferred,
        reasoning=reasoning,
    )
