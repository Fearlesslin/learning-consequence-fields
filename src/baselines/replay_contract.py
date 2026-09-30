from __future__ import annotations

from typing import Any


EPSILON = 1e-9

CONTRACT_COLUMNS = [
    "target_service_fraction",
    "deadline_service_fraction",
    "delivered_service_fraction",
]

COMMITMENT_COLUMNS = [
    "first_positive_execution_time",
    "committed_target",
    "committed_target_attainment_ms",
    "target_at_deadline",
    "maximum_target_after_first_service",
    "minimum_target_after_first_service",
    "post_start_downward_revision_count",
    "post_start_promotion_count",
    "last_downward_revision_ms_before_deadline",
    "commitment_breach",
    "committed_target_completed_by_deadline",
    "final_target_completed_by_deadline",
]


def target_work_ms(
    intrinsic_actual_ms: float,
    target_fraction: float,
    *,
    task_class: str,
) -> float:
    actual = float(intrinsic_actual_ms)
    fraction = float(target_fraction)
    if actual <= 0.0:
        raise ValueError("intrinsic actual execution must be positive")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("target service fraction must lie in [0, 1]")
    if task_class == "hard_rt" and fraction < 1.0 - EPSILON:
        raise ValueError("hard-RT work cannot be degraded")
    return actual * fraction


def delivered_fraction(consumed_ms: float, intrinsic_actual_ms: float) -> float:
    actual = float(intrinsic_actual_ms)
    if actual <= 0.0:
        raise ValueError("intrinsic actual execution must be positive")
    return min(max(float(consumed_ms) / actual, 0.0), 1.0)


def contract_fields(
    *,
    intrinsic_actual_ms: float,
    consumed_ms: float,
    target_fraction: float,
    deadline_fraction: float | None,
) -> dict[str, float]:
    delivered = delivered_fraction(consumed_ms, intrinsic_actual_ms)
    deadline_service = delivered if deadline_fraction is None else float(deadline_fraction)
    return {
        "target_service_fraction": float(target_fraction),
        "deadline_service_fraction": min(max(deadline_service, 0.0), 1.0),
        "delivered_service_fraction": delivered,
    }


def snapshot_deadline(row: dict[str, Any], now_ms: float) -> None:
    if row.get("_deadline_service_fraction") is not None:
        return
    if float(row["absolute_deadline_ms"]) > float(now_ms) + EPSILON:
        return
    row["_deadline_service_fraction"] = delivered_fraction(
        float(row.get("_consumed_ms", 0.0)),
        float(row["_intrinsic_ms"]),
    )


def initialize_target_timeline(row: dict[str, Any]) -> None:
    row["_first_positive_execution_time"] = None
    row["_committed_target"] = None
    row["_committed_target_attainment_ms"] = None
    row["_target_at_deadline"] = None
    row["_maximum_target_after_first_service"] = None
    row["_minimum_target_after_first_service"] = None
    row["_post_start_downward_revision_count"] = 0
    row["_post_start_promotion_count"] = 0
    row["_last_post_start_downward_revision_ms_before_deadline"] = None
    row["_commitment_breach"] = False


def record_first_positive_execution(
    row: dict[str, Any],
    *,
    now_ms: float,
    target_fraction: float,
) -> None:
    if row.get("_first_positive_execution_time") is not None:
        return
    target = float(target_fraction)
    if not 0.0 <= target <= 1.0:
        raise ValueError("committed target must lie in [0, 1]")
    row["_first_positive_execution_time"] = float(now_ms)
    row["_committed_target"] = target
    row["_maximum_target_after_first_service"] = target
    row["_minimum_target_after_first_service"] = target


def record_committed_target_progress(
    row: dict[str, Any],
    *,
    interval_start_ms: float,
    consumed_before_ms: float,
    consumed_after_ms: float,
    intrinsic_actual_ms: float,
    tolerance: float = EPSILON,
) -> None:
    if row.get("_committed_target_attainment_ms") is not None:
        return
    committed = row.get("_committed_target")
    if committed is None or float(committed) <= tolerance:
        return
    threshold = float(intrinsic_actual_ms) * float(committed)
    before = float(consumed_before_ms)
    after = float(consumed_after_ms)
    if before + tolerance >= threshold:
        row["_committed_target_attainment_ms"] = float(interval_start_ms)
        return
    if after + tolerance < threshold:
        return
    row["_committed_target_attainment_ms"] = float(interval_start_ms) + max(
        threshold - before, 0.0
    )


def record_post_start_target_revision(
    row: dict[str, Any],
    *,
    previous_target: float,
    new_target: float,
    now_ms: float,
    tolerance: float = EPSILON,
) -> None:
    if row.get("_first_positive_execution_time") is None:
        return
    previous = float(previous_target)
    current = float(new_target)
    if abs(previous - current) <= tolerance:
        return
    row["_maximum_target_after_first_service"] = max(
        float(row["_maximum_target_after_first_service"]), current
    )
    row["_minimum_target_after_first_service"] = min(
        float(row["_minimum_target_after_first_service"]), current
    )
    if current < previous - tolerance:
        row["_post_start_downward_revision_count"] += 1
        deadline = float(row["absolute_deadline_ms"])
        if float(now_ms) <= deadline + tolerance:
            row["_last_post_start_downward_revision_ms_before_deadline"] = max(
                0.0, deadline - float(now_ms)
            )
        if current < float(row["_committed_target"]) - tolerance:
            row["_commitment_breach"] = True
    elif current > previous + tolerance:
        row["_post_start_promotion_count"] += 1


def snapshot_target_at_deadline(
    row: dict[str, Any],
    *,
    now_ms: float,
    target_fraction: float,
) -> None:
    if row.get("_target_at_deadline") is not None:
        return
    if float(row["absolute_deadline_ms"]) > float(now_ms) + EPSILON:
        return
    row["_target_at_deadline"] = float(target_fraction)


def target_timeline_fields(
    row: dict[str, Any],
    *,
    final_target: float,
    deadline_service_fraction: float,
    tolerance: float = EPSILON,
) -> dict[str, Any]:
    final = float(final_target)
    deadline_service = float(deadline_service_fraction)
    committed = row.get("_committed_target")
    target_at_deadline = row.get("_target_at_deadline")
    if target_at_deadline is None:
        target_at_deadline = final
    committed_completed = (
        committed is not None
        and float(committed) > tolerance
        and deadline_service + tolerance >= float(committed)
    )
    final_completed = (
        final > tolerance and deadline_service + tolerance >= final
    )
    return {
        "first_positive_execution_time": row.get(
            "_first_positive_execution_time"
        ),
        "committed_target": committed,
        "committed_target_attainment_ms": row.get(
            "_committed_target_attainment_ms"
        ),
        "target_at_deadline": float(target_at_deadline),
        "maximum_target_after_first_service": row.get(
            "_maximum_target_after_first_service"
        ),
        "minimum_target_after_first_service": row.get(
            "_minimum_target_after_first_service"
        ),
        "post_start_downward_revision_count": int(
            row.get("_post_start_downward_revision_count", 0)
        ),
        "post_start_promotion_count": int(
            row.get("_post_start_promotion_count", 0)
        ),
        "last_downward_revision_ms_before_deadline": row.get(
            "_last_post_start_downward_revision_ms_before_deadline"
        ),
        "commitment_breach": bool(row.get("_commitment_breach", False)),
        "committed_target_completed_by_deadline": bool(committed_completed),
        "final_target_completed_by_deadline": bool(final_completed),
    }
