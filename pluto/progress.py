"""Timezone-aware progress estimates based on current-session measured work."""

from datetime import timedelta


def estimate_progress(
    completed,
    total,
    session_completed,
    session_active,
    previous_active,
    first_started,
    now,
):
    if not (
        0 <= completed <= total
        and 0 < session_completed <= completed
        and session_active >= 0
        and previous_active >= 0
    ):
        raise ValueError("invalid progress or elapsed time")
    if first_started.utcoffset() is None or now.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    remaining = session_active / session_completed * (total - completed)
    return {
        "session_active_seconds": session_active,
        "total_active_seconds": previous_active + session_active,
        "wall_seconds": max(0.0, (now - first_started).total_seconds()),
        "remaining_seconds": remaining,
        "estimated_finish": (now + timedelta(seconds=remaining)).isoformat(),
        "provisional": session_completed == 1,
        "updated_at": now.isoformat(),
    }
