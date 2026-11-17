from datetime import datetime, timedelta, timezone
from pluto.progress import estimate_progress


def test_resume_estimate_uses_only_current_session_throughput():
    now = datetime(2026, 11, 16, 12, 0, tzinfo=timezone(timedelta(hours=9)))
    first = now - timedelta(hours=3)
    result = estimate_progress(
        completed=6,
        total=10,
        session_completed=2,
        session_active=20.0,
        previous_active=30.0,
        first_started=first,
        now=now,
    )
    assert result["remaining_seconds"] == 40.0
    assert result["total_active_seconds"] == 50.0
    assert result["wall_seconds"] == 10800.0
    assert result["estimated_finish"] == "2026-11-16T12:00:40+09:00"
    assert not result["provisional"]


def test_initial_estimate_is_provisional_and_never_negative():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = estimate_progress(
        completed=1,
        total=1,
        session_completed=1,
        session_active=4.0,
        previous_active=0.0,
        first_started=now,
        now=now,
    )
    assert result["provisional"] and result["remaining_seconds"] == 0
