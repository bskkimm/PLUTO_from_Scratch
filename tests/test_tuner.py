import json
from pluto.tune import tune


def test_bounded_tuner_records_trials_and_valid_choice(tmp_path):
    out = tmp_path / "trials.json"
    result = tune(
        out, threads=(1, 2), iterations=2, warmup=1, repeats=2, max_seconds=30
    )
    saved = json.loads(out.read_text())
    assert saved == result
    assert saved["trials"][0]["threads"] == 1
    assert saved["selected_threads"] in [
        row["threads"] for row in saved["trials"] if row["accepted"]
    ]
    assert saved["effective_batch_size"] == 2
    assert saved["iterations"] == 2 and saved["warmup_steps"] == 1
    assert saved["environment"]["torch"]
    assert all("reason" in row for row in saved["trials"])
