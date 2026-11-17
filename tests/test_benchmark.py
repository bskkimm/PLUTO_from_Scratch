import math
from pluto.benchmark import benchmark


def test_benchmark_measures_real_loader_and_optimizer_steps():
    result = benchmark(iterations=2, warmup=1, batch_size=2)
    assert set(result["paths"]) == {"loader", "cached_step", "end_to_end"}
    for row in result["paths"].values():
        assert row["samples_per_second"] > 0
        assert row["seconds"] > 0
    assert math.isfinite(result["paths"]["cached_step"]["last_loss"])
    assert math.isfinite(result["paths"]["end_to_end"]["last_loss"])
    assert result["paths"]["end_to_end"]["loader_wait_seconds"] >= 0


def test_benchmark_records_code_hashes():
    result = benchmark(iterations=1, warmup=0, batch_size=1)
    assert len(result["source_sha256"]["pluto/model.py"]) == 64
    assert len(result["source_sha256"]["pluto/benchmark.py"]) == 64
