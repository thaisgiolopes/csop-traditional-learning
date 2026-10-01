import pytest

import src.experiments.timer as timer_module
from src.experiments.timer import Timer, track_time


def test_timer_starts_and_stops_correctly(monkeypatch):
    clock_values = iter([10.0, 12.5])
    monkeypatch.setattr(
        timer_module.time,
        "perf_counter",
        lambda: next(clock_values),
    )
    timer = Timer()

    timer.start()

    assert timer.running is True
    assert timer.stop() == 2.5
    assert timer.running is False
    assert timer.elapsed == 2.5


def test_timer_measures_positive_elapsed_time():
    timer = Timer()

    with timer:
        sum(range(100_000))

    assert timer.elapsed > 0


def test_timer_cannot_stop_before_start():
    timer = Timer()

    with pytest.raises(RuntimeError, match="not been started"):
        timer.stop()


def test_timer_cannot_start_twice_while_running():
    timer = Timer()
    timer.start()

    with pytest.raises(RuntimeError, match="already running"):
        timer.start()

    assert timer.running is True
    timer.stop()


def test_context_manager_records_elapsed_seconds(monkeypatch):
    clock_values = iter([3.0, 3.75])
    monkeypatch.setattr(
        timer_module.time,
        "perf_counter",
        lambda: next(clock_values),
    )

    with Timer() as timer:
        pass

    assert timer.running is False
    assert timer.elapsed == 0.75
    assert timer.elapsed_seconds == 0.75


def test_exception_propagates_from_context_manager():
    timer = Timer()

    with pytest.raises(ValueError, match="measured operation failed"):
        with timer:
            raise ValueError("measured operation failed")


def test_timer_stops_and_records_duration_after_exception(monkeypatch):
    clock_values = iter([7.0, 8.0])
    monkeypatch.setattr(
        timer_module.time,
        "perf_counter",
        lambda: next(clock_values),
    )
    timer = Timer()

    with pytest.raises(ValueError):
        with timer:
            raise ValueError("operation failed")

    assert timer.running is False
    assert timer.elapsed == 1.0


def test_named_stage_measurement_exposes_name_and_duration(monkeypatch):
    clock_values = iter([20.0, 20.125])
    monkeypatch.setattr(
        timer_module.time,
        "perf_counter",
        lambda: next(clock_values),
    )

    with track_time("subgraph_generation") as measurement:
        pass

    assert measurement.stage_name == "subgraph_generation"
    assert measurement.elapsed_seconds == 0.125


def test_timer_does_not_create_files_or_logging_side_effects(tmp_path):
    with Timer() as timer:
        sum(range(100))

    assert timer.elapsed >= 0
    assert list(tmp_path.iterdir()) == []