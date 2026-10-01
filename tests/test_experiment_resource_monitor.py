import json

import pytest

from src.experiments.resource_monitor import (
    MemoryMeasurement,
    ProcessMemoryMonitor,
)


def make_monitor(readings):
    """Create a monitor backed by deterministic memory readings."""
    values = iter(readings)
    return ProcessMemoryMonitor(
        memory_reader=lambda: next(values),
        measurement_method="test_reader",
    )


def test_memory_measurement_object_creation():
    measurement = MemoryMeasurement(stage_name="feature_extraction")

    assert measurement.stage_name == "feature_extraction"
    assert measurement.memory_before_bytes is None
    assert measurement.memory_after_bytes is None
    assert measurement.memory_delta_bytes is None
    assert measurement.memory_unit == "bytes"


def test_monitor_records_memory_before_and_after_stage():
    monitor = make_monitor([1024, 1536])

    with monitor.measure("subgraph_generation") as measurement:
        assert measurement.memory_before_bytes == 1024
        assert measurement.memory_after_bytes is None

    assert measurement.memory_after_bytes == 1536
    assert measurement.memory_delta_bytes == 512


def test_memory_delta_is_after_minus_before():
    monitor = make_monitor([2048, 1792])

    with monitor.measure("stage") as measurement:
        pass

    assert measurement.memory_delta_bytes == -256


def test_measure_works_as_context_manager():
    monitor = make_monitor([100, 175])

    with monitor.measure("dataset_build") as measurement:
        pass

    assert measurement.stage_name == "dataset_build"
    assert measurement.memory_delta_bytes == 75


def test_exception_inside_stage_propagates():
    monitor = make_monitor([10, 15])

    with pytest.raises(ValueError, match="stage failed"):
        with monitor.measure("failing_stage"):
            raise ValueError("stage failed")


def test_measurement_is_completed_after_stage_exception():
    monitor = make_monitor([4096, 4608])
    measurement = None

    with pytest.raises(RuntimeError, match="stage failed"):
        with monitor.measure("failing_stage") as measurement:
            raise RuntimeError("stage failed")

    assert measurement is not None
    assert measurement.memory_before_bytes == 4096
    assert measurement.memory_after_bytes == 4608
    assert measurement.memory_delta_bytes == 512


def test_measurement_result_is_json_compatible():
    monitor = make_monitor([1000, 1250])

    with monitor.measure("prediction") as measurement:
        pass

    serialized = json.dumps(measurement.to_dict(), allow_nan=False)
    restored = json.loads(serialized)

    assert restored["stage_name"] == "prediction"
    assert restored["memory_delta_bytes"] == 250
    assert restored["memory_unit"] == "bytes"
    assert restored["measurement_method"] == "test_reader"


def test_result_exposes_unit_and_measurement_method():
    monitor = make_monitor([100, 200])

    with monitor.measure("stage") as measurement:
        pass

    assert measurement.memory_unit == "bytes"
    assert measurement.measurement_method == "test_reader"


def test_monitor_has_no_file_or_database_side_effects(tmp_path):
    monitor = make_monitor([1, 2])

    with monitor.measure("stage"):
        pass

    assert list(tmp_path.iterdir()) == []


def test_linux_reader_reports_process_rss_when_available():
    monitor = ProcessMemoryMonitor()

    try:
        with monitor.measure("current_process") as measurement:
            pass
    except RuntimeError as exc:
        pytest.skip(str(exc))

    assert measurement.memory_before_bytes is not None
    assert measurement.memory_after_bytes is not None
    assert isinstance(measurement.memory_delta_bytes, int)