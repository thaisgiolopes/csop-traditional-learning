from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Callable, Iterator


_MEMORY_UNIT = "bytes"
_DEFAULT_MEASUREMENT_METHOD = "linux_procfs_resident_set_size"


def _read_process_rss_bytes() -> int:
    """Read current process RSS from Linux procfs, in bytes."""
    statm_path = Path("/proc/self/statm")

    try:
        fields = statm_path.read_text(encoding="ascii").split()
        resident_pages = int(fields[1])
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, IndexError, ValueError) as exc:
        raise RuntimeError(
            "Current-process RSS measurement requires readable "
            "/proc/self/statm and the SC_PAGE_SIZE system setting."
        ) from exc

    if resident_pages < 0 or page_size <= 0:
        raise RuntimeError("Process RSS or page size returned an invalid value.")

    return resident_pages * page_size


@dataclass
class MemoryMeasurement:
    """Records process RSS before and after a named stage."""

    stage_name: str
    memory_before_bytes: int | None = None
    memory_after_bytes: int | None = None
    memory_delta_bytes: int | None = None
    memory_unit: str = _MEMORY_UNIT
    measurement_method: str = _DEFAULT_MEASUREMENT_METHOD

    def to_dict(self) -> dict[str, str | int | None]:
        """Return the measurement as JSON-compatible data."""
        return asdict(self)


class ProcessMemoryMonitor:
    """Measure current-process resident memory around execution stages.

    On Linux, measurements come from `/proc/self/statm` and represent the
    process's resident set size (RSS), including resident Python and native
    library memory. RSS is not Python allocation tracking, and its delta is
    not necessarily memory exclusively consumed by the stage. Shared pages,
    allocator behavior, and operating-system accounting can affect readings.
    """

    def __init__(
        self,
        memory_reader: Callable[[], int] = _read_process_rss_bytes,
        measurement_method: str = _DEFAULT_MEASUREMENT_METHOD,
    ) -> None:
        if not callable(memory_reader):
            raise TypeError("memory_reader must be callable.")
        if not isinstance(measurement_method, str) or not measurement_method.strip():
            raise ValueError("measurement_method must be a non-empty string.")

        self._memory_reader = memory_reader
        self._measurement_method = measurement_method

    @contextmanager
    def measure(self, stage_name: str) -> Iterator[MemoryMeasurement]:
        """Measure current-process RSS before and after a named stage."""
        if not isinstance(stage_name, str) or not stage_name.strip():
            raise ValueError("stage_name must be a non-empty string.")

        measurement = MemoryMeasurement(
            stage_name=stage_name,
            measurement_method=self._measurement_method,
        )
        measurement.memory_before_bytes = self._read_memory()

        try:
            yield measurement
        finally:
            measurement.memory_after_bytes = self._read_memory()
            measurement.memory_delta_bytes = (
                measurement.memory_after_bytes
                - measurement.memory_before_bytes
            )

    def _read_memory(self) -> int:
        memory_bytes = self._memory_reader()

        if isinstance(memory_bytes, bool) or not isinstance(memory_bytes, int):
            raise TypeError("memory_reader must return an integer byte count.")
        if memory_bytes < 0:
            raise ValueError("memory_reader must not return a negative value.")

        return memory_bytes