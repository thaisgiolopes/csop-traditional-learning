import time


class Timer:
    """Measure elapsed time in seconds using a monotonic high-resolution clock."""

    def __init__(self, stage_name: str | None = None) -> None:
        if stage_name is not None and (
            not isinstance(stage_name, str) or not stage_name.strip()
        ):
            raise ValueError("stage_name must be a non-empty string or None.")

        self.stage_name = stage_name
        self._started_at: float | None = None
        self._elapsed_seconds: float | None = None

    @property
    def running(self) -> bool:
        """Return whether the timer is currently running."""
        return self._started_at is not None

    @property
    def elapsed(self) -> float:
        """Return elapsed seconds, including time since start if still running."""
        if self._started_at is not None:
            return time.perf_counter() - self._started_at

        if self._elapsed_seconds is None:
            raise RuntimeError("Timer has not been started.")

        return self._elapsed_seconds

    @property
    def elapsed_seconds(self) -> float:
        """Return the elapsed duration in seconds."""
        return self.elapsed

    def start(self) -> None:
        """Start or restart the timer."""
        if self.running:
            raise RuntimeError("Timer is already running.")

        self._started_at = time.perf_counter()
        self._elapsed_seconds = None

    def stop(self) -> float:
        """Stop the timer and return the measured duration in seconds."""
        if self._started_at is None:
            raise RuntimeError("Timer has not been started.")

        self._elapsed_seconds = time.perf_counter() - self._started_at
        self._started_at = None
        return self._elapsed_seconds

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        self.stop()
        return False


def track_time(stage_name: str) -> Timer:
    """Create a context-manager timer labeled with an experiment stage name."""
    return Timer(stage_name=stage_name)