import logging
from pathlib import Path


_LOGGER_NAMESPACE = "csop.experiments"
_HANDLER_MARKER = "_csop_experiment_logger_handler"
_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(experiment_id)s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _ExperimentIdFilter(logging.Filter):
    """Attach an experiment identifier to log records."""

    def __init__(self, experiment_id: str) -> None:
        super().__init__()
        self._experiment_id = experiment_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.experiment_id = self._experiment_id
        return True


def _resolve_level(level: int | str) -> int:
    """Resolve a standard logging level name or integer."""
    if isinstance(level, bool):
        raise TypeError("level must be a logging level name or integer.")

    if isinstance(level, int):
        return level

    if isinstance(level, str):
        resolved_level = logging.getLevelName(level.upper())
        if isinstance(resolved_level, int):
            return resolved_level

    raise ValueError(f"Unknown logging level: {level!r}.")


def create_experiment_logger(
    experiment_id: str,
    log_file_path: str | Path,
    level: int | str = logging.INFO,
    console: bool = False,
) -> logging.Logger:
    """Create or reconfigure a dedicated logger for one experiment."""
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment_id must be a non-empty string.")

    if not isinstance(console, bool):
        raise TypeError("console must be a boolean.")

    resolved_level = _resolve_level(level)
    output_path = Path(log_file_path)

    if not str(output_path).strip():
        raise ValueError("log_file_path must not be empty.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(
        f"{_LOGGER_NAMESPACE}.{experiment_id}"
    )
    logger.setLevel(resolved_level)
    logger.propagate = False

    # Replace only handlers created by this function. Any unrelated handlers
    # on the named logger are left untouched.
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        fmt=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
    )
    experiment_filter = _ExperimentIdFilter(experiment_id)

    file_handler = logging.FileHandler(
        output_path,
        encoding="utf-8",
    )
    file_handler.setLevel(resolved_level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(experiment_filter)
    setattr(file_handler, _HANDLER_MARKER, True)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(experiment_filter)
        setattr(console_handler, _HANDLER_MARKER, True)
        logger.addHandler(console_handler)

    return logger