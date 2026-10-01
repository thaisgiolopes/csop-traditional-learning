import logging
import re

from src.experiments.logger import create_experiment_logger


def test_logger_writes_info_message_to_experiment_file(tmp_path):
    log_path = tmp_path / "experiment.log"
    logger = create_experiment_logger(
        "EXP_000001",
        log_path,
    )

    logger.info("Experiment started")

    contents = log_path.read_text(encoding="utf-8")
    assert "Experiment started" in contents


def test_logger_creates_nested_log_directory(tmp_path):
    log_path = tmp_path / "logs" / "experiment" / "run.log"
    logger = create_experiment_logger("EXP_NESTED", log_path)

    logger.info("Nested log created")

    assert log_path.is_file()
    assert "Nested log created" in log_path.read_text(encoding="utf-8")


def test_log_format_includes_timestamp_level_identifier_and_message(
    tmp_path,
):
    log_path = tmp_path / "formatted.log"
    logger = create_experiment_logger(
        "EXP_FORMAT",
        log_path,
    )

    logger.info("Loading graph")

    line = log_path.read_text(encoding="utf-8").strip()
    assert re.match(
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
        r" \| INFO \| EXP_FORMAT \| Loading graph$",
        line,
    )


def test_repeated_creation_does_not_duplicate_log_messages(tmp_path):
    log_path = tmp_path / "reconfigured.log"

    first_logger = create_experiment_logger("EXP_REPEAT", log_path)
    first_logger.info("Before reconfiguration")

    second_logger = create_experiment_logger("EXP_REPEAT", log_path)
    second_logger.info("After reconfiguration")

    contents = log_path.read_text(encoding="utf-8")
    assert contents.count("Before reconfiguration") == 1
    assert contents.count("After reconfiguration") == 1
    assert first_logger is second_logger


def test_info_level_filters_debug_messages(tmp_path):
    log_path = tmp_path / "info.log"
    logger = create_experiment_logger(
        "EXP_INFO",
        log_path,
        level=logging.INFO,
    )

    logger.debug("Hidden debug message")
    logger.info("Visible info message")

    contents = log_path.read_text(encoding="utf-8")
    assert "Hidden debug message" not in contents
    assert "Visible info message" in contents


def test_warning_and_error_messages_are_recorded(tmp_path):
    log_path = tmp_path / "warnings.log"
    logger = create_experiment_logger(
        "EXP_WARNINGS",
        log_path,
        level="DEBUG",
    )

    logger.warning("Potential issue")
    logger.error("Experiment failed")

    contents = log_path.read_text(encoding="utf-8")
    assert "| WARNING | EXP_WARNINGS | Potential issue" in contents
    assert "| ERROR | EXP_WARNINGS | Experiment failed" in contents


def test_console_logging_can_be_enabled_or_disabled_independently(
    tmp_path,
    capsys,
):
    disabled_path = tmp_path / "console-disabled.log"
    disabled_logger = create_experiment_logger(
        "EXP_NO_CONSOLE",
        disabled_path,
        console=False,
    )
    disabled_logger.info("File only")
    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == ""
    assert "File only" in disabled_path.read_text(encoding="utf-8")

    enabled_path = tmp_path / "console-enabled.log"
    enabled_logger = create_experiment_logger(
        "EXP_WITH_CONSOLE",
        enabled_path,
        console=True,
    )
    enabled_logger.info("File and console")
    captured = capsys.readouterr()

    assert "File and console" in captured.err
    assert "File and console" in enabled_path.read_text(encoding="utf-8")


def test_different_experiments_write_to_separate_files(tmp_path):
    first_path = tmp_path / "first.log"
    second_path = tmp_path / "second.log"

    first_logger = create_experiment_logger("EXP_FIRST", first_path)
    second_logger = create_experiment_logger("EXP_SECOND", second_path)

    first_logger.info("First experiment")
    second_logger.info("Second experiment")

    first_contents = first_path.read_text(encoding="utf-8")
    second_contents = second_path.read_text(encoding="utf-8")

    assert "EXP_FIRST | First experiment" in first_contents
    assert "Second experiment" not in first_contents
    assert "EXP_SECOND | Second experiment" in second_contents
    assert "First experiment" not in second_contents