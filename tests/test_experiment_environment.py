import json
import platform
from pathlib import Path
import subprocess

from src.experiments.environment import EnvironmentMetadata
import src.experiments.environment as environment_module


def make_metadata() -> EnvironmentMetadata:
    """Create environment metadata with representative values."""
    return EnvironmentMetadata(
        python_version="3.12.0",
        operating_system="Linux",
        os_release="test-release",
        machine_architecture="x86_64",
        processor="test-processor",
        hostname="test-host",
        repository_git_commit="abc123",
        repository_git_branch="main",
        repository_dirty=False,
        dependency_versions={
            "lightgbm": "4.0.0",
            "networkx": "3.0",
        },
    )


def test_environment_metadata_object_creation():
    environment = make_metadata()

    assert environment.python_version == "3.12.0"
    assert environment.operating_system == "Linux"
    assert environment.repository_git_commit == "abc123"
    assert environment.repository_dirty is False


def test_environment_metadata_to_dict_is_json_compatible():
    environment_dict = make_metadata().to_dict()

    encoded = json.dumps(environment_dict, allow_nan=False)

    assert json.loads(encoded) == environment_dict
    assert environment_dict["dependency_versions"]["lightgbm"] == "4.0.0"


def test_environment_metadata_saves_readable_utf8_json(tmp_path):
    output_path = tmp_path / "metadata" / "environment.json"
    environment = make_metadata()

    environment.save_json(output_path)

    assert output_path.is_file()
    saved_data = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_data == environment.to_dict()
    assert "\n" in output_path.read_text(encoding="utf-8")


def test_git_metadata_is_unknown_when_git_command_is_unavailable(
    monkeypatch,
    tmp_path,
):
    def unavailable_git(*args, **kwargs):
        raise FileNotFoundError("git executable not found")

    monkeypatch.setattr(
        environment_module.subprocess,
        "run",
        unavailable_git,
    )

    environment = EnvironmentMetadata.collect(repository_path=tmp_path)

    assert environment.repository_git_commit is None
    assert environment.repository_git_branch is None
    assert environment.repository_dirty is None


def test_collection_includes_python_and_operating_system_information(
    tmp_path,
):
    environment = EnvironmentMetadata.collect(repository_path=tmp_path)

    assert environment.python_version
    assert environment.operating_system == platform.system()
    assert environment.os_release == platform.release()
    assert environment.machine_architecture == platform.machine()
    assert environment.hostname