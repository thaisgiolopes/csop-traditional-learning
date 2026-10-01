import json
from pathlib import Path
import sqlite3

import pytest

from src.experiments.artifact_store import ArtifactStore


def test_store_initialization_does_not_create_category_directories(tmp_path):
    experiment_root = tmp_path / "experiments" / "EXP_000001"

    store = ArtifactStore(experiment_root)

    assert store.experiment_root == experiment_root.resolve()
    assert not experiment_root.exists()


def test_save_json_returns_path_and_writes_json(tmp_path):
    store = ArtifactStore(tmp_path / "EXP_JSON")

    artifact_path = store.save_json(
        "config",
        "experiment.json",
        {"seed": 42, "name": "baseline"},
    )

    assert artifact_path.is_file()
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == {
        "name": "baseline",
        "seed": 42,
    }


def test_save_json_is_deterministic_and_rejects_nan(tmp_path):
    store = ArtifactStore(tmp_path / "EXP_JSON")

    artifact_path = store.save_json(
        "metadata",
        "values.json",
        {"z": 1, "a": 2},
    )

    assert artifact_path.read_text(encoding="utf-8").startswith(
        '{\n  "a": 2,'
    )

    with pytest.raises(ValueError):
        store.save_json("metadata", "invalid.json", {"value": float("nan")})


def test_save_text_preserves_utf8_content(tmp_path):
    store = ArtifactStore(tmp_path / "EXP_TEXT")
    content = "Etapa concluída: geração de subgrafos."

    artifact_path = store.save_text("logs", "run.txt", content)

    assert artifact_path.read_text(encoding="utf-8") == content


def test_copy_existing_file(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"\x00artifact-data")
    store = ArtifactStore(tmp_path / "EXP_COPY")

    artifact_path = store.copy_file(
        "models",
        "model.bin",
        source,
    )

    assert artifact_path.read_bytes() == source.read_bytes()


def test_returned_artifact_paths_are_inside_experiment_root(tmp_path):
    root = tmp_path / "EXP_PATHS"
    store = ArtifactStore(root)

    artifact_path = store.save_text("logs", "run.txt", "started")

    assert artifact_path.is_relative_to(root.resolve())


def test_nested_artifact_directories_are_created_on_save(tmp_path):
    store = ArtifactStore(tmp_path / "EXP_NESTED")

    artifact_path = store.save_text(
        "logs",
        "stages/generation/run.txt",
        "done",
    )

    assert artifact_path.is_file()
    assert artifact_path.parent.name == "generation"


@pytest.mark.parametrize(
    ("category", "filename"),
    [
        ("unknown", "file.txt"),
        ("../outside", "file.txt"),
        ("logs", "../outside.txt"),
        ("logs", "nested/../../outside.txt"),
        ("logs", "/tmp/outside.txt"),
        ("logs", r"..\outside.txt"),
        ("logs", r"C:\outside.txt"),
    ],
)
def test_invalid_or_traversing_paths_are_rejected(
    tmp_path,
    category,
    filename,
):
    store = ArtifactStore(tmp_path / "EXP_SAFE")

    with pytest.raises(ValueError):
        store.save_text(category, filename, "blocked")


def test_nonexistent_copy_source_is_rejected(tmp_path):
    store = ArtifactStore(tmp_path / "EXP_SOURCE")

    with pytest.raises(FileNotFoundError):
        store.copy_file(
            "graphs",
            "graph.dat",
            tmp_path / "missing.dat",
        )


@pytest.mark.parametrize("operation", ["json", "text", "copy"])
def test_existing_artifact_is_not_overwritten(tmp_path, operation):
    store = ArtifactStore(tmp_path / "EXP_NO_OVERWRITE")

    if operation == "json":
        save = lambda: store.save_json("metadata", "item.json", {"value": 1})
    elif operation == "text":
        save = lambda: store.save_text("logs", "item.txt", "first")
    else:
        source = tmp_path / "source.txt"
        source.write_text("source", encoding="utf-8")
        save = lambda: store.copy_file("datasets", "item.txt", source)

    artifact_path = save()

    with pytest.raises(FileExistsError):
        save()

    assert artifact_path.is_file()


def test_saved_json_can_be_loaded_again(tmp_path):
    store = ArtifactStore(tmp_path / "EXP_ROUND_TRIP")
    expected = {"features": ["degree_mean", "degree_max"], "seed": 17}

    artifact_path = store.save_json("config", "run.json", expected)

    with artifact_path.open("r", encoding="utf-8") as json_file:
        loaded = json.load(json_file)

    assert loaded == expected
    assert store.exists("config", "run.json")


def test_store_has_no_database_side_effects(tmp_path):
    experiment_root = tmp_path / "EXP_NO_DATABASE"
    store = ArtifactStore(experiment_root)
    store.save_json("metadata", "record.json", {"status": "complete"})

    database_files = list(experiment_root.rglob("*.db"))
    sqlite_files = list(experiment_root.rglob("*.sqlite"))
    sqlite3_files = list(experiment_root.rglob("*.sqlite3"))

    assert database_files == []
    assert sqlite_files == []
    assert sqlite3_files == []
    assert sqlite3 is not None