from collections.abc import Mapping
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
from typing import Any


ARTIFACT_CATEGORIES = frozenset(
    {
        "config",
        "environment",
        "logs",
        "graphs",
        "subgraphs",
        "datasets",
        "models",
        "predictions",
        "evaluation",
        "metadata",
    }
)


class ArtifactStore:
    """Store and locate files under one experiment-specific root directory."""

    def __init__(self, experiment_root: str | Path) -> None:
        root = Path(experiment_root)

        if not str(root).strip():
            raise ValueError("experiment_root must not be empty.")

        self._root = root.resolve()

    @property
    def experiment_root(self) -> Path:
        """Return the resolved experiment root directory."""
        return self._root

    def save_json(
        self,
        category: str,
        filename: str | Path,
        data: Any,
    ) -> Path:
        """Save deterministic, readable UTF-8 JSON without overwriting files."""
        destination = self._artifact_path(category, filename)
        serialized = json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )

        destination.parent.mkdir(parents=True, exist_ok=True)

        with destination.open("x", encoding="utf-8") as json_file:
            json_file.write(serialized)
            json_file.write("\n")

        return destination

    def save_text(
        self,
        category: str,
        filename: str | Path,
        content: str,
    ) -> Path:
        """Save UTF-8 text without overwriting an existing artifact."""
        if not isinstance(content, str):
            raise TypeError("content must be a string.")

        destination = self._artifact_path(category, filename)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with destination.open("x", encoding="utf-8") as text_file:
            text_file.write(content)

        return destination

    def copy_file(
        self,
        category: str,
        filename: str | Path,
        source_path: str | Path,
    ) -> Path:
        """Copy an existing file without overwriting an artifact."""
        source = Path(source_path)

        if not source.exists():
            raise FileNotFoundError(f"Source file does not exist: {source}")
        if not source.is_file():
            raise IsADirectoryError(f"Source path is not a file: {source}")

        destination = self._artifact_path(category, filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination_created = False

        try:
            with source.open("rb") as source_file:
                with destination.open("xb") as destination_file:
                    destination_created = True
                    shutil.copyfileobj(source_file, destination_file)
        except BaseException:
            if destination_created and destination.exists():
                destination.unlink()
            raise

        return destination

    def exists(self, category: str, filename: str | Path) -> bool:
        """Return whether the named artifact file exists."""
        return self._artifact_path(category, filename).is_file()

    def _artifact_path(
        self,
        category: str,
        filename: str | Path,
    ) -> Path:
        if not isinstance(category, str) or category not in ARTIFACT_CATEGORIES:
            raise ValueError(
                f"Unsupported artifact category: {category!r}."
            )

        if not isinstance(filename, (str, Path)):
            raise TypeError("filename must be a string or Path.")

        filename_text = str(filename)
        if not filename_text.strip():
            raise ValueError("filename must not be empty.")
        if "\\" in filename_text:
            raise ValueError("filename must use safe relative path separators.")

        relative_path = Path(filename_text)
        windows_path = PureWindowsPath(filename_text)

        if (
            relative_path.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or ".." in filename_text.split("/")
            or relative_path == Path(".")
        ):
            raise ValueError("filename must be a safe relative path.")

        destination = (self._root / category / relative_path).resolve()

        try:
            destination.relative_to(self._root)
        except ValueError as exc:
            raise ValueError(
                "Artifact destination must remain inside experiment_root."
            ) from exc

        return destination