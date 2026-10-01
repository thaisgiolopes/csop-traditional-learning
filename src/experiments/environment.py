from dataclasses import asdict, dataclass
import json
from importlib import metadata
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
from typing import Any


def _run_git(repository_path: Path, *arguments: str) -> str | None:
    """Run one Git command safely and return its trimmed output."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_path), *arguments],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None

    return result.stdout.strip()


def _collect_git_metadata(
    repository_path: Path,
) -> tuple[str | None, str | None, bool | None, Path | None]:
    """Collect Git metadata, returning unknown values outside a Git repo."""
    repository_root_value = _run_git(
        repository_path,
        "rev-parse",
        "--show-toplevel",
    )
    if repository_root_value is None:
        return None, None, None, None

    repository_root = Path(repository_root_value)
    commit = _run_git(repository_root, "rev-parse", "HEAD")
    branch = _run_git(repository_root, "branch", "--show-current")
    status = _run_git(
        repository_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )

    return (
        commit or None,
        branch or None,
        None if status is None else bool(status),
        repository_root,
    )


def _read_requirement_names(requirements_path: Path) -> list[str]:
    """Read package names from the repository's requirements file."""
    try:
        lines = requirements_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    package_names = set()

    for line in lines:
        requirement = line.partition("#")[0].strip()
        if not requirement or requirement.startswith("-"):
            continue

        match = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
        if match:
            package_names.add(match.group(1))

    return sorted(package_names, key=str.casefold)


def _collect_dependency_versions(
    project_root: Path,
) -> dict[str, str | None]:
    """Resolve installed versions for dependencies declared by the project."""
    requirements_path = project_root / "requirements.txt"
    package_names = _read_requirement_names(requirements_path)
    versions: dict[str, str | None] = {}

    for package_name in package_names:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None

    return versions


@dataclass
class EnvironmentMetadata:
    """Records software and host metadata relevant to experiment reproduction."""

    python_version: str
    operating_system: str
    os_release: str
    machine_architecture: str
    processor: str | None
    hostname: str
    repository_git_commit: str | None
    repository_git_branch: str | None
    repository_dirty: bool | None
    dependency_versions: dict[str, str | None]

    @classmethod
    def collect(
        cls,
        repository_path: str | Path | None = None,
    ) -> "EnvironmentMetadata":
        """Collect environment and repository metadata for the current run."""
        if repository_path is None:
            repository_path = Path(__file__).resolve().parents[2]
        else:
            repository_path = Path(repository_path).resolve()

        commit, branch, dirty, git_root = _collect_git_metadata(
            repository_path
        )
        project_root = git_root or repository_path

        try:
            hostname = socket.gethostname() or "unknown"
        except OSError:
            hostname = "unknown"

        processor = platform.processor() or None

        return cls(
            python_version=sys.version.split()[0],
            operating_system=platform.system() or "unknown",
            os_release=platform.release() or "unknown",
            machine_architecture=platform.machine() or "unknown",
            processor=processor,
            hostname=hostname,
            repository_git_commit=commit,
            repository_git_branch=branch,
            repository_dirty=dirty,
            dependency_versions=_collect_dependency_versions(project_root),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return metadata as JSON-compatible data."""
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        """Save metadata as readable UTF-8 JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as json_file:
            json.dump(
                self.to_dict(),
                json_file,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            json_file.write("\n")