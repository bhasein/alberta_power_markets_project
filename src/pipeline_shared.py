"""Shared provenance contracts for every pipeline stage."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


MANIFEST_VERSION = 2


def file_record(path: Path, *, hash_content: bool) -> dict[str, Any]:
    """Return stable identity fields for one source, code, or output file."""

    resolved = path.resolve()
    stat = resolved.stat()
    record: dict[str, Any] = {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_content:
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        record["sha256"] = digest.hexdigest()
    return record


def build_manifest(
    dataset: str,
    source_paths: Sequence[Path],
    code_paths: Sequence[Path],
    configuration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic provenance record for a pipeline stage."""

    required = [*source_paths, *code_paths]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing provenance files: " + ", ".join(str(path) for path in missing)
        )

    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset": dataset,
        "sources": [
            file_record(path, hash_content=False)
            for path in sorted(set(source_paths))
        ],
        "code": [
            file_record(path, hash_content=True)
            for path in sorted(set(code_paths))
        ],
        "configuration": dict(sorted((configuration or {}).items())),
    }


def manifest_path(output_path: Path) -> Path:
    """Return the provenance-manifest path associated with an artifact."""

    return output_path.with_suffix(output_path.suffix + ".manifest.json")


def output_is_current(
    output_path: Path,
    expected_manifest: Mapping[str, Any],
) -> bool:
    """Return whether an artifact and its exact provenance record are current."""

    provenance_path = manifest_path(output_path)
    if not output_path.exists() or not provenance_path.exists():
        return False
    try:
        observed = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        observed.get("pipeline") == expected_manifest
        and observed.get("artifact") == file_record(output_path, hash_content=False)
    )


def outputs_are_current(
    output_paths: Sequence[Path],
    expected_manifest: Mapping[str, Any],
) -> bool:
    """Return whether every requested artifact is current."""

    return all(
        output_is_current(output_path, expected_manifest)
        for output_path in output_paths
    )


def write_manifest(
    output_path: Path,
    manifest: Mapping[str, Any],
) -> Path:
    """Bind one artifact to the pipeline manifest that produced it."""

    provenance_path = manifest_path(output_path)
    payload = {
        "pipeline": dict(manifest),
        "artifact": file_record(output_path, hash_content=False),
    }
    provenance_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance_path


def write_manifests(
    output_paths: Sequence[Path],
    manifest: Mapping[str, Any],
) -> list[Path]:
    """Bind every supplied artifact to one pipeline manifest."""

    return [write_manifest(path, manifest) for path in output_paths]
