"""Shared contracts for preprocessing pipelines.

This module centralizes provenance, duplicate resolution, and audit-result
construction so every preprocessing dataset follows the same rules.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline_shared import (
    MANIFEST_VERSION,
    build_manifest,
    outputs_are_current,
    write_manifests,
)

DUPLICATE_STATS_ATTR = "preprocessing_duplicate_stats"


class DuplicateConflictError(ValueError):
    """Raised when multiple source observations disagree for one key."""

    def __init__(
        self,
        dataset_name: str,
        exact_duplicate_rows: int,
        conflicting_keys: int,
        examples: list[dict[str, Any]],
    ) -> None:
        self.dataset_name = dataset_name
        self.exact_duplicate_rows = exact_duplicate_rows
        self.conflicting_keys = conflicting_keys
        self.examples = examples
        super().__init__(
            f"{dataset_name} contains {conflicting_keys} conflicting duplicate "
            f"keys; examples={examples}"
        )


def add_check(
    rows: list[dict[str, Any]],
    check: str,
    passed: bool,
    observed: Any = None,
    expected: Any = None,
    severity: str = "error",
    notes: str = "",
    **extra: Any,
) -> None:
    """Append one standardized audit result to ``rows``."""

    rows.append(
        {
            **extra,
            "check": check,
            "pass": bool(passed),
            "severity": severity,
            "observed": observed,
            "expected": expected,
            "notes": notes,
        }
    )


def deduplicate_or_raise(
    frame: pd.DataFrame,
    keys: Sequence[str],
    *,
    ignore_columns: Iterable[str] = (),
    dataset_name: str,
) -> tuple[pd.DataFrame, int]:
    """Collapse exact duplicates and reject conflicting duplicate keys.

    Columns listed in ``ignore_columns`` are treated as provenance rather than
    values. They do not make otherwise identical observations conflict.
    """

    if frame.empty:
        return frame.copy(), 0

    missing = sorted(set(keys) - set(frame.columns))
    if missing:
        raise ValueError(f"{dataset_name} is missing duplicate keys: {missing}")

    ignored = set(ignore_columns)
    value_columns = [
        column
        for column in frame.columns
        if column not in set(keys) | ignored
    ]
    comparison_columns = [*keys, *value_columns]
    exact_mask = frame.duplicated(subset=comparison_columns, keep="first")
    exact_count = int(exact_mask.sum())
    deduplicated = frame.loc[~exact_mask].copy()

    conflicting = deduplicated.duplicated(subset=list(keys), keep=False)
    if conflicting.any():
        conflicting_key_frame = deduplicated.loc[
            conflicting,
            list(keys),
        ].drop_duplicates()
        raise DuplicateConflictError(
            dataset_name=dataset_name,
            exact_duplicate_rows=exact_count,
            conflicting_keys=len(conflicting_key_frame),
            examples=conflicting_key_frame.head(5).to_dict("records"),
        )

    return deduplicated.reset_index(drop=True), exact_count


def set_duplicate_stats(
    frame: pd.DataFrame,
    *,
    exact_duplicate_rows: int,
    conflicting_keys: int = 0,
) -> pd.DataFrame:
    """Attach source-level duplicate statistics to a cleaned frame."""

    frame.attrs[DUPLICATE_STATS_ATTR] = {
        "exact_duplicate_rows": int(exact_duplicate_rows),
        "conflicting_keys": int(conflicting_keys),
    }
    return frame


def add_duplicate_checks(
    rows: list[dict[str, Any]],
    frame: pd.DataFrame,
) -> None:
    """Record duplicate collapse and conflict counts in an audit table."""

    stats = frame.attrs.get(
        DUPLICATE_STATS_ATTR,
        {"exact_duplicate_rows": 0, "conflicting_keys": 0},
    )
    add_check(
        rows,
        "exact_duplicate_source_rows_collapsed",
        True,
        stats["exact_duplicate_rows"],
        "recorded",
        severity="info",
    )
    add_check(
        rows,
        "conflicting_duplicate_source_keys",
        stats["conflicting_keys"] == 0,
        stats["conflicting_keys"],
        0,
    )


def audit_passes(audit_df: pd.DataFrame) -> bool:
    """Return whether every error-severity audit check passes."""

    error_checks = audit_df.loc[
        audit_df["severity"].eq("error"),
        "pass",
    ]
    return bool(error_checks.all()) if not error_checks.empty else True


def duplicate_failure_audit(error: DuplicateConflictError) -> pd.DataFrame:
    """Build audit evidence for a duplicate-conflict pipeline failure."""

    rows: list[dict[str, Any]] = []
    add_check(
        rows,
        "exact_duplicate_source_rows_collapsed",
        True,
        error.exact_duplicate_rows,
        "recorded",
        severity="info",
    )
    add_check(
        rows,
        "conflicting_duplicate_source_keys",
        False,
        error.conflicting_keys,
        0,
        notes=f"examples={error.examples}",
    )
    return pd.DataFrame(rows)


def preprocessing_code_paths(module_path: Path) -> list[Path]:
    """Return the code and configuration files governing one stage."""

    module_path = module_path.resolve()
    return [
        module_path,
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[1] / "pipeline_shared.py",
        module_path.parents[1] / "config.py",
    ]


def write_audit_artifacts(
    artifacts: Mapping[Path, pd.DataFrame],
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """Write audit tables and optional matching provenance manifests."""

    for path, frame in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    if manifest is not None:
        write_manifests(list(artifacts), manifest)


def write_tabular_outputs(
    frame: pd.DataFrame,
    *,
    parquet_path: Path,
    csv_path: Path | None,
    manifest: Mapping[str, Any],
    provenance_artifacts: Sequence[Path] = (),
) -> None:
    """Write canonical tabular outputs and bind all artifacts to provenance."""

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)
    written = [parquet_path]
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(csv_path, index=False)
        written.append(csv_path)
    write_manifests([*written, *provenance_artifacts], manifest)
