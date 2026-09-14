"""Normalize, audit, and integrate the AESO API research archive.

Detailed event and reference products remain separate. Information that can
be represented honestly at hourly grain is also assembled into one continuous
``aeso_api_dataset_hourly.parquet`` for the canonical master. Duplicate actual
AIL and pool-price fields are retained only for reconciliation, not copied into
the mergeable dataset. Timing-sensitive and ex-post fields remain explicitly
labelled in the field register.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    AESO_API_AUDITS_DIR,
    AESO_API_DATASET_PARQUET,
    AESO_API_PREPROCESSING_DIR,
    AESO_API_RAW_DIR,
    OUTAGES_PARQUET,
    PA_TABLE_PARQUET,
)
from pipeline_shared import build_manifest, outputs_are_current, write_manifests
from preprocessing.shared import add_check, audit_passes, preprocessing_code_paths


OUTPUTS = {
    "aeso_api_dataset": AESO_API_DATASET_PARQUET,
    "actual_forecast": AESO_API_PREPROCESSING_DIR
    / "aeso_actual_forecast_hourly.parquet",
    "generation_capacity_by_fuel": AESO_API_PREPROCESSING_DIR
    / "aeso_generation_capacity_by_fuel_hourly.parquet",
    "generation_capacity_system": AESO_API_PREPROCESSING_DIR
    / "aeso_generation_capacity_system_hourly.parquet",
    "load_outage_forecast": AESO_API_PREPROCESSING_DIR
    / "aeso_load_outage_forecast_hourly.parquet",
    "intertie_outage_events": AESO_API_PREPROCESSING_DIR
    / "aeso_intertie_outage_events.parquet",
    "intertie_outage_hourly": AESO_API_PREPROCESSING_DIR
    / "aeso_intertie_outage_hourly.parquet",
    "pool_price_forecast": AESO_API_PREPROCESSING_DIR
    / "aeso_pool_price_forecast_hourly.parquet",
    "smp_intervals": AESO_API_PREPROCESSING_DIR
    / "aeso_smp_intervals.parquet",
    "smp_hourly": AESO_API_PREPROCESSING_DIR / "aeso_smp_hourly.parquet",
    "asset_list": AESO_API_PREPROCESSING_DIR
    / "aeso_asset_list_snapshot.parquet",
    "energy_merit_order_hourly": AESO_API_PREPROCESSING_DIR
    / "aeso_energy_merit_order_hourly.parquet",
    "operating_reserve_hourly": AESO_API_PREPROCESSING_DIR
    / "aeso_operating_reserve_hourly.parquet",
    "unit_commitment_events": AESO_API_PREPROCESSING_DIR
    / "aeso_unit_commitment_events.parquet",
    "unit_commitment_hourly": AESO_API_PREPROCESSING_DIR
    / "aeso_unit_commitment_hourly.parquet",
    "pool_participants": AESO_API_PREPROCESSING_DIR
    / "aeso_pool_participants_snapshot.parquet",
    "pool_participant_agents": AESO_API_PREPROCESSING_DIR
    / "aeso_pool_participant_agents_snapshot.parquet",
}

AUDIT_CHECKS = AESO_API_AUDITS_DIR / "aeso_api_audit_checks.csv"
DATASET_SUMMARY = AESO_API_AUDITS_DIR / "aeso_api_dataset_summary.csv"
OVERLAP_AUDIT = AESO_API_AUDITS_DIR / "aeso_api_overlap_audit.csv"
FIELD_REGISTER = AESO_API_AUDITS_DIR / "aeso_api_field_register.csv"

DATASET_DIRECTORIES = {
    "actual_forecast": "actual_forecast",
    "generation_capacity": "generation_capacity",
    "load_outage_forecast": "load_outage_forecast",
    "intertie_outages": "intertie_outages",
    "pool_price": "pool_price",
    "system_marginal_price": "system_marginal_price",
    "asset_list": "asset_list",
    "energy_merit_order": "energy_merit_order",
    "operating_reserve_offer_control": "operating_reserve_offer_control",
    "unit_commitment": "unit_commitment",
    "pool_participants": "pool_participants",
}


def _raw_json_files(directory_name: str) -> list[Path]:
    directory = AESO_API_RAW_DIR / directory_name
    files = sorted(
        path
        for path in directory.glob("*.json")
        if not path.name.endswith(".metadata.json")
    )
    if not files:
        raise FileNotFoundError(f"No AESO API JSON files found in {directory}")
    return files


def _all_source_files() -> list[Path]:
    files: list[Path] = []
    for directory_name in DATASET_DIRECTORIES.values():
        directory = AESO_API_RAW_DIR / directory_name
        files.extend(sorted(directory.glob("*.json")))
    return files


def _load_return(path: Path) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8"))
    response_code = str(payload.get("responseCode", ""))
    if response_code != "200":
        raise ValueError(
            f"AESO response in {path.name} has responseCode={response_code!r}"
        )
    if "return" not in payload:
        raise ValueError(f"AESO response in {path.name} has no 'return' field")
    return payload["return"]


def _records(directory_name: str, report_key: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in _raw_json_files(directory_name):
        returned = _load_return(path)
        records.extend(returned.get(report_key, []))
    return pd.DataFrame(records)


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True)


def _numeric(values: pd.Series) -> pd.Series:
    cleaned = values.replace("", np.nan)
    if pd.api.types.is_object_dtype(cleaned) or isinstance(
        cleaned.dtype, pd.StringDtype
    ):
        cleaned = cleaned.astype("string").str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _number(value: Any) -> float:
    """Parse one API numeric value without converting a full Series."""

    if value in {None, ""}:
        return float("nan")
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return float("nan")


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return float("nan")
    selected_values = values[valid]
    selected_weights = weights[valid]
    order = np.argsort(selected_values, kind="stable")
    selected_values = selected_values[order]
    selected_weights = selected_weights[order]
    cutoff = quantile * selected_weights.sum()
    position = int(np.searchsorted(np.cumsum(selected_weights), cutoff, side="left"))
    return float(selected_values[min(position, len(selected_values) - 1)])


def _volume_hhi(labels: pd.Series, volumes: pd.Series) -> float:
    valid = labels.notna() & volumes.notna() & volumes.gt(0)
    if not valid.any():
        return float("nan")
    grouped = volumes.loc[valid].groupby(labels.loc[valid]).sum()
    shares = grouped / grouped.sum()
    return float((shares**2).sum())


def _snake_label(value: Any) -> str | None:
    if value is None:
        return None
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _deduplicate(frame: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame:
    frame = frame.drop_duplicates().copy()
    conflicts = frame.duplicated(keys, keep=False)
    if conflicts.any():
        examples = frame.loc[conflicts, keys].drop_duplicates().head().to_dict("records")
        raise ValueError(f"{name} has conflicting duplicate keys: {examples}")
    return frame.sort_values(keys).reset_index(drop=True)


def normalize_actual_forecast() -> pd.DataFrame:
    raw = _records("actual_forecast", "Actual Forecast Report")
    out = pd.DataFrame(
        {
            "timestamp_utc": _utc(raw["begin_datetime_utc"]),
            "actual_ail_api_mw": _numeric(raw["alberta_internal_load"]),
            "forecast_ail_mw": _numeric(raw["forecast_alberta_internal_load"]),
        }
    ).dropna(subset=["timestamp_utc"])
    out["load_forecast_error_mw"] = (
        out["actual_ail_api_mw"] - out["forecast_ail_mw"]
    )
    return _deduplicate(out, ["timestamp_utc"], "actual forecast")


def normalize_generation_capacity() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for path in _raw_json_files("generation_capacity"):
        for group in _load_return(path):
            for hour in group.get("Hours", []):
                values = hour.get("outage_grouping", {})
                rows.append(
                    {
                        "timestamp_utc": hour.get("begin_datetime_utc"),
                        "fuel_type": _snake_label(group.get("fuel_type")),
                        "sub_fuel_type": _snake_label(
                            group.get("sub_fuel_type")
                        ),
                        "maximum_capability_mw": values.get("MC"),
                        "mothball_outage_mw": values.get("MBO OUT"),
                        "operating_outage_mw": values.get("OP OUT"),
                        "available_capability_mw": values.get("AC"),
                    }
                )
    long = pd.DataFrame(rows)
    long["timestamp_utc"] = _utc(long["timestamp_utc"])
    for column in [
        "maximum_capability_mw",
        "mothball_outage_mw",
        "operating_outage_mw",
        "available_capability_mw",
    ]:
        long[column] = _numeric(long[column])
    long["total_unavailable_mw"] = (
        long["maximum_capability_mw"] - long["available_capability_mw"]
    )
    long = _deduplicate(
        long.dropna(subset=["timestamp_utc", "fuel_type", "sub_fuel_type"]),
        ["timestamp_utc", "fuel_type", "sub_fuel_type"],
        "generation capacity",
    )
    measures = [
        "maximum_capability_mw",
        "mothball_outage_mw",
        "operating_outage_mw",
        "available_capability_mw",
        "total_unavailable_mw",
    ]
    system = (
        long.groupby("timestamp_utc", as_index=False)[measures]
        .sum(min_count=1)
        .rename(columns={column: f"system_{column}" for column in measures})
    )
    return long, system


def normalize_load_outage_forecast() -> pd.DataFrame:
    raw = _records("load_outage_forecast", "loadOutagePerHourVO")
    value_column = next(
        column for column in raw if column.startswith("load_outage_forecast")
    )
    out = pd.DataFrame(
        {
            "timestamp_utc": _utc(raw["begin_datetime_utc"]),
            "load_outage_forecast_mw": _numeric(raw[value_column]),
        }
    ).dropna(subset=["timestamp_utc"])
    return _deduplicate(out, ["timestamp_utc"], "load outage forecast")


def _localize_alberta(values: pd.Series) -> pd.Series:
    naive = pd.to_datetime(values, errors="coerce")
    return naive.dt.tz_localize(
        "America/Edmonton",
        ambiguous="NaT",
        nonexistent="shift_forward",
    )


def normalize_intertie_outages() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for path in _raw_json_files("intertie_outages"):
        returned = _load_return(path)
        events = returned.get("Outages", {}).get("Outage", [])
        for event in events:
            affected = event.get("affectedIntertieOrFlowgate") or [{}]
            for item in affected:
                if isinstance(item, str):
                    item = {"displayValue": item, "affectedLines": [item]}
                lines = item.get("affectedLines") or [None]
                for line in lines:
                    rows.append(
                        {
                            "element": event.get("element"),
                            "affected_intertie_or_flowgate": item.get(
                                "displayValue"
                            ),
                            "affected_line": line,
                            "start_time_alberta_raw": event.get(
                                "fromInLocalTime"
                            ),
                            "end_time_alberta_raw": event.get("toInLocalTime"),
                        }
                    )
    events = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    events["start_time_alberta"] = _localize_alberta(
        events["start_time_alberta_raw"]
    )
    events["end_time_alberta"] = _localize_alberta(events["end_time_alberta_raw"])
    events["start_time_utc"] = events["start_time_alberta"].dt.tz_convert("UTC")
    events["end_time_utc"] = events["end_time_alberta"].dt.tz_convert("UTC")
    natural_event_key = pd.MultiIndex.from_frame(
        events[["element", "start_time_alberta_raw", "end_time_alberta_raw"]]
    )
    event_codes, _ = pd.factorize(natural_event_key, sort=True)
    events.insert(0, "event_id", event_codes + 1)

    expanded: list[pd.DataFrame] = []
    valid = events.dropna(subset=["start_time_utc", "end_time_utc"])
    valid = valid.loc[valid["end_time_utc"] > valid["start_time_utc"]]
    for row in valid.itertuples(index=False):
        hours = pd.date_range(
            row.start_time_utc.floor("h"),
            (row.end_time_utc - pd.Timedelta("1ns")).floor("h"),
            freq="h",
        )
        expanded.append(
            pd.DataFrame(
                {
                    "timestamp_utc": hours,
                    "event_id": row.event_id,
                    "affected_intertie_or_flowgate": (
                        row.affected_intertie_or_flowgate
                    ),
                }
            )
        )
    if expanded:
        active = pd.concat(expanded, ignore_index=True)
        hourly = (
            active.groupby("timestamp_utc")
            .agg(
                active_intertie_outage_records=("event_id", "nunique"),
                affected_interties=(
                    "affected_intertie_or_flowgate",
                    lambda values: ";".join(sorted(set(values.dropna()))),
                ),
            )
            .reset_index()
        )
        affected_sets = hourly["affected_interties"].str.split(";").apply(set)

        hourly["bc_outage_active"] = (
            affected_sets
            .apply(lambda values: bool(values & {"BC", "BC/MATL"}))
            .astype("int8")
        )

        hourly["matl_outage_active"] = (
            affected_sets
            .apply(lambda values: bool(values & {"MATL", "BC/MATL"}))
            .astype("int8")
        )

        hourly["sk_outage_active"] = (
            affected_sets
            .apply(lambda values: any(value.startswith("SK") for value in values))
            .astype("int8")
        )

        hourly["bc_matl_outage_active"] = (
            affected_sets
            .apply(lambda values: "BC/MATL" in values)
            .astype("int8")
        )
    else:
        hourly = pd.DataFrame(
            columns=["timestamp_utc", "active_intertie_outage_records"]
        )
    return events, hourly


def normalize_pool_price() -> pd.DataFrame:
    raw = _records("pool_price", "Pool Price Report")
    out = pd.DataFrame(
        {
            "timestamp_utc": _utc(raw["begin_datetime_utc"]),
            "pool_price_api_cad_mwh": _numeric(raw["pool_price"]),
            "forecast_pool_price_cad_mwh": _numeric(raw["forecast_pool_price"]),
            "rolling_30day_average_cad_mwh": _numeric(raw["rolling_30day_avg"]),
        }
    ).dropna(subset=["timestamp_utc"])
    return _deduplicate(out, ["timestamp_utc"], "pool price")


def normalize_smp() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = _records("system_marginal_price", "System Marginal Price Report")
    intervals = pd.DataFrame(
        {
            "begin_timestamp_utc": _utc(raw["begin_datetime_utc"]),
            "end_timestamp_utc": _utc(raw["end_datetime_utc"]),
            "system_marginal_price_cad_mwh": _numeric(
                raw["system_marginal_price"]
            ),
            "available_block_volume_mw": _numeric(raw["volume"]),
        }
    ).dropna(subset=["begin_timestamp_utc", "end_timestamp_utc"])
    intervals["duration_minutes"] = (
        intervals["end_timestamp_utc"] - intervals["begin_timestamp_utc"]
    ).dt.total_seconds() / 60
    intervals["timestamp_utc"] = intervals["begin_timestamp_utc"].dt.floor("h")
    intervals = _deduplicate(
        intervals,
        ["begin_timestamp_utc", "end_timestamp_utc"],
        "system marginal price",
    )
    intervals["weighted_price"] = (
        intervals["system_marginal_price_cad_mwh"]
        * intervals["duration_minutes"]
    )
    hourly = (
        intervals.groupby("timestamp_utc", as_index=False)
        .agg(
            smp_weighted_price_sum=("weighted_price", "sum"),
            smp_observed_minutes=("duration_minutes", "sum"),
            smp_minimum_cad_mwh=("system_marginal_price_cad_mwh", "min"),
            smp_maximum_cad_mwh=("system_marginal_price_cad_mwh", "max"),
            smp_last_cad_mwh=("system_marginal_price_cad_mwh", "last"),
            smp_interval_count=("system_marginal_price_cad_mwh", "size"),
        )
    )
    hourly["smp_time_weighted_mean_cad_mwh"] = (
        hourly.pop("smp_weighted_price_sum")
        / hourly["smp_observed_minutes"].replace(0, np.nan)
    )
    return intervals.drop(columns="weighted_price"), hourly


def normalize_asset_list() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in _raw_json_files("asset_list"):
        match = re.search(r"snapshot_(\d{8}T\d{6}Z)", path.name)
        snapshot = pd.to_datetime(
            match.group(1) if match else path.stat().st_mtime,
            utc=True,
        )
        for record in _load_return(path):
            rows.append(
                {
                    "snapshot_timestamp_utc": snapshot,
                    "asset_name": record.get("asset_name"),
                    "asset_id": record.get("asset_ID"),
                    "asset_type": record.get("asset_type"),
                    "operating_status": record.get("operating_status"),
                    "pool_participant_name": record.get("pool_participant_name"),
                    "pool_participant_id": record.get("pool_participant_ID"),
                    "net_to_grid_asset_flag": record.get(
                        "net_to_grid_asset_flag"
                    ),
                    "asset_includes_storage_flag": record.get(
                        "asset_incl_storage_flag"
                    ),
                }
            )
    assets = pd.DataFrame(rows)
    return _deduplicate(
        assets,
        ["snapshot_timestamp_utc", "asset_id", "pool_participant_id"],
        "asset list",
    )


def normalize_pool_participants() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize current participant metadata without propagating contacts."""

    participant_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []
    for path in _raw_json_files("pool_participants"):
        match = re.search(r"snapshot_(\d{8}T\d{6}Z)", path.name)
        snapshot = pd.to_datetime(
            match.group(1) if match else path.stat().st_mtime,
            utc=True,
        )
        for record in _load_return(path):
            participant_id = record.get("pool_participant_ID")
            participant_rows.append(
                {
                    "snapshot_timestamp_utc": snapshot,
                    "pool_participant_id": participant_id,
                    "pool_participant_name": record.get("pool_participant_name"),
                }
            )
            for agent in record.get("agent_list") or []:
                agent_rows.append(
                    {
                        "snapshot_timestamp_utc": snapshot,
                        "pool_participant_id": participant_id,
                        "agent_id": agent.get("agent_ID"),
                        "agent_name": agent.get("agent_name"),
                    }
                )
    participants = _deduplicate(
        pd.DataFrame(participant_rows),
        ["snapshot_timestamp_utc", "pool_participant_id"],
        "pool participants",
    )
    agents = pd.DataFrame(agent_rows)
    if not agents.empty:
        agents = _deduplicate(
            agents,
            ["snapshot_timestamp_utc", "pool_participant_id", "agent_id"],
            "pool participant agents",
        )
    return participants, agents


def normalize_energy_merit_order() -> pd.DataFrame:
    """Reduce daily nested offer stacks to one market-structure row per hour."""

    rows: list[dict[str, Any]] = []
    thresholds = [0, 50, 100, 250, 500]
    for path in _raw_json_files("energy_merit_order"):
        returned = _load_return(path)
        for snapshot in returned.get("data", []):
            blocks = pd.DataFrame(snapshot.get("energy_blocks") or [])
            if blocks.empty:
                continue
            for column in [
                "block_price",
                "block_size",
                "available_MW",
                "dispatched_MW",
            ]:
                blocks[column] = _numeric(blocks[column])
            available = blocks["available_MW"].clip(lower=0).fillna(0)
            dispatched = blocks["dispatched_MW"].clip(lower=0).fillna(0)
            prices = blocks["block_price"].to_numpy(dtype=float)
            available_values = available.to_numpy(dtype=float)
            dispatched_values = dispatched.to_numpy(dtype=float)
            import_flag = blocks["import_or_export"].astype("string").eq("I")
            flexible_flag = blocks["flexible?"].astype("string").eq("Y")
            dispatched_flag = dispatched.gt(0) | blocks["dispatched?"].astype(
                "string"
            ).eq("Y")
            row: dict[str, Any] = {
                "timestamp_utc": pd.to_datetime(
                    snapshot.get("begin_dateTime_utc"), utc=True, errors="coerce"
                ),
                "aeso_merit_block_count": len(blocks),
                "aeso_merit_asset_count": blocks["asset_ID"].nunique(dropna=True),
                "aeso_merit_offer_controller_count": blocks[
                    "offer_control"
                ].nunique(dropna=True),
                "aeso_merit_total_block_mw": blocks["block_size"].sum(min_count=1),
                "aeso_merit_available_mw": available.sum(),
                "aeso_merit_dispatched_mw": dispatched.sum(),
                "aeso_merit_import_available_mw": available.loc[import_flag].sum(),
                "aeso_merit_flexible_available_mw": available.loc[
                    flexible_flag
                ].sum(),
                "aeso_merit_available_price_weighted_mean": _weighted_mean(
                    prices, available_values
                ),
                "aeso_merit_available_price_p50": _weighted_quantile(
                    prices, available_values, 0.50
                ),
                "aeso_merit_available_price_p90": _weighted_quantile(
                    prices, available_values, 0.90
                ),
                "aeso_merit_dispatched_price_weighted_mean": _weighted_mean(
                    prices, dispatched_values
                ),
                "aeso_merit_highest_dispatched_offer": blocks.loc[
                    dispatched_flag, "block_price"
                ].max(),
                "aeso_merit_offer_control_hhi": _volume_hhi(
                    blocks["offer_control"], available
                ),
            }
            for threshold in thresholds:
                row[f"aeso_merit_available_mw_at_or_below_{threshold}"] = (
                    available.loc[blocks["block_price"].le(threshold)].sum()
                )
            rows.append(row)
    result = pd.DataFrame(rows).dropna(subset=["timestamp_utc"])
    return _deduplicate(result, ["timestamp_utc"], "energy merit order hourly")


def normalize_operating_reserve() -> pd.DataFrame:
    """Reduce nested reserve offers to one summary row per settlement hour."""

    rows: list[dict[str, Any]] = []
    products = ["RR", "SR", "SUPG", "SUPL"]
    for path in _raw_json_files("operating_reserve_offer_control"):
        returned = _load_return(path)
        snapshots = returned.get("Operating Reserve Trade Merit Order", [])
        for snapshot in snapshots:
            blocks = pd.DataFrame(snapshot.get("operating_reserve_blocks") or [])
            if blocks.empty:
                continue
            for column in [
                "volume",
                "active_price",
                "premium_price",
                "activation_price",
            ]:
                blocks[column] = _numeric(blocks[column])
            volumes = blocks["volume"].clip(lower=0).fillna(0)
            active = blocks["commodity"].astype("string").str.upper().eq("ACTIVE")
            standby = blocks["commodity"].astype("string").str.upper().eq(
                "STANDBY"
            )
            row: dict[str, Any] = {
                "timestamp_utc": pd.to_datetime(
                    snapshot.get("begin_datetime_utc"), utc=True, errors="coerce"
                ),
                "aeso_or_block_count": len(blocks),
                "aeso_or_asset_count": blocks["asset_ID"].nunique(dropna=True),
                "aeso_or_offer_controller_count": blocks[
                    "offer_control"
                ].nunique(dropna=True),
                "aeso_or_active_volume_mw": volumes.loc[active].sum(),
                "aeso_or_standby_volume_mw": volumes.loc[standby].sum(),
                "aeso_or_active_price_weighted_mean": _weighted_mean(
                    blocks.loc[active, "active_price"].to_numpy(dtype=float),
                    volumes.loc[active].to_numpy(dtype=float),
                ),
                "aeso_or_standby_premium_weighted_mean": _weighted_mean(
                    blocks.loc[standby, "premium_price"].to_numpy(dtype=float),
                    volumes.loc[standby].to_numpy(dtype=float),
                ),
                "aeso_or_standby_activation_price_weighted_mean": _weighted_mean(
                    blocks.loc[standby, "activation_price"].to_numpy(dtype=float),
                    volumes.loc[standby].to_numpy(dtype=float),
                ),
            }
            normalized_products = blocks["product"].astype("string").str.upper()
            for product in products:
                mask = normalized_products.eq(product)
                row[f"aeso_or_{product.lower()}_active_volume_mw"] = volumes.loc[
                    mask & active
                ].sum()
                row[f"aeso_or_{product.lower()}_standby_volume_mw"] = volumes.loc[
                    mask & standby
                ].sum()
            rows.append(row)
    result = pd.DataFrame(rows).dropna(subset=["timestamp_utc"])
    return _deduplicate(result, ["timestamp_utc"], "operating reserve hourly")


def normalize_unit_commitment() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize directives and derive issued-hour and operating-hour counts."""

    records: list[dict[str, Any]] = []
    for path in _raw_json_files("unit_commitment"):
        returned = _load_return(path)
        records.extend(returned.get("unit_commitment", []))
    events = pd.DataFrame(records).rename(
        columns={
            "asset_ID": "asset_id",
            "issued_time_utc": "issued_timestamp_utc",
            "begin_time_utc": "directive_begin_timestamp_utc",
            "operation_start_time_utc": "operation_start_timestamp_utc",
            "operation_end_time_utc": "operation_end_timestamp_utc",
        }
    )
    utc_columns = [
        "issued_timestamp_utc",
        "directive_begin_timestamp_utc",
        "operation_start_timestamp_utc",
        "operation_end_timestamp_utc",
    ]
    for column in utc_columns:
        events[column] = _utc(events[column])
    events = events[["asset_id", *utc_columns]].dropna(
        subset=["asset_id", "issued_timestamp_utc"]
    )
    events = _deduplicate(
        events,
        ["asset_id", "issued_timestamp_utc", "operation_start_timestamp_utc"],
        "unit commitment",
    )
    events["issue_to_operation_lead_hours"] = (
        events["operation_start_timestamp_utc"] - events["issued_timestamp_utc"]
    ).dt.total_seconds() / 3600

    issued = events.assign(
        timestamp_utc=events["issued_timestamp_utc"].dt.floor("h")
    ).groupby("timestamp_utc", as_index=False).agg(
        aeso_unit_commitment_directives_issued=("asset_id", "size"),
        aeso_unit_commitment_assets_issued=("asset_id", "nunique"),
        aeso_unit_commitment_median_lead_hours=(
            "issue_to_operation_lead_hours",
            "median",
        ),
    )

    active_rows: list[dict[str, Any]] = []
    valid = events.dropna(
        subset=["operation_start_timestamp_utc", "operation_end_timestamp_utc"]
    )
    valid = valid.loc[
        valid["operation_end_timestamp_utc"]
        > valid["operation_start_timestamp_utc"]
    ]
    for event in valid.itertuples(index=False):
        for timestamp in pd.date_range(
            event.operation_start_timestamp_utc.floor("h"),
            (event.operation_end_timestamp_utc - pd.Timedelta("1ns")).floor("h"),
            freq="h",
        ):
            active_rows.append({"timestamp_utc": timestamp, "asset_id": event.asset_id})
    if active_rows:
        active = pd.DataFrame(active_rows).groupby(
            "timestamp_utc", as_index=False
        ).agg(
            aeso_unit_commitment_active_directives=("asset_id", "size"),
            aeso_unit_commitment_active_assets=("asset_id", "nunique"),
        )
        hourly = issued.merge(active, on="timestamp_utc", how="outer")
    else:
        hourly = issued
    return events, hourly.sort_values("timestamp_utc").reset_index(drop=True)


def build_generation_capacity_wide(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot fuel-level generation capacity into hourly master-safe columns."""

    source = long.copy()
    source["technology"] = source["sub_fuel_type"].fillna(source["fuel_type"])
    measures = {
        "maximum_capability_mw": "maximum_capability_mw",
        "mothball_outage_mw": "mothball_outage_mw",
        "operating_outage_mw": "operating_outage_mw",
        "available_capability_mw": "available_capability_mw",
        "total_unavailable_mw": "total_unavailable_mw",
    }
    pieces: list[pd.DataFrame] = []
    for measure, suffix in measures.items():
        pivot = source.pivot(
            index="timestamp_utc",
            columns="technology",
            values=measure,
        )
        pivot.columns = [
            f"aeso_gen_{_snake_label(technology)}_{suffix}"
            for technology in pivot.columns
        ]
        pieces.append(pivot)
    return pd.concat(pieces, axis=1).sort_index().reset_index()


def _merge_hourly(
    base: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    name: str,
) -> pd.DataFrame:
    """Left-merge one unique hourly table without permitting name collisions."""

    if incoming["timestamp_utc"].duplicated().any():
        raise ValueError(f"{name} contains duplicate hourly timestamps")
    overlaps = (set(base.columns) & set(incoming.columns)) - {"timestamp_utc"}
    if overlaps:
        raise ValueError(f"{name} duplicates hourly columns: {sorted(overlaps)}")
    return base.merge(incoming, on="timestamp_utc", how="left", validate="one_to_one")


def build_aeso_api_dataset(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble the canonical hourly AESO API table for master integration."""

    actual = datasets["actual_forecast"]
    hourly = actual[
        ["timestamp_utc", "forecast_ail_mw", "load_forecast_error_mw"]
    ].rename(
        columns={
            "forecast_ail_mw": "aeso_forecast_ail_mw",
            "load_forecast_error_mw": "aeso_load_forecast_error_mw",
        }
    )

    generation_system = datasets["generation_capacity_system"].rename(
        columns={
            column: f"aeso_gen_{column}"
            for column in datasets["generation_capacity_system"].columns
            if column != "timestamp_utc"
        }
    )
    generation_wide = build_generation_capacity_wide(
        datasets["generation_capacity_by_fuel"]
    )
    load_outage = datasets["load_outage_forecast"].rename(
        columns={"load_outage_forecast_mw": "aeso_load_outage_forecast_mw"}
    )
    pool_price = datasets["pool_price_forecast"][
        ["timestamp_utc", "forecast_pool_price_cad_mwh", "pool_price_api_cad_mwh"]
    ].copy()
    pool_price["aeso_pool_price_forecast_error_cad_mwh"] = (
        pool_price["pool_price_api_cad_mwh"]
        - pool_price["forecast_pool_price_cad_mwh"]
    )
    pool_price = pool_price.drop(columns="pool_price_api_cad_mwh").rename(
        columns={
            "forecast_pool_price_cad_mwh": "aeso_forecast_pool_price_cad_mwh"
        }
    )
    smp = datasets["smp_hourly"].rename(
        columns={
            column: f"aeso_{column}"
            for column in datasets["smp_hourly"].columns
            if column != "timestamp_utc"
        }
    )
    intertie = datasets["intertie_outage_hourly"].drop(
        columns=["affected_interties"], errors="ignore"
    ).rename(
        columns={
            column: f"aeso_{column}"
            for column in datasets["intertie_outage_hourly"].columns
            if column not in {"timestamp_utc", "affected_interties"}
        }
    )
    unit_commitment = datasets["unit_commitment_hourly"]

    for name, frame in [
        ("generation system", generation_system),
        ("generation by technology", generation_wide),
        ("load outage forecast", load_outage),
        ("pool price forecast", pool_price),
        ("system marginal price", smp),
        ("energy merit order", datasets["energy_merit_order_hourly"]),
        ("operating reserve", datasets["operating_reserve_hourly"]),
    ]:
        hourly = _merge_hourly(hourly, frame, name=name)

    hourly = _merge_hourly(hourly, intertie, name="intertie outages")
    intertie_start = pd.Timestamp("2020-11-09 07:00:00", tz="UTC")
    intertie_coverage = hourly["timestamp_utc"].ge(intertie_start)
    intertie_columns = [
        column for column in intertie.columns if column != "timestamp_utc"
    ]
    hourly.loc[intertie_coverage, intertie_columns] = hourly.loc[
        intertie_coverage, intertie_columns
    ].fillna(0)

    intertie_flag_columns = [
        column
        for column in intertie_columns
        if column.endswith("_outage_active")
    ]

    hourly.loc[intertie_coverage, intertie_flag_columns] = (
        hourly.loc[intertie_coverage, intertie_flag_columns]
        .fillna(0)
        .astype("int8")
    )

    hourly["aeso_intertie_outage_archive_available"] = intertie_coverage.astype(
        "int8"
    )

    hourly = _merge_hourly(hourly, unit_commitment, name="unit commitment")
    commitment_start = pd.Timestamp("2024-07-01 00:00:00", tz="UTC")
    commitment_coverage = hourly["timestamp_utc"].ge(commitment_start)
    commitment_count_columns = [
        column
        for column in unit_commitment.columns
        if column != "timestamp_utc" and not column.endswith("lead_hours")
    ]
    hourly.loc[commitment_coverage, commitment_count_columns] = hourly.loc[
        commitment_coverage, commitment_count_columns
    ].fillna(0)
    hourly["aeso_unit_commitment_archive_available"] = commitment_coverage.astype(
        "int8"
    )

    source_indicators = {
        "aeso_actual_forecast_archive_available": "aeso_forecast_ail_mw",
        "aeso_generation_capacity_archive_available": (
            "aeso_gen_system_available_capability_mw"
        ),
        "aeso_load_outage_archive_available": "aeso_load_outage_forecast_mw",
        "aeso_pool_price_archive_available": "aeso_forecast_pool_price_cad_mwh",
        "aeso_smp_archive_available": "aeso_smp_observed_minutes",
        "aeso_merit_order_archive_available": "aeso_merit_block_count",
        "aeso_operating_reserve_archive_available": "aeso_or_block_count",
    }
    for indicator, source_column in source_indicators.items():
        hourly[indicator] = hourly[source_column].notna().astype("int8")

    return hourly.sort_values("timestamp_utc").reset_index(drop=True)


def _hourly_audit(
    rows: list[dict[str, Any]],
    name: str,
    frame: pd.DataFrame,
    *,
    coverage_severity: str = "error",
) -> None:
    timestamps = pd.DatetimeIndex(frame["timestamp_utc"])
    expected = pd.date_range(timestamps.min(), timestamps.max(), freq="h", tz="UTC")
    missing_timestamps = expected.difference(timestamps)
    add_check(rows, f"{name}__rows_positive", len(frame) > 0, len(frame), "> 0")
    add_check(
        rows,
        f"{name}__duplicate_timestamps",
        not timestamps.has_duplicates,
        int(timestamps.duplicated().sum()),
        0,
    )
    add_check(
        rows,
        f"{name}__continuous_hourly_coverage",
        len(missing_timestamps) == 0,
        len(missing_timestamps),
        0,
        severity=coverage_severity,
        notes=(
            "; ".join(missing_timestamps.astype(str).tolist()[:10])
            if len(missing_timestamps)
            else ""
        ),
    )


def build_audits(datasets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for name in [
        "actual_forecast",
        "generation_capacity_system",
        "load_outage_forecast",
        "pool_price_forecast",
        "smp_hourly",
        "energy_merit_order_hourly",
        "operating_reserve_hourly",
        "aeso_api_dataset",
    ]:
        _hourly_audit(
            rows,
            name,
            datasets[name],
            coverage_severity=(
                "warning" if name == "energy_merit_order_hourly" else "error"
            ),
        )

    reference_timestamps = pd.Index(datasets["actual_forecast"]["timestamp_utc"])
    for name in [
        "generation_capacity_system",
        "load_outage_forecast",
        "pool_price_forecast",
        "smp_hourly",
        "energy_merit_order_hourly",
        "operating_reserve_hourly",
    ]:
        observed = pd.Index(datasets[name]["timestamp_utc"])
        add_check(
            rows,
            f"{name}__matches_actual_forecast_timeline",
            observed.equals(reference_timestamps),
            len(reference_timestamps.symmetric_difference(observed)),
            0,
            severity=(
                "warning" if name == "energy_merit_order_hourly" else "error"
            ),
            notes=(
                "Missing merit-order hours remain null in the integrated "
                "dataset and are identified by the archive-availability flag."
                if name == "energy_merit_order_hourly"
                else ""
            ),
        )

    for dataset_name, columns in {
        "actual_forecast": ["actual_ail_api_mw", "forecast_ail_mw"],
        "load_outage_forecast": ["load_outage_forecast_mw"],
        "pool_price_forecast": ["pool_price_api_cad_mwh"],
    }.items():
        for column in columns:
            missing = int(datasets[dataset_name][column].isna().sum())
            add_check(
                rows,
                f"{dataset_name}__missing__{column}",
                missing == 0,
                missing,
                0,
            )

    forecast_price_missing = int(
        datasets["pool_price_forecast"][
            "forecast_pool_price_cad_mwh"
        ].isna().sum()
    )
    add_check(
        rows,
        "pool_price_forecast__missing__forecast_pool_price_cad_mwh",
        forecast_price_missing == 0,
        forecast_price_missing,
        0,
        severity="warning",
    )

    generation = datasets["generation_capacity_by_fuel"]
    identity_difference = (
        generation["total_unavailable_mw"]
        - generation["mothball_outage_mw"]
        - generation["operating_outage_mw"]
    ).abs()
    add_check(
        rows,
        "generation_capacity__unavailability_identity",
        bool(identity_difference.le(1e-9).all()),
        int(identity_difference.gt(1e-9).sum()),
        0,
    )
    incomplete_smp_hours = int(
        (~datasets["smp_hourly"]["smp_observed_minutes"].eq(60)).sum()
    )
    add_check(
        rows,
        "smp__hours_without_60_observed_minutes",
        incomplete_smp_hours == 0,
        incomplete_smp_hours,
        0,
    )

    intervals = datasets["smp_intervals"]
    valid_duration = intervals["duration_minutes"].gt(0) & intervals[
        "duration_minutes"
    ].le(60)
    add_check(
        rows,
        "smp__valid_interval_duration",
        bool(valid_duration.all()),
        int((~valid_duration).sum()),
        0,
    )
    ambiguous_events = datasets["intertie_outage_events"][
        ["start_time_utc", "end_time_utc"]
    ].isna().any(axis=1).sum()
    add_check(
        rows,
        "intertie_outages__unresolved_local_timestamps",
        ambiguous_events == 0,
        int(ambiguous_events),
        0,
        severity="warning",
        notes="Raw local timestamp strings are retained for unresolved DST cases.",
    )

    summary_rows = []
    for name, frame in datasets.items():
        timestamp_column = next(
            (
                column
                for column in [
                    "timestamp_utc",
                    "begin_timestamp_utc",
                    "start_time_utc",
                    "snapshot_timestamp_utc",
                ]
                if column in frame
            ),
            None,
        )
        summary_rows.append(
            {
                "dataset": name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "first_timestamp": (
                    frame[timestamp_column].min() if timestamp_column else pd.NaT
                ),
                "last_timestamp": (
                    frame[timestamp_column].max() if timestamp_column else pd.NaT
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def build_overlap_audit(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if PA_TABLE_PARQUET.exists():
        existing = pd.read_parquet(PA_TABLE_PARQUET)
        existing["timestamp_utc"] = _utc(existing["timestamp_utc"])
        comparisons = [
            (
                "actual_ail",
                datasets["actual_forecast"],
                "actual_ail_api_mw",
                "ail_mw",
            ),
            (
                "pool_price",
                datasets["pool_price_forecast"],
                "pool_price_api_cad_mwh",
                "pool_price_cad_mwh",
            ),
        ]
        for name, api, api_column, existing_column in comparisons:
            joined = api[["timestamp_utc", api_column]].merge(
                existing[["timestamp_utc", existing_column]],
                on="timestamp_utc",
                how="inner",
            ).dropna(subset=[api_column, existing_column])
            difference = (joined[api_column] - joined[existing_column]).abs()
            rows.append(
                {
                    "comparison": name,
                    "overlapping_rows": len(joined),
                    "exact_match_rows": int(difference.eq(0).sum()),
                    "exact_match_pct": difference.eq(0).mean() * 100,
                    "mean_absolute_difference": difference.mean(),
                    "maximum_absolute_difference": difference.max(),
                    "interpretation": "duplicate/audit field; do not overwrite automatically",
                }
            )

    if OUTAGES_PARQUET.exists():
        existing = pd.read_parquet(OUTAGES_PARQUET)
        existing["timestamp_utc"] = _utc(existing["timestamp_utc"])
        api = datasets["generation_capacity_by_fuel"].copy()
        fuel_map = {
            "coal": "coal_outage",
            "cogeneration": "cogeneration_outage",
            "combined_cycle": "combined_cycle_outage",
            "dual_fuel": "dual_fuel_outage",
            "gas_fired_steam": "gas_fired_steam_outage",
            "hydro": "hydro_outage",
            "other": "other_outage",
            "simple_cycle": "simple_cycle_outage",
            "solar": "solar_outage",
            "energy_storage": "storage_outage",
            "wind": "wind_outage",
        }
        api["existing_column"] = api["sub_fuel_type"].map(fuel_map)
        for existing_column, group in api.dropna(
            subset=["existing_column"]
        ).groupby("existing_column"):
            if existing_column not in existing:
                continue
            joined = group[["timestamp_utc", "operating_outage_mw"]].merge(
                existing[["timestamp_utc", existing_column]],
                on="timestamp_utc",
                how="inner",
            ).dropna(subset=["operating_outage_mw", existing_column])
            difference = (
                joined["operating_outage_mw"] - joined[existing_column]
            ).abs()
            rows.append(
                {
                    "comparison": f"operating_outage__{existing_column}",
                    "overlapping_rows": len(joined),
                    "exact_match_rows": int(difference.eq(0).sum()),
                    "exact_match_pct": difference.eq(0).mean() * 100,
                    "mean_absolute_difference": difference.mean(),
                    "maximum_absolute_difference": difference.max(),
                    "interpretation": "related definitions; retain separately",
                }
            )
    return pd.DataFrame(rows)


def build_field_register() -> pd.DataFrame:
    """Document overlap, timing status, and current integration decisions."""

    return pd.DataFrame(
        [
            {
                "field_family": "Actual Alberta internal load",
                "source": "Actual Forecast Report",
                "new_information": False,
                "timing_status": "realized outcome",
                "integration_decision": "retain for overlap audit only",
            },
            {
                "field_family": "Forecast Alberta internal load",
                "source": "Actual Forecast Report",
                "new_information": True,
                "timing_status": "issue/revision timing requires audit",
                "integration_decision": "candidate forecast feature",
            },
            {
                "field_family": "MC, AC, OP outage, and mothball outage",
                "source": "AIES Generation Capacity",
                "new_information": True,
                "timing_status": "historical snapshot timing not exposed",
                "integration_decision": "hourly prefixed fields in master; timing-dependent",
            },
            {
                "field_family": "Load-outage forecast",
                "source": "Load Outage Forecast",
                "new_information": True,
                "timing_status": "issue/revision timing requires audit",
                "integration_decision": "candidate conditional feature",
            },
            {
                "field_family": "Intertie outage events",
                "source": "Intertie Public Reports",
                "new_information": True,
                "timing_status": "event publication timing not exposed",
                "integration_decision": "descriptive until timing is audited",
            },
            {
                "field_family": "Actual pool price",
                "source": "Pool Price Report",
                "new_information": False,
                "timing_status": "realized outcome",
                "integration_decision": "retain for overlap audit only",
            },
            {
                "field_family": "AESO forecast pool price",
                "source": "Pool Price Report",
                "new_information": True,
                "timing_status": "issue/revision timing requires audit",
                "integration_decision": "benchmark candidate after timing audit",
            },
            {
                "field_family": "Rolling 30-day pool-price average",
                "source": "Pool Price Report",
                "new_information": False,
                "timing_status": "derivable from prior prices",
                "integration_decision": "recompute locally if needed",
            },
            {
                "field_family": "System marginal price intervals",
                "source": "System Marginal Price Report",
                "new_information": True,
                "timing_status": "realized market outcome",
                "integration_decision": "hourly descriptive fields in master; lagged forecast use only",
            },
            {
                "field_family": "Asset metadata",
                "source": "Asset List",
                "new_information": True,
                "timing_status": "current snapshot, not historical",
                "integration_decision": "reference lookup only",
            },
            {
                "field_family": "Energy merit-order structure",
                "source": "Energy Merit Order Report",
                "new_information": True,
                "timing_status": "published with 60-day delay; ex-post",
                "integration_decision": "hourly descriptive fields in master",
            },
            {
                "field_family": "Operating-reserve offer structure",
                "source": "Operating Reserve Offer Control Report",
                "new_information": True,
                "timing_status": "published with 60-day delay; ex-post",
                "integration_decision": "hourly descriptive fields in master",
            },
            {
                "field_family": "Unit commitment directives",
                "source": "Unit Commitment Data API",
                "new_information": True,
                "timing_status": "explicit issued time; archive begins 2024-07",
                "integration_decision": "hourly counts in master with coverage flag",
            },
            {
                "field_family": "Pool participant metadata",
                "source": "Pool Participant API",
                "new_information": True,
                "timing_status": "current snapshot, not historical",
                "integration_decision": "reference lookup only; contacts excluded",
            },
        ]
    )


def normalize_all() -> dict[str, pd.DataFrame]:
    generation_long, generation_system = normalize_generation_capacity()
    intertie_events, intertie_hourly = normalize_intertie_outages()
    smp_intervals, smp_hourly = normalize_smp()
    unit_events, unit_hourly = normalize_unit_commitment()
    participants, participant_agents = normalize_pool_participants()
    datasets = {
        "actual_forecast": normalize_actual_forecast(),
        "generation_capacity_by_fuel": generation_long,
        "generation_capacity_system": generation_system,
        "load_outage_forecast": normalize_load_outage_forecast(),
        "intertie_outage_events": intertie_events,
        "intertie_outage_hourly": intertie_hourly,
        "pool_price_forecast": normalize_pool_price(),
        "smp_intervals": smp_intervals,
        "smp_hourly": smp_hourly,
        "asset_list": normalize_asset_list(),
        "energy_merit_order_hourly": normalize_energy_merit_order(),
        "operating_reserve_hourly": normalize_operating_reserve(),
        "unit_commitment_events": unit_events,
        "unit_commitment_hourly": unit_hourly,
        "pool_participants": participants,
        "pool_participant_agents": participant_agents,
    }
    datasets["aeso_api_dataset"] = build_aeso_api_dataset(datasets)
    return datasets


def process_aeso_api(*, overwrite: bool = False) -> dict[str, pd.DataFrame]:
    source_paths = _all_source_files()
    manifest = build_manifest(
        "aeso_api_normalized",
        source_paths,
        preprocessing_code_paths(Path(__file__)),
        configuration={
            "timezone": "UTC",
            "master_integration": "hourly_prefixed_fields",
            "actual_fields": "retained_for_overlap_audit",
            "reference_snapshots": "not_repeated_at_hourly_grain",
        },
    )
    artifact_paths = [
        *OUTPUTS.values(),
        AUDIT_CHECKS,
        DATASET_SUMMARY,
        OVERLAP_AUDIT,
        FIELD_REGISTER,
    ]
    if not overwrite and outputs_are_current(artifact_paths, manifest):
        print("AESO API preprocessing outputs are current; nothing to do.")
        return {name: pd.read_parquet(path) for name, path in OUTPUTS.items()}

    datasets = normalize_all()
    checks, summary = build_audits(datasets)
    overlap = build_overlap_audit(datasets)
    field_register = build_field_register()
    AESO_API_AUDITS_DIR.mkdir(parents=True, exist_ok=True)
    checks.to_csv(AUDIT_CHECKS, index=False)
    summary.to_csv(DATASET_SUMMARY, index=False)
    overlap.to_csv(OVERLAP_AUDIT, index=False)
    field_register.to_csv(FIELD_REGISTER, index=False)

    if not audit_passes(checks):
        failed = checks.loc[checks["severity"].eq("error") & ~checks["pass"]]
        raise ValueError(
            "AESO API preprocessing failed quality checks: "
            + "; ".join(failed["check"].astype(str))
        )

    AESO_API_PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in datasets.items():
        frame.to_parquet(OUTPUTS[name], index=False)
    write_manifests(artifact_paths, manifest)
    print(summary.to_string(index=False))
    print(f"Wrote {len(datasets)} normalized AESO datasets.")
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    process_aeso_api(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
