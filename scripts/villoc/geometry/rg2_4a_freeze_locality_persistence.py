#!/usr/bin/env python3
"""RG2.4-A reference-free locality persistence / evidence-independence freeze."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
import yaml

from uavloc.data.trajectory_adapter import FORBIDDEN_BLIND_REFERENCE_COLUMNS
from uavloc.geometry.locality_persistence import (
    euclidean,
    max_pairwise_span,
    path_length,
    segment_localities,
    support_redundancy_ratio,
)

ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/local_map_neighborhood_rg2_4a.yaml"

CONSENSUS_REQUIRED = {
    "trajectory_key", "trajectory_id", "query_id", "subset",
    "projection_view", "method", "consensus_easting", "consensus_northing",
    "radial_median_m", "pairwise_p90_m", "bbox_diagonal_m",
}
PAIR_REQUIRED = {
    "query_id", "tile_id", "is_anchor", "in_immediate",
}
MANIFEST_REQUIRED = {
    "query_id", "timestamp_s", "reference_available",
}
FORBIDDEN_PARTIAL = (
    "ground_truth", "oracle", "eval_ref_", "reference_x", "reference_y",
    "projected_error", "tile_center_error", "chosen_error", "gt_error",
)


def resolve(value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def assert_blind_safe(frame: pd.DataFrame, label: str) -> None:
    cols = {str(c).strip().lower() for c in frame.columns}
    exact = {
        str(c).strip().lower()
        for c in FORBIDDEN_BLIND_REFERENCE_COLUMNS
        if str(c).strip().lower() != "reference_available"
    }
    leaked = sorted(
        (cols & exact)
        | {
            c for c in cols
            if c != "reference_available"
            and any(token in c for token in FORBIDDEN_PARTIAL)
        }
    )
    if leaked:
        raise RuntimeError(f"{label} contains forbidden reference/eval columns: {leaked}")


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")


def bool_series(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {
        "true": True, "false": False, "1": True, "0": False,
        "yes": True, "no": False, "y": True, "n": False,
    }
    out = series.astype(str).str.strip().str.lower().map(mapping)
    if out.isna().any():
        bad = sorted(series[out.isna()].astype(str).unique().tolist())
        raise RuntimeError(f"{label} has non-boolean values: {bad[:10]}")
    return out.astype(bool)


def canonical_csv(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise RuntimeError("Cannot freeze empty RG2.4-A table")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        cooked = {}
        for k, v in row.items():
            if isinstance(v, bool):
                cooked[k] = "true" if v else "false"
            elif isinstance(v, float):
                cooked[k] = "" if not math.isfinite(v) else f"{v:.9f}"
            elif v is None:
                cooked[k] = ""
            else:
                cooked[k] = v
        writer.writerow(cooked)
    return stream.getvalue().encode("utf-8")


def canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def freeze(files: dict[Path, bytes]) -> None:
    bad = [str(p) for p, data in files.items() if p.exists() and p.read_bytes() != data]
    if bad:
        raise RuntimeError("Refusing to overwrite differing RG2.4-A freeze: " + ", ".join(bad))
    for path, data in files.items():
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as f:
            f.write(data)


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "RG2.4-A":
        raise RuntimeError("Expected RG2.4-A config")
    if cfg.get("map_variant") != "768_s256" or cfg.get("map_crs") != "EPSG:3346":
        raise RuntimeError("Map contract changed")
    contract = cfg["locality_contract"]
    if contract.get("projection_view") != "all_finite":
        raise RuntimeError("RG2.4-A must use primary all_finite view")
    if contract.get("link_thresholds_m") != {
        "quarter_stride": 12.8,
        "half_stride": 25.6,
    }:
        raise RuntimeError("Locality thresholds changed")
    if float(contract.get("contiguous_gap_multiplier")) != 1.5:
        raise RuntimeError("Temporal continuity contract changed")
    if contract.get("method_selection_enabled") is not False:
        raise RuntimeError("Method selection must remain disabled")

    for key, value in cfg["scientific_boundary"].items():
        if value is not False:
            raise RuntimeError(f"{key} must be false")


def verify_rg23a(cfg: dict[str, Any]) -> dict[str, Any]:
    path = resolve(cfg["rg2_3a"]["report"])
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_RG2_3A_REFERENCE_FREE_CONSENSUS_FREEZE":
        raise RuntimeError("RG2.3-A report not PASS")
    boundary = report.get("scientific_boundary", {})
    if boundary.get("reference_loaded") is not False:
        raise RuntimeError("RG2.3-A reference boundary violated")
    if boundary.get("accuracy_evaluation_executed") is not False:
        raise RuntimeError("RG2.3-A accuracy boundary violated")
    return report


def load_inputs(
    cfg: dict[str, Any],
    rg23a_report: dict[str, Any],
    key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    c = cfg["rg2_3a"][key]
    p = cfg["rg2_2"][key]
    consensus_path = resolve(c["consensus"])
    pair_path = resolve(p["pair_evidence"])
    manifest_path = resolve(cfg["manifests"][key])

    if sha256_file(consensus_path) != c["expected_sha256"]:
        raise RuntimeError(f"{key} RG2.3-A consensus hash changed")
    if sha256_file(pair_path) != p["expected_sha256"]:
        raise RuntimeError(f"{key} RG2.2 pair hash changed")
    if rg23a_report["trajectories"][key]["output"]["sha256"] != c["expected_sha256"]:
        raise RuntimeError(f"{key} RG2.3-A report hash mismatch")

    consensus = pd.read_csv(consensus_path).copy()
    pair = pd.read_csv(pair_path).copy()
    manifest = pd.read_csv(manifest_path).copy()
    for frame, label in (
        (consensus, f"{key} consensus"),
        (pair, f"{key} pair"),
        (manifest, f"{key} manifest"),
    ):
        assert_blind_safe(frame, label)

    require_columns(consensus, CONSENSUS_REQUIRED, f"{key} consensus")
    require_columns(pair, PAIR_REQUIRED, f"{key} pair")
    require_columns(manifest, MANIFEST_REQUIRED, f"{key} manifest")

    consensus["query_id"] = pd.to_numeric(consensus["query_id"], errors="raise").astype(int)
    pair["query_id"] = pd.to_numeric(pair["query_id"], errors="raise").astype(int)
    manifest["query_id"] = pd.to_numeric(manifest["query_id"], errors="raise").astype(int)
    manifest["timestamp_s"] = pd.to_numeric(manifest["timestamp_s"], errors="raise").astype(float)

    pair["is_anchor"] = bool_series(pair["is_anchor"], "is_anchor")
    pair["in_immediate"] = bool_series(pair["in_immediate"], "in_immediate")

    refs = manifest["reference_available"].astype(str).str.strip().str.lower()
    if not refs.isin(["false", "0", "no", "n", "f"]).all():
        raise RuntimeError(f"{key} manifest exposes reference")

    if manifest["query_id"].duplicated().any():
        raise RuntimeError(f"{key} manifest query IDs duplicated")
    if len(manifest) != int(c["expected_queries"]):
        raise RuntimeError(f"{key} query count mismatch")

    primary = consensus[consensus["projection_view"] == cfg["locality_contract"]["projection_view"]].copy()
    expected_methods = set(cfg["locality_contract"]["methods"])
    expected_subsets = set(cfg["locality_contract"]["subsets"])
    if set(primary["method"]) != expected_methods:
        raise RuntimeError(f"{key} method set changed")
    if set(primary["subset"]) != expected_subsets:
        raise RuntimeError(f"{key} subset set changed")
    expected_rows = int(c["expected_queries"]) * len(expected_methods) * len(expected_subsets)
    if len(primary) != expected_rows:
        raise RuntimeError(f"{key} primary consensus row count mismatch")
    if primary[["query_id", "subset", "method"]].duplicated().any():
        raise RuntimeError(f"{key} duplicate primary consensus key")

    anchors = pair[pair["is_anchor"]][["query_id", "tile_id"]].copy()
    if len(anchors) != int(c["expected_queries"]) or anchors["query_id"].duplicated().any():
        raise RuntimeError(f"{key} anchor identity mismatch")
    anchor_map = {
        int(row.query_id): str(row.tile_id)
        for row in anchors.itertuples(index=False)
    }

    support: dict[str, dict[int, set[str]]] = {"immediate": {}, "full_overlap": {}}
    for qid, group in pair.groupby("query_id", sort=True):
        support["full_overlap"][int(qid)] = set(group["tile_id"].astype(str))
        support["immediate"][int(qid)] = set(
            group.loc[group["in_immediate"], "tile_id"].astype(str)
        )
        if not support["immediate"][int(qid)]:
            raise RuntimeError(f"{key} q{qid} missing immediate support")

    provenance = {
        "consensus_sha256": sha256_file(consensus_path),
        "pair_evidence_sha256": sha256_file(pair_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    return primary, manifest, pair, {
        "anchor_map": anchor_map,
        "support": support,
        "provenance": provenance,
    }


def build_localities(
    primary: pd.DataFrame,
    manifest: pd.DataFrame,
    context: dict[str, Any],
    cfg: dict[str, Any],
    trajectory_key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    m = manifest[["query_id", "timestamp_s"]].copy()
    transitions: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    track_summaries: dict[str, Any] = {}

    for (subset, method), track in primary.groupby(["subset", "method"], sort=True):
        track = track.merge(m, on="query_id", how="left", validate="one_to_one")
        track = track.sort_values("timestamp_s", kind="mergesort").reset_index(drop=True)
        if track["timestamp_s"].isna().any():
            raise RuntimeError("Timestamp merge failed")
        pts = track[["consensus_easting", "consensus_northing"]].to_numpy(float)
        timestamps = track["timestamp_s"].to_numpy(float)

        for threshold_name, threshold_m in cfg["locality_contract"]["link_thresholds_m"].items():
            steps, nominal_dt, max_dt = segment_localities(
                timestamps,
                pts,
                link_threshold_m=float(threshold_m),
                contiguous_gap_multiplier=float(cfg["locality_contract"]["contiguous_gap_multiplier"]),
            )
            locality_by_row = [step.locality_index for step in steps]
            track_id = f"{subset}|{method}|{threshold_name}"

            for row, step in zip(track.itertuples(index=False), steps):
                transitions.append({
                    "trajectory_key": trajectory_key,
                    "trajectory_id": str(row.trajectory_id),
                    "track_id": track_id,
                    "subset": subset,
                    "method": method,
                    "threshold_name": threshold_name,
                    "link_threshold_m": float(threshold_m),
                    "nominal_dt_s": float(nominal_dt),
                    "contiguous_max_dt_s": float(max_dt),
                    "query_id": int(row.query_id),
                    "timestamp_s": float(row.timestamp_s),
                    "consensus_easting": float(row.consensus_easting),
                    "consensus_northing": float(row.consensus_northing),
                    "locality_index": int(step.locality_index),
                    "dt_from_previous_s": step.dt_s,
                    "jump_from_previous_m": step.jump_m,
                    "contiguous_time": bool(step.contiguous_time),
                    "linked_to_previous": bool(step.linked_to_previous),
                    "anchor_tile_id": context["anchor_map"][int(row.query_id)],
                })

            localities = []
            for locality_index in sorted(set(locality_by_row)):
                mask = np.array(locality_by_row) == locality_index
                group = track.loc[mask].copy().reset_index(drop=True)
                group_pts = group[["consensus_easting", "consensus_northing"]].to_numpy(float)
                qids = group["query_id"].astype(int).tolist()
                support_sets = [context["support"][subset][qid] for qid in qids]
                distinct_support, observations, redundancy = support_redundancy_ratio(support_sets)
                anchor_tiles = [context["anchor_map"][qid] for qid in qids]
                anchor_changes = sum(a != b for a, b in zip(anchor_tiles[:-1], anchor_tiles[1:]))
                duration = float(group["timestamp_s"].iloc[-1] - group["timestamp_s"].iloc[0])
                row = {
                    "trajectory_key": trajectory_key,
                    "trajectory_id": str(group["trajectory_id"].iloc[0]),
                    "track_id": track_id,
                    "subset": subset,
                    "method": method,
                    "threshold_name": threshold_name,
                    "link_threshold_m": float(threshold_m),
                    "locality_index": int(locality_index),
                    "start_query_id": int(qids[0]),
                    "end_query_id": int(qids[-1]),
                    "start_timestamp_s": float(group["timestamp_s"].iloc[0]),
                    "end_timestamp_s": float(group["timestamp_s"].iloc[-1]),
                    "query_count": int(len(group)),
                    "duration_s": duration,
                    "consensus_path_length_m": path_length(group_pts),
                    "consensus_max_pairwise_span_m": max_pairwise_span(group_pts),
                    "start_end_displacement_m": euclidean(group_pts[0], group_pts[-1]),
                    "unique_anchor_tiles": int(len(set(anchor_tiles))),
                    "anchor_tile_changes": int(anchor_changes),
                    "distinct_supporting_tile_windows": int(distinct_support),
                    "supporting_window_observations": int(observations),
                    "support_redundancy_ratio": float(redundancy),
                    "radial_median_median_m": float(group["radial_median_m"].median()),
                    "pairwise_p90_median_m": float(group["pairwise_p90_m"].median()),
                    "bbox_diagonal_median_m": float(group["bbox_diagonal_m"].median()),
                }
                runs.append(row)
                localities.append(row)

            lengths = [r["query_count"] for r in localities]
            track_summaries[track_id] = {
                "locality_count": int(len(localities)),
                "singleton_localities": int(sum(v == 1 for v in lengths)),
                "multiquery_localities": int(sum(v > 1 for v in lengths)),
                "queries_in_multiquery_localities": int(sum(v for v in lengths if v > 1)),
                "queries_in_multiquery_fraction": float(
                    sum(v for v in lengths if v > 1) / float(len(track))
                ),
                "run_length_histogram": {
                    str(int(k)): int(v)
                    for k, v in sorted(Counter(lengths).items())
                },
                "median_run_length_queries": float(np.median(lengths)),
                "max_run_length_queries": int(max(lengths)),
                "nominal_dt_s": float(nominal_dt),
                "contiguous_max_dt_s": float(max_dt),
            }

    return transitions, runs, track_summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RG2.4-A reference-free locality persistence and evidence-independence freeze"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verify_config(cfg)
    rg23a = verify_rg23a(cfg)

    files: dict[Path, bytes] = {}
    reports: dict[str, Any] = {}

    for key in ("development", "blind_stress"):
        primary, manifest, pair, context = load_inputs(cfg, rg23a, key)
        transitions, runs, track_summaries = build_localities(
            primary, manifest, context, cfg, key
        )
        out_root = resolve(cfg["output_root"]) / key
        transition_path = out_root / "rg2_4a_locality_transitions.csv"
        run_path = out_root / "rg2_4a_locality_runs.csv"
        transition_bytes = canonical_csv(transitions)
        run_bytes = canonical_csv(runs)
        files[transition_path] = transition_bytes
        files[run_path] = run_bytes

        reports[key] = {
            "queries": int(manifest["query_id"].nunique()),
            "primary_consensus_rows": int(len(primary)),
            "transition_rows": int(len(transitions)),
            "locality_run_rows": int(len(runs)),
            "track_count": int(len(track_summaries)),
            "track_summaries": track_summaries,
            "provenance": context["provenance"],
            "outputs": {
                "transitions": {
                    "sha256": hashlib.sha256(transition_bytes).hexdigest(),
                    "rows": len(transitions),
                },
                "runs": {
                    "sha256": hashlib.sha256(run_bytes).hexdigest(),
                    "rows": len(runs),
                },
            },
        }

    report = {
        "stage": "RG2.4-A",
        "status": "PASS_RG2_4A_REFERENCE_FREE_LOCALITY_PERSISTENCE_FREEZE",
        "locality_contract": cfg["locality_contract"],
        "trajectories": reports,
        "scientific_boundary": {
            "reference_loaded": False,
            "srt_gps_loaded": False,
            "oracle_labels_loaded": False,
            "rg2_3b_accuracy_loaded": False,
            "method_selected": False,
            "threshold_tuned": False,
            "consensus_rerun": False,
            "orb_rerun": False,
            "projection_rerun": False,
            "bootstrap_modified": False,
            "state_modified": False,
            "relative_frontend_fused": False,
        },
        "interpretation": {
            "locality_ids_are_diagnostic_tracks_not_absolute_truth": True,
            "repeated_overlapping_windows_not_counted_as_independent_by_default": True,
            "support_redundancy_ratio_definition": "1 - distinct_support_windows / support_window_observations",
            "no_accuracy_claim": True,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "locality_module_sha256": sha256_file(
                Path(sys.modules[segment_localities.__module__].__file__).resolve()
            ),
            "rg2_3a_report_sha256": sha256_file(resolve(cfg["rg2_3a"]["report"])),
        },
    }

    report_path = resolve(cfg["output_root"]) / "rg2_4a_freeze_report.json"
    files[report_path] = canonical_json(report)
    freeze(files)

    print("Status:", report["status"])
    for key, r in reports.items():
        print(
            f"{key}: queries={r['queries']} primary_rows={r['primary_consensus_rows']} "
            f"transition_rows={r['transition_rows']} locality_runs={r['locality_run_rows']} "
            f"tracks={r['track_count']}"
        )
        for track_id in sorted(r["track_summaries"]):
            s = r["track_summaries"][track_id]
            print(
                f"  {track_id}: localities={s['locality_count']} "
                f"multiquery_fraction={s['queries_in_multiquery_fraction']:.3f} "
                f"median_run={s['median_run_length_queries']:.1f} "
                f"max_run={s['max_run_length_queries']}"
            )
        print("  transitions SHA256:", r["outputs"]["transitions"]["sha256"])
        print("  runs SHA256:", r["outputs"]["runs"]["sha256"])
    print("Reference loaded: False")
    print("RG2.3-B accuracy loaded: False")
    print("Bootstrap/state modified: False")
    print("Freeze report:", report_path)


if __name__ == "__main__":
    main()
