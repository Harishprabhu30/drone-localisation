#!/usr/bin/env python3
"""RG2.3-B: development-only post-freeze consensus accuracy audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
import yaml

from uavloc.evaluation.reference import load_reference_xy


ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/local_map_neighborhood_rg2_3b.yaml"

CONSENSUS_REQUIRED = {
    "trajectory_key", "trajectory_id", "query_id", "subset",
    "projection_view", "method", "point_count",
    "consensus_easting", "consensus_northing",
    "anchor_projection_easting", "anchor_projection_northing",
    "anchor_projected_inside_tile", "cluster_support_count",
    "radial_median_m", "radial_p90_m", "radial_max_m",
    "pairwise_median_m", "pairwise_p90_m", "pairwise_max_m",
    "bbox_diagonal_m",
}
PAIR_REQUIRED = {
    "query_id", "tile_id", "is_anchor", "in_immediate",
    "finite_projection", "projected_inside_tile",
    "projected_easting", "projected_northing",
}
TILE_REQUIRED = {"tile_id", "center_easting", "center_northing"}


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")


def bool_series(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {
        "true": True, "false": False, "1": True, "0": False,
        "yes": True, "no": False,
    }
    out = series.astype(str).str.strip().str.lower().map(mapping)
    if out.isna().any():
        bad = sorted(series[out.isna()].astype(str).unique().tolist())
        raise RuntimeError(f"{label} has non-boolean values: {bad[:10]}")
    return out.astype(bool)


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "RG2.3-B":
        raise RuntimeError("Expected RG2.3-B config")
    if cfg.get("map_variant") != "768_s256" or cfg.get("map_crs") != "EPSG:3346":
        raise RuntimeError("Map contract changed")

    ac = cfg["audit_contract"]
    if ac.get("deployment_method_selection_allowed") is not False:
        raise RuntimeError("Deployment method selection must remain disabled")
    if ac.get("reference_tuning_allowed") is not False:
        raise RuntimeError("Reference tuning must remain disabled")
    if ac.get("primary_projection_view") != "all_finite":
        raise RuntimeError("Primary projection view changed")
    if ac.get("methods") != [
        "coordinate_median",
        "geometric_median",
        "medoid",
        "radius_cluster_medoid",
        "inlier_huber_centroid",
    ]:
        raise RuntimeError("Frozen method family changed")

    boundary = cfg["scientific_boundary"]
    if boundary.get("development_reference_allowed") is not True:
        raise RuntimeError("Development reference must be explicitly allowed")
    if boundary.get("reference_after_consensus_freeze_only") is not True:
        raise RuntimeError("Reference-after-freeze boundary missing")
    for key in (
        "blind_stress_loaded", "blind_stress_reference_allowed",
        "consensus_coordinates_mutable", "orb_rerun_enabled",
        "projection_rerun_enabled", "consensus_rerun_enabled",
        "method_tuning_enabled", "bootstrap_enabled", "state_enabled",
    ):
        if boundary.get(key) is not False:
            raise RuntimeError(f"{key} must be false")


def verify_rg23a(cfg: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    block = cfg["rg2_3a"]
    report_path = resolve(block["report"])
    consensus_path = resolve(block["development_consensus"])
    if not report_path.is_file() or not consensus_path.is_file():
        raise FileNotFoundError("Missing RG2.3-A freeze input")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_RG2_3A_REFERENCE_FREE_CONSENSUS_FREEZE":
        raise RuntimeError("RG2.3-A report not PASS")
    boundary = report.get("scientific_boundary", {})
    if boundary.get("reference_loaded") is not False:
        raise RuntimeError("RG2.3-A reference boundary violated")
    if boundary.get("accuracy_evaluation_executed") is not False:
        raise RuntimeError("RG2.3-A accuracy evaluation already executed")

    dev = report.get("trajectories", {}).get("development", {})
    blind = report.get("trajectories", {}).get("blind_stress", {})
    if dev.get("output", {}).get("sha256") != block["expected_development_consensus_sha256"]:
        raise RuntimeError("RG2.3-A development consensus hash/report mismatch")
    if blind.get("output", {}).get("sha256") != block["expected_blind_stress_consensus_sha256"]:
        raise RuntimeError("RG2.3-A blind-stress consensus hash/report mismatch")
    if sha256_file(consensus_path) != block["expected_development_consensus_sha256"]:
        raise RuntimeError("RG2.3-A development consensus bytes changed")
    return consensus_path, report


def verify_rg22(cfg: dict[str, Any]) -> Path:
    block = cfg["rg2_2"]
    report_path = resolve(block["report"])
    pair_path = resolve(block["development_pair_evidence"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_RG2_2_REFERENCE_FREE_MULTI_TILE_PROJECTION_FREEZE":
        raise RuntimeError("RG2.2 report not PASS")
    if report.get("scientific_boundary", {}).get("reference_loaded") is not False:
        raise RuntimeError("RG2.2 reference boundary violated")
    if sha256_file(pair_path) != block["expected_development_pair_sha256"]:
        raise RuntimeError("RG2.2 development pair evidence changed")
    return pair_path


def verify_rg20(cfg: dict[str, Any]) -> Path:
    block = cfg["rg2_0"]
    report_path = resolve(block["report"])
    tile_path = resolve(block["tile_geometry"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_RG2_0_MAP_TOPOLOGY_PREFLIGHT":
        raise RuntimeError("RG2.0 report not PASS")
    if sha256_file(tile_path) != report["provenance"]["tile_geometry_sha256"]:
        raise RuntimeError("RG2.0 tile geometry changed")
    return tile_path


def prepare_frozen_inputs(
    consensus_path: Path,
    pair_path: Path,
    tile_path: Path,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    consensus = pd.read_csv(consensus_path).copy()
    pair = pd.read_csv(pair_path).copy()
    tile = pd.read_csv(tile_path).copy()
    require_columns(consensus, CONSENSUS_REQUIRED, "RG2.3-A consensus")
    require_columns(pair, PAIR_REQUIRED, "RG2.2 pair evidence")
    require_columns(tile, TILE_REQUIRED, "RG2.0 tile geometry")

    consensus["query_id"] = pd.to_numeric(consensus["query_id"], errors="raise").astype(int)
    pair["query_id"] = pd.to_numeric(pair["query_id"], errors="raise").astype(int)
    for c in ("is_anchor", "in_immediate", "finite_projection", "projected_inside_tile"):
        pair[c] = bool_series(pair[c], c)
    consensus["anchor_projected_inside_tile"] = bool_series(
        consensus["anchor_projected_inside_tile"], "anchor_projected_inside_tile"
    )

    for c in (
        "consensus_easting", "consensus_northing",
        "anchor_projection_easting", "anchor_projection_northing",
        "radial_median_m", "radial_p90_m", "radial_max_m",
        "pairwise_median_m", "pairwise_p90_m", "pairwise_max_m",
        "bbox_diagonal_m",
    ):
        consensus[c] = pd.to_numeric(consensus[c], errors="raise").astype(float)
    for c in ("projected_easting", "projected_northing"):
        pair[c] = pd.to_numeric(pair[c], errors="coerce")
    tile["tile_id"] = tile["tile_id"].astype(str)
    pair["tile_id"] = pair["tile_id"].astype(str)
    tile["center_easting"] = pd.to_numeric(tile["center_easting"], errors="raise").astype(float)
    tile["center_northing"] = pd.to_numeric(tile["center_northing"], errors="raise").astype(float)

    if len(consensus) != int(cfg["rg2_3a"]["expected_development_rows"]):
        raise RuntimeError("RG2.3-A development row count mismatch")
    if consensus["query_id"].nunique() != int(cfg["development"]["expected_queries"]):
        raise RuntimeError("RG2.3-A query count mismatch")
    if len(pair) != int(cfg["rg2_2"]["expected_development_pairs"]):
        raise RuntimeError("RG2.2 development pair count mismatch")

    expected_methods = set(cfg["audit_contract"]["methods"])
    expected_subsets = set(cfg["audit_contract"]["subsets"])
    expected_views = set(cfg["audit_contract"]["projection_views"])
    if set(consensus["method"]) != expected_methods:
        raise RuntimeError("RG2.3-A method set changed")
    if set(consensus["subset"]) != expected_subsets:
        raise RuntimeError("RG2.3-A subset set changed")
    if set(consensus["projection_view"]) != expected_views:
        raise RuntimeError("RG2.3-A projection-view set changed")

    key_cols = ["query_id", "subset", "projection_view", "method"]
    if consensus[key_cols].duplicated().any():
        raise RuntimeError("Duplicate RG2.3-A consensus key")
    expected_per_query = len(expected_methods) * len(expected_subsets) * len(expected_views)
    counts = consensus.groupby("query_id").size()
    if not (counts == expected_per_query).all():
        raise RuntimeError("RG2.3-A missing consensus combinations")

    anchors = pair[pair["is_anchor"]].copy()
    if len(anchors) != cfg["development"]["expected_queries"]:
        raise RuntimeError("RG2.2 anchor count mismatch")
    if anchors["query_id"].duplicated().any():
        raise RuntimeError("Duplicate RG2.2 anchor query")
    if not anchors["finite_projection"].all():
        raise RuntimeError("Frozen anchor projection unexpectedly non-finite")

    if tile["tile_id"].duplicated().any():
        raise RuntimeError("Duplicate tile geometry IDs")
    return consensus, pair, tile


def add_error(frame: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    dx = frame[x_col].astype(float) - frame["gt_x"].astype(float)
    dy = frame[y_col].astype(float) - frame["gt_y"].astype(float)
    return np.hypot(dx, dy)


def summary(errors: pd.Series, thresholds: list[float]) -> dict[str, Any]:
    e = pd.to_numeric(errors, errors="raise").astype(float)
    if len(e) == 0 or not np.isfinite(e.to_numpy()).all():
        raise RuntimeError("Error summary requires finite non-empty values")
    return {
        "count": int(len(e)),
        "rmse_m": float(np.sqrt(np.mean(np.square(e)))),
        "mean_m": float(e.mean()),
        "median_m": float(e.median()),
        "p75_m": float(e.quantile(0.75)),
        "p90_m": float(e.quantile(0.90)),
        "p95_m": float(e.quantile(0.95)),
        "max_m": float(e.max()),
        "threshold_hits": {
            f"le{int(t) if float(t).is_integer() else t}m": int((e <= float(t)).sum())
            for t in thresholds
        },
    }


def paired(candidate: pd.Series, baseline: pd.Series) -> dict[str, Any]:
    c = pd.to_numeric(candidate, errors="raise").astype(float)
    b = pd.to_numeric(baseline, errors="raise").astype(float)
    if len(c) != len(b):
        raise RuntimeError("Paired comparison length mismatch")
    delta = c.to_numpy() - b.to_numpy()
    eps = 1e-9
    return {
        "improved_count": int((delta < -eps).sum()),
        "worsened_count": int((delta > eps).sum()),
        "tie_count": int((np.abs(delta) <= eps).sum()),
        "median_delta_m": float(np.median(delta)),
        "mean_delta_m": float(np.mean(delta)),
        "p90_delta_m": float(np.quantile(delta, 0.90)),
    }


def rank_correlation(x: pd.Series, y: pd.Series) -> dict[str, float | None]:
    xv = pd.to_numeric(x, errors="coerce").astype(float)
    yv = pd.to_numeric(y, errors="coerce").astype(float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[mask]
    yv = yv[mask]
    if len(xv) < 3 or xv.nunique() < 2 or yv.nunique() < 2:
        return {"pearson": None, "spearman": None}
    pearson = float(np.corrcoef(xv.to_numpy(), yv.to_numpy())[0, 1])
    xr = xv.rank(method="average").to_numpy()
    yr = yv.rank(method="average").to_numpy()
    spearman = float(np.corrcoef(xr, yr)[0, 1])
    return {"pearson": pearson, "spearman": spearman}


def evaluate(
    consensus: pd.DataFrame,
    pair: pd.DataFrame,
    tile: pd.DataFrame,
    reference: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ref = reference[["gt_x", "gt_y"]].reset_index()
    qids = sorted(consensus["query_id"].unique().tolist())
    if sorted(reference.index.astype(int).tolist()) != qids:
        raise RuntimeError("Reference query set differs from frozen RG2.3-A queries")

    evaluated = consensus.merge(ref, on="query_id", how="left", validate="many_to_one")
    evaluated["consensus_error_m"] = add_error(
        evaluated, "consensus_easting", "consensus_northing"
    )
    evaluated["anchor_projection_error_m"] = add_error(
        evaluated, "anchor_projection_easting", "anchor_projection_northing"
    )

    anchors = pair[pair["is_anchor"]].copy()
    anchors = anchors.merge(
        tile[["tile_id", "center_easting", "center_northing"]],
        on="tile_id", how="left", validate="many_to_one",
    ).merge(ref, on="query_id", how="left", validate="one_to_one")
    if anchors[["center_easting", "center_northing"]].isna().any().any():
        raise RuntimeError("Anchor tile center join failed")
    anchors["anchor_tile_center_error_m"] = add_error(
        anchors, "center_easting", "center_northing"
    )
    anchors["anchor_projection_error_m"] = add_error(
        anchors, "projected_easting", "projected_northing"
    )
    baseline = anchors[[
        "query_id", "tile_id", "anchor_tile_center_error_m",
        "anchor_projection_error_m", "projected_inside_tile",
    ]].rename(columns={
        "tile_id": "anchor_tile_id",
        "projected_inside_tile": "anchor_projected_inside_tile_rg22",
    }).sort_values("query_id").reset_index(drop=True)

    evaluated = evaluated.merge(
        baseline[["query_id", "anchor_tile_center_error_m"]],
        on="query_id", how="left", validate="many_to_one",
    )

    oracle_rows = []
    for qid, group in pair.groupby("query_id", sort=True):
        gt = ref[ref["query_id"] == qid].iloc[0]
        g = group[group["finite_projection"]].copy()
        g["projection_error_m"] = np.hypot(
            g["projected_easting"].astype(float) - float(gt["gt_x"]),
            g["projected_northing"].astype(float) - float(gt["gt_y"]),
        )
        for subset in cfg["audit_contract"]["subsets"]:
            base = g if subset == "full_overlap" else g[g["in_immediate"]]
            for view in cfg["audit_contract"]["projection_views"]:
                use = base if view == "all_finite" else base[base["projected_inside_tile"]]
                if len(use) == 0:
                    raise RuntimeError(f"No oracle diagnostic points q{qid} {subset} {view}")
                best = use.sort_values(
                    ["projection_error_m", "tile_id"], kind="mergesort"
                ).iloc[0]
                oracle_rows.append({
                    "query_id": int(qid),
                    "subset": subset,
                    "projection_view": view,
                    "best_tile_id": str(best["tile_id"]),
                    "best_projection_error_m": float(best["projection_error_m"]),
                    "candidate_count": int(len(use)),
                })
    oracle = pd.DataFrame(oracle_rows)

    thresholds = [float(v) for v in cfg["audit_contract"]["error_thresholds_m"]]
    report: dict[str, Any] = {
        "baselines": {
            "anchor_tile_center": summary(
                baseline["anchor_tile_center_error_m"], thresholds
            ),
            "anchor_subtile_projection": summary(
                baseline["anchor_projection_error_m"], thresholds
            ),
            "anchor_projection_vs_tile_center": paired(
                baseline["anchor_projection_error_m"],
                baseline["anchor_tile_center_error_m"],
            ),
        },
        "consensus": {},
        "oracle_diagnostics": {},
    }

    for (subset, view, method), group in evaluated.groupby(
        ["subset", "projection_view", "method"], sort=True
    ):
        group = group.sort_values("query_id")
        key = f"{subset}|{view}|{method}"
        report["consensus"][key] = {
            "errors": summary(group["consensus_error_m"], thresholds),
            "vs_anchor_projection": paired(
                group["consensus_error_m"], group["anchor_projection_error_m"]
            ),
            "vs_anchor_tile_center": paired(
                group["consensus_error_m"], group["anchor_tile_center_error_m"]
            ),
            "dispersion_error_association": {
                "radial_median_m": rank_correlation(
                    group["radial_median_m"], group["consensus_error_m"]
                ),
                "pairwise_p90_m": rank_correlation(
                    group["pairwise_p90_m"], group["consensus_error_m"]
                ),
                "bbox_diagonal_m": rank_correlation(
                    group["bbox_diagonal_m"], group["consensus_error_m"]
                ),
            },
        }

    for (subset, view), group in oracle.groupby(
        ["subset", "projection_view"], sort=True
    ):
        key = f"{subset}|{view}"
        report["oracle_diagnostics"][key] = summary(
            group["best_projection_error_m"], thresholds
        )

    return evaluated, baseline, oracle, report


def deterministic_csv(frame: pd.DataFrame) -> bytes:
    text = frame.to_csv(index=False, lineterminator="\n", float_format="%.9f")
    return text.encode("utf-8")


def deterministic_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_immutable(files: dict[Path, bytes]) -> None:
    changed = [str(p) for p, data in files.items() if p.exists() and p.read_bytes() != data]
    if changed:
        raise RuntimeError("Refusing to overwrite differing RG2.3-B outputs: " + ", ".join(changed))
    for path, data in files.items():
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RG2.3-B post-freeze projection consensus accuracy audit"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verify_config(cfg)

    consensus_path, rg23a_report = verify_rg23a(cfg)
    pair_path = verify_rg22(cfg)
    tile_path = verify_rg20(cfg)
    consensus, pair, tile = prepare_frozen_inputs(
        consensus_path, pair_path, tile_path, cfg
    )

    # Critical boundary: all frozen-coordinate and lineage checks are complete
    # before the reference attachment is opened.
    reference_path = resolve(cfg["development"]["reference_attachment"])
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    print("Frozen RG2.3-A coordinates verified before reference attachment.")
    print("Attaching development reference for RG2.3-B evaluation only.")
    reference = load_reference_xy(reference_path)

    evaluated, baseline, oracle, analysis = evaluate(
        consensus, pair, tile, reference, cfg
    )

    out_root = resolve(cfg["output_root"])
    evaluated_path = out_root / "rg2_3b_consensus_accuracy.csv"
    baseline_path = out_root / "rg2_3b_anchor_baselines.csv"
    oracle_path = out_root / "rg2_3b_oracle_projection_diagnostics.csv"

    files = {
        evaluated_path: deterministic_csv(
            evaluated.sort_values(
                ["query_id", "subset", "projection_view", "method"],
                kind="mergesort",
            )
        ),
        baseline_path: deterministic_csv(baseline),
        oracle_path: deterministic_csv(
            oracle.sort_values(
                ["query_id", "subset", "projection_view"], kind="mergesort"
            )
        ),
    }

    report = {
        "stage": "RG2.3-B",
        "status": "PASS_RG2_3B_POSTFREEZE_CONSENSUS_ACCURACY_AUDIT",
        "scope": "development_only_postfreeze",
        "query_count": int(cfg["development"]["expected_queries"]),
        "analysis": analysis,
        "scientific_boundary": {
            "consensus_generated_before_reference": True,
            "consensus_coordinates_changed_after_reference": False,
            "blind_stress_loaded": False,
            "blind_stress_reference_used": False,
            "orb_rerun": False,
            "projection_rerun": False,
            "consensus_rerun": False,
            "deployment_method_selected": False,
            "reference_tuning_executed": False,
            "bootstrap_state_modified": False,
        },
        "interpretation": {
            "oracle_best_is_diagnostic_only": True,
            "inside_only_is_predeclared_diagnostic": True,
            "all_finite_is_primary_projection_view": True,
            "no_deployment_winner_selected": True,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "reference_attachment_sha256": sha256_file(reference_path),
            "rg2_3a_report_sha256": sha256_file(resolve(cfg["rg2_3a"]["report"])),
            "rg2_3a_development_consensus_sha256": sha256_file(consensus_path),
            "rg2_2_development_pair_sha256": sha256_file(pair_path),
            "rg2_0_tile_geometry_sha256": sha256_file(tile_path),
        },
        "outputs": {},
    }

    for label, path in (
        ("consensus_accuracy", evaluated_path),
        ("anchor_baselines", baseline_path),
        ("oracle_projection_diagnostics", oracle_path),
    ):
        report["outputs"][label] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(files[path]).hexdigest(),
        }

    report_path = out_root / "rg2_3b_accuracy_report.json"
    report_bytes = deterministic_json(report)
    files[report_path] = report_bytes
    write_immutable(files)

    print("Status:", report["status"])
    b = analysis["baselines"]
    print(
        "anchor_tile_center:",
        f"median={b['anchor_tile_center']['median_m']:.3f}",
        f"rmse={b['anchor_tile_center']['rmse_m']:.3f}",
        f"le40={b['anchor_tile_center']['threshold_hits']['le40m']}/403",
        f"le80={b['anchor_tile_center']['threshold_hits']['le80m']}/403",
    )
    print(
        "anchor_subtile_projection:",
        f"median={b['anchor_subtile_projection']['median_m']:.3f}",
        f"rmse={b['anchor_subtile_projection']['rmse_m']:.3f}",
        f"le40={b['anchor_subtile_projection']['threshold_hits']['le40m']}/403",
        f"le80={b['anchor_subtile_projection']['threshold_hits']['le80m']}/403",
    )
    print("Consensus methods (primary all_finite):")
    for key in sorted(analysis["consensus"]):
        if "|all_finite|" not in key:
            continue
        x = analysis["consensus"][key]
        print(
            f"  {key}: median={x['errors']['median_m']:.3f} "
            f"rmse={x['errors']['rmse_m']:.3f} "
            f"le40={x['errors']['threshold_hits']['le40m']}/403 "
            f"le80={x['errors']['threshold_hits']['le80m']}/403 "
            f"improved_vs_anchor={x['vs_anchor_projection']['improved_count']}/403"
        )
    print("Oracle best frozen projection diagnostics:")
    for key in sorted(analysis["oracle_diagnostics"]):
        x = analysis["oracle_diagnostics"][key]
        print(
            f"  {key}: median={x['median_m']:.3f} "
            f"le40={x['threshold_hits']['le40m']}/403 "
            f"le80={x['threshold_hits']['le80m']}/403"
        )
    print("Blind stress loaded: False")
    print("Deployment method selected: False")
    print("Report:", report_path)
    print("Consensus accuracy SHA256:", report["outputs"]["consensus_accuracy"]["sha256"])


if __name__ == "__main__":
    main()
