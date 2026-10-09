#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path.cwd().resolve()
DEFAULT_CONFIG = Path(
    "configs/research/region_candidate_geometry_v1.yaml"
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def numeric_summary(values: list[float] | pd.Series) -> dict[str, Any]:
    series = pd.Series(values, dtype=float)
    series = pd.to_numeric(series, errors="coerce").dropna()

    if not len(series):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }

    return {
        "count": int(len(series)),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
    }


def load_gate(
    path: Path,
    *,
    expected_family: str,
    expected_threshold: int,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    frozen = payload.get("frozen_gate")
    if not isinstance(frozen, dict):
        raise RuntimeError(
            "RG1.4 gate artifact does not contain frozen_gate."
        )

    family = str(frozen.get("family"))
    threshold = int(frozen.get("inlier_threshold"))

    if family != expected_family:
        raise RuntimeError(
            f"Expected frozen family {expected_family!r}, got {family!r}."
        )

    if threshold != expected_threshold:
        raise RuntimeError(
            f"Expected frozen inlier threshold {expected_threshold}, got {threshold}."
        )

    return frozen


def load_anchor_evidence(
    *,
    path: Path,
    expected_queries: int,
    threshold: int,
    accepted_column_optional: bool,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path).copy()

    required = {
        "query_id",
        "pool_rank",
        "tile_id",
        "source_view",
        "source_rank",
        "inliers",
        "projected_easting",
        "projected_northing",
        "projected_inside_tile",
    }

    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"Geometry evidence missing columns: {missing}"
        )

    frame["query_id"] = pd.to_numeric(
        frame["query_id"], errors="raise"
    ).astype(int)

    frame["pool_rank"] = pd.to_numeric(
        frame["pool_rank"], errors="raise"
    ).astype(int)

    frame["source_rank"] = pd.to_numeric(
        frame["source_rank"], errors="raise"
    ).astype(int)

    frame["tile_id"] = frame["tile_id"].astype(str)

    # RG1.1 contains all Top20 rows; RG1.4 already contains only anchors.
    anchor = frame[
        frame["pool_rank"] == 1
    ].copy()

    if len(anchor) != expected_queries:
        raise RuntimeError(
            f"Expected {expected_queries} anchor rows, got {len(anchor)}"
        )

    if anchor["query_id"].nunique() != expected_queries:
        raise RuntimeError(
            "Anchor query IDs are not unique."
        )

    if not (
        (anchor["source_view"].astype(str) == "center_square")
        & (anchor["source_rank"].astype(int) == 1)
    ).all():
        raise RuntimeError(
            "Anchor evidence does not preserve frozen center-square Top1."
        )

    computed_accept = (
        pd.to_numeric(
            anchor["inliers"],
            errors="raise",
        )
        >= int(threshold)
    )

    if "accepted" in anchor.columns:
        stored_accept = anchor["accepted"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
        if not (stored_accept.to_numpy() == computed_accept.to_numpy()).all():
            raise RuntimeError(
                "Stored RG1.4 accepted flags do not match frozen gate."
            )

    anchor["accepted_rg1_5"] = computed_accept.astype(bool)

    return anchor.sort_values(
        "query_id",
        kind="mergesort",
    ).reset_index(drop=True)


def load_manifest(
    path: Path,
    *,
    expected_queries: int,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path).copy()

    required = {
        "query_id",
        "timestamp_s",
        "reference_available",
    }

    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"Canonical manifest missing columns: {missing}"
        )

    frame["query_id"] = pd.to_numeric(
        frame["query_id"],
        errors="raise",
    ).astype(int)

    frame["timestamp_s"] = pd.to_numeric(
        frame["timestamp_s"],
        errors="raise",
    ).astype(float)

    if len(frame) != expected_queries:
        raise RuntimeError(
            f"Expected {expected_queries} manifest rows, got {len(frame)}"
        )

    refs = (
        frame["reference_available"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if not refs.isin(
        ["false", "0", "no", "n", "f"]
    ).all():
        raise RuntimeError(
            "RG1.5 requires blind-safe manifests with reference_available=false."
        )

    return frame[
        [
            "query_id",
            "timestamp_s",
        ]
    ].copy()


def load_tile_centers(path: Path) -> dict[str, tuple[float, float]]:
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path).copy()

    required = {
        "tile_id",
        "center_easting",
        "center_northing",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"Tile index missing center columns: {missing}"
        )

    frame["tile_id"] = frame["tile_id"].astype(str)

    return {
        str(row.tile_id): (
            float(row.center_easting),
            float(row.center_northing),
        )
        for row in frame.itertuples(index=False)
    }


def contiguous_run_lengths(
    accepted: pd.Series,
) -> list[int]:
    values = accepted.astype(bool).tolist()
    runs: list[int] = []
    current = 0

    for value in values:
        if value:
            current += 1
        elif current:
            runs.append(current)
            current = 0

    if current:
        runs.append(current)

    return runs


def trajectory_behavior(
    *,
    manifest: pd.DataFrame,
    anchor: pd.DataFrame,
    tile_centers: dict[str, tuple[float, float]],
    contiguous_gap_multiplier: float,
) -> dict[str, Any]:
    merged = anchor.merge(
        manifest,
        on="query_id",
        how="left",
        validate="one_to_one",
    )

    if merged["timestamp_s"].isna().any():
        raise RuntimeError(
            "Timestamp merge failed for one or more anchor rows."
        )

    merged = merged.sort_values(
        "timestamp_s",
        kind="mergesort",
    ).reset_index(drop=True)

    all_time_diffs = (
        merged["timestamp_s"]
        .diff()
        .dropna()
    )

    nominal_dt = float(
        all_time_diffs.median()
    )

    if nominal_dt <= 0:
        raise RuntimeError(
            "Non-positive nominal trajectory sampling interval."
        )

    contiguous_max_dt = float(
        nominal_dt
        * contiguous_gap_multiplier
    )

    accepted = merged[
        merged["accepted_rg1_5"].astype(bool)
    ].copy()

    accepted_inside_rate = (
        float(
            accepted[
                "projected_inside_tile"
            ].astype(bool).mean()
        )
        if len(accepted)
        else None
    )

    accepted_gaps_s: list[float] = []
    accepted_query_gaps: list[float] = []
    projected_jumps_all: list[float] = []
    projected_speeds_all: list[float] = []
    tile_center_jumps_all: list[float] = []

    projected_jumps_contiguous: list[float] = []
    projected_speeds_contiguous: list[float] = []
    tile_center_jumps_contiguous: list[float] = []

    same_tile_flags_contiguous: list[bool] = []
    same_tile_projected_jumps: list[float] = []
    changed_tile_projected_jumps: list[float] = []
    changed_tile_center_jumps: list[float] = []

    rows = list(
        accepted.itertuples(index=False)
    )

    for previous, current in zip(
        rows[:-1],
        rows[1:],
    ):
        dt = float(
            current.timestamp_s
            - previous.timestamp_s
        )
        qgap = int(
            current.query_id
            - previous.query_id
        )

        accepted_gaps_s.append(dt)
        accepted_query_gaps.append(float(qgap))

        pxy_ok = all(
            math.isfinite(
                float(value)
            )
            for value in (
                previous.projected_easting,
                previous.projected_northing,
                current.projected_easting,
                current.projected_northing,
            )
        )

        jump = None
        speed = None

        if pxy_ok and dt > 0:
            jump = float(
                math.hypot(
                    float(current.projected_easting)
                    - float(previous.projected_easting),
                    float(current.projected_northing)
                    - float(previous.projected_northing),
                )
            )
            speed = float(jump / dt)
            projected_jumps_all.append(jump)
            projected_speeds_all.append(speed)

        previous_tile = str(previous.tile_id)
        current_tile = str(current.tile_id)

        if (
            previous_tile not in tile_centers
            or current_tile not in tile_centers
        ):
            raise RuntimeError(
                "Accepted anchor tile missing from tile-center index."
            )

        a = tile_centers[previous_tile]
        b = tile_centers[current_tile]

        tile_jump = float(
            math.hypot(
                b[0] - a[0],
                b[1] - a[1],
            )
        )
        tile_center_jumps_all.append(
            tile_jump
        )

        contiguous = (
            dt > 0
            and dt <= contiguous_max_dt
        )

        if contiguous:
            same_tile = (
                previous_tile
                == current_tile
            )
            same_tile_flags_contiguous.append(
                same_tile
            )
            tile_center_jumps_contiguous.append(
                tile_jump
            )

            if jump is not None:
                projected_jumps_contiguous.append(
                    jump
                )
                projected_speeds_contiguous.append(
                    speed
                )

                if same_tile:
                    same_tile_projected_jumps.append(
                        jump
                    )
                else:
                    changed_tile_projected_jumps.append(
                        jump
                    )
                    changed_tile_center_jumps.append(
                        tile_jump
                    )

    accepted_series = merged[
        "accepted_rg1_5"
    ].astype(bool)

    return {
        "query_count": int(
            len(
                merged
            )
        ),
        "accepted_query_count": int(
            accepted_series.sum()
        ),
        "accepted_query_fraction": float(
            accepted_series.mean()
        ),
        "accepted_projection_inside_rate": (
            accepted_inside_rate
        ),
        "nominal_sampling_interval_s": (
            nominal_dt
        ),
        "contiguous_pair_max_dt_s": (
            contiguous_max_dt
        ),
        "accepted_query_gap": numeric_summary(
            accepted_query_gaps
        ),
        "accepted_time_gap_s": numeric_summary(
            accepted_gaps_s
        ),
        "accepted_run_length_queries": numeric_summary(
            contiguous_run_lengths(
                accepted_series
            )
        ),
        "all_successive_accepted_pairs": {
            "projected_jump_m": numeric_summary(
                projected_jumps_all
            ),
            "projected_speed_mps": numeric_summary(
                projected_speeds_all
            ),
            "tile_center_jump_m": numeric_summary(
                tile_center_jumps_all
            ),
        },
        "contiguous_accepted_pairs": {
            "count": int(
                len(
                    same_tile_flags_contiguous
                )
            ),
            "same_tile_count": int(
                sum(
                    same_tile_flags_contiguous
                )
            ),
            "changed_tile_count": int(
                len(
                    same_tile_flags_contiguous
                )
                - sum(
                    same_tile_flags_contiguous
                )
            ),
            "same_tile_fraction": (
                float(
                    np.mean(
                        same_tile_flags_contiguous
                    )
                )
                if len(
                    same_tile_flags_contiguous
                )
                else None
            ),
            "projected_jump_m": numeric_summary(
                projected_jumps_contiguous
            ),
            "projected_speed_mps": numeric_summary(
                projected_speeds_contiguous
            ),
            "tile_center_jump_m": numeric_summary(
                tile_center_jumps_contiguous
            ),
            "same_tile_projected_jump_m": numeric_summary(
                same_tile_projected_jumps
            ),
            "changed_tile_projected_jump_m": numeric_summary(
                changed_tile_projected_jumps
            ),
            "changed_tile_center_jump_m": numeric_summary(
                changed_tile_center_jumps
            ),
        },
    }


def delta_block(
    development: dict[str, Any],
    blind: dict[str, Any],
    path: list[str],
) -> dict[str, Any]:
    a: Any = development
    b: Any = blind

    for key in path:
        a = a[key]
        b = b[key]

    if a is None or b is None:
        return {
            "development": a,
            "blind_stress": b,
            "delta_blind_minus_development": None,
        }

    return {
        "development": float(a),
        "blind_stress": float(b),
        "delta_blind_minus_development": float(
            b - a
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RG1.5: compare temporal behavior of the same frozen retrieval-anchor "
            "trust/refinement gate across development and no-GT blind stress."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(
        resolve(
            args.config
        ).read_text(
            encoding="utf-8"
        )
    )

    rg = cfg[
        "rg1_5"
    ]

    gate_path = resolve(
        rg[
            "frozen_gate"
        ][
            "artifact"
        ]
    )

    frozen_gate = load_gate(
        gate_path,
        expected_family=str(
            rg[
                "frozen_gate"
            ][
                "expected_family"
            ]
        ),
        expected_threshold=int(
            rg[
                "frozen_gate"
            ][
                "expected_inlier_threshold"
            ]
        ),
    )

    threshold = int(
        frozen_gate[
            "inlier_threshold"
        ]
    )

    tile_centers = load_tile_centers(
        resolve(
            rg[
                "map"
            ][
                "tile_index"
            ]
        )
    )

    dev_cfg = rg[
        "development"
    ]
    blind_cfg = rg[
        "blind_stress"
    ]

    dev_manifest = load_manifest(
        resolve(
            dev_cfg[
                "canonical_manifest"
            ]
        ),
        expected_queries=int(
            dev_cfg[
                "expected_queries"
            ]
        ),
    )
    blind_manifest = load_manifest(
        resolve(
            blind_cfg[
                "canonical_manifest"
            ]
        ),
        expected_queries=int(
            blind_cfg[
                "expected_queries"
            ]
        ),
    )

    dev_anchor = load_anchor_evidence(
        path=resolve(
            dev_cfg[
                "geometry_evidence"
            ]
        ),
        expected_queries=int(
            dev_cfg[
                "expected_queries"
            ]
        ),
        threshold=threshold,
        accepted_column_optional=True,
    )
    blind_anchor = load_anchor_evidence(
        path=resolve(
            blind_cfg[
                "geometry_evidence"
            ]
        ),
        expected_queries=int(
            blind_cfg[
                "expected_queries"
            ]
        ),
        threshold=threshold,
        accepted_column_optional=False,
    )

    multiplier = float(
        rg[
            "temporal"
        ][
            "contiguous_gap_multiplier"
        ]
    )

    development = trajectory_behavior(
        manifest=dev_manifest,
        anchor=dev_anchor,
        tile_centers=tile_centers,
        contiguous_gap_multiplier=multiplier,
    )
    blind = trajectory_behavior(
        manifest=blind_manifest,
        anchor=blind_anchor,
        tile_centers=tile_centers,
        contiguous_gap_multiplier=multiplier,
    )

    comparison = {
        "accepted_query_fraction": delta_block(
            development,
            blind,
            [
                "accepted_query_fraction",
            ],
        ),
        "accepted_inside_rate": delta_block(
            development,
            blind,
            [
                "accepted_projection_inside_rate",
            ],
        ),
        "accepted_gap_median_s": delta_block(
            development,
            blind,
            [
                "accepted_time_gap_s",
                "median",
            ],
        ),
        "accepted_run_length_median": delta_block(
            development,
            blind,
            [
                "accepted_run_length_queries",
                "median",
            ],
        ),
        "all_accepted_projection_jump_median_m": delta_block(
            development,
            blind,
            [
                "all_successive_accepted_pairs",
                "projected_jump_m",
                "median",
            ],
        ),
        "contiguous_projection_jump_median_m": delta_block(
            development,
            blind,
            [
                "contiguous_accepted_pairs",
                "projected_jump_m",
                "median",
            ],
        ),
        "contiguous_projection_speed_median_mps": delta_block(
            development,
            blind,
            [
                "contiguous_accepted_pairs",
                "projected_speed_mps",
                "median",
            ],
        ),
        "contiguous_same_tile_fraction": delta_block(
            development,
            blind,
            [
                "contiguous_accepted_pairs",
                "same_tile_fraction",
            ],
        ),
        "contiguous_changed_tile_center_jump_median_m": delta_block(
            development,
            blind,
            [
                "contiguous_accepted_pairs",
                "changed_tile_center_jump_m",
                "median",
            ],
        ),
        "contiguous_changed_tile_projected_jump_median_m": delta_block(
            development,
            blind,
            [
                "contiguous_accepted_pairs",
                "changed_tile_projected_jump_m",
                "median",
            ],
        ),
    }

    stride = float(
        rg[
            "map"
        ][
            "expected_stride_m"
        ]
    )
    diagonal_stride = float(
        stride
        * math.sqrt(
            2.0
        )
    )

    blind_all_jump_median = blind[
        "all_successive_accepted_pairs"
    ][
        "projected_jump_m"
    ][
        "median"
    ]

    grid_relation = {
        "map_stride_m": stride,
        "diagonal_stride_m": diagonal_stride,
        "blind_all_accepted_projected_jump_median_m": (
            blind_all_jump_median
        ),
        "difference_from_diagonal_stride_m": (
            float(
                blind_all_jump_median
                - diagonal_stride
            )
            if blind_all_jump_median
            is not None
            else None
        ),
        "interpretation": (
            "Numerical proximity alone does not prove the jump is a grid transition; "
            "use same-tile/changed-tile and tile-center-jump diagnostics."
        ),
    }

    report = {
        "stage": "RG1.5",
        "status": "PASS_RG1_5_CROSS_TRAJECTORY_TEMPORAL_BEHAVIOR_AUDIT",
        "created_at_utc": now_utc(),
        "scientific_question": (
            "Does the frozen inliers-only anchor trust/refinement mechanism show "
            "coherent temporal projection behavior on the blind flight, and is the "
            "large accepted projection-jump statistic explained by abstention gaps "
            "or map-anchor tile transitions?"
        ),
        "frozen_gate": {
            "artifact": str(
                gate_path
            ),
            "artifact_sha256": sha256_file(
                gate_path
            ),
            "family": frozen_gate[
                "family"
            ],
            "inlier_threshold": threshold,
            "retuned": False,
        },
        "development": development,
        "blind_stress": blind,
        "cross_trajectory_comparison": comparison,
        "map_grid_relation": grid_relation,
        "scope_guarantees": {
            "reference_used": False,
            "gate_retuned": False,
            "candidate_switching_enabled": False,
            "top20_reranked": False,
            "bootstrap_enabled": False,
            "state_enabled": False,
            "temporal_fusion_enabled": False,
        },
        "promotion": rg[
            "promotion"
        ],
        "next_decision": (
            "Use this audit to decide whether RG1 can be closed with geometry "
            "retained as a provisional trust/refinement hypothesis, or whether "
            "projection continuity requires a dedicated downstream state/temporal "
            "stage. Do not retune the geometry gate from blind behavior."
        ),
    }

    output_root = resolve(
        rg[
            "output_root"
        ]
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        output_root
        / "rg1_5_cross_trajectory_temporal_behavior_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("=" * 116)
    print(
        "RG1.5 — CROSS-TRAJECTORY ANCHOR-PROJECTION TEMPORAL BEHAVIOR AUDIT"
    )
    print("=" * 116)
    print(
        "frozen gate:",
        f"inliers >= {threshold}",
    )
    print()

    for name, block in (
        (
            "development",
            development,
        ),
        (
            "blind stress",
            blind,
        ),
    ):
        print(
            f"{name}:"
        )
        print(
            "  accepted fraction:",
            block[
                "accepted_query_fraction"
            ],
        )
        print(
            "  accepted gap median s:",
            block[
                "accepted_time_gap_s"
            ][
                "median"
            ],
        )
        print(
            "  accepted run median queries:",
            block[
                "accepted_run_length_queries"
            ][
                "median"
            ],
        )
        print(
            "  all accepted jump median m:",
            block[
                "all_successive_accepted_pairs"
            ][
                "projected_jump_m"
            ][
                "median"
            ],
        )
        print(
            "  contiguous accepted pairs:",
            block[
                "contiguous_accepted_pairs"
            ][
                "count"
            ],
        )
        print(
            "  contiguous jump median m:",
            block[
                "contiguous_accepted_pairs"
            ][
                "projected_jump_m"
            ][
                "median"
            ],
        )
        print(
            "  contiguous speed median m/s:",
            block[
                "contiguous_accepted_pairs"
            ][
                "projected_speed_mps"
            ][
                "median"
            ],
        )
        print(
            "  contiguous same-tile fraction:",
            block[
                "contiguous_accepted_pairs"
            ][
                "same_tile_fraction"
            ],
        )
        print(
            "  changed-tile center-jump median m:",
            block[
                "contiguous_accepted_pairs"
            ][
                "changed_tile_center_jump_m"
            ][
                "median"
            ],
        )
        print(
            "  changed-tile projected-jump median m:",
            block[
                "contiguous_accepted_pairs"
            ][
                "changed_tile_projected_jump_m"
            ][
                "median"
            ],
        )
        print()

    print(
        "map stride m:",
        grid_relation[
            "map_stride_m"
        ],
    )
    print(
        "map diagonal stride m:",
        grid_relation[
            "diagonal_stride_m"
        ],
    )
    print(
        "blind all-accepted jump median minus diagonal stride m:",
        grid_relation[
            "difference_from_diagonal_stride_m"
        ],
    )
    print()
    print(
        "reference used:",
        False,
    )
    print(
        "gate retuned:",
        False,
    )
    print()
    print(
        "STATUS: PASS_RG1_5_CROSS_TRAJECTORY_TEMPORAL_BEHAVIOR_AUDIT"
    )
    print(
        "report:",
        report_path,
    )


if __name__ == "__main__":
    main()
