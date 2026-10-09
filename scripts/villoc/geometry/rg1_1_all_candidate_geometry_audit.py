#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uavloc.data.trajectory_adapter import (  # noqa: E402
    FORBIDDEN_BLIND_REFERENCE_COLUMNS,
    load_trajectory_spec,
)
from scripts.villoc.research.minimum_confident_bootstrap.diagnostics import (  # noqa: E402
    r4_11_blind_subtile_projection_recompute as r411,
)
from scripts.villoc.retrieval.qv1_3_fixed_budget_multiview_pool import (  # noqa: E402
    load_reference_xy,
)


ROOT = REPO_ROOT
DEFAULT_CONFIG = Path(
    "configs/research/region_candidate_geometry_v1.yaml"
)


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assert_blind_safe(frame: pd.DataFrame, *, label: str) -> None:
    lower = {
        str(column).strip().lower()
        for column in frame.columns
    }
    forbidden = {
        str(column).strip().lower()
        for column in FORBIDDEN_BLIND_REFERENCE_COLUMNS
    }
    exact = sorted(lower & forbidden)
    tokens = (
        "ground_truth",
        "oracle",
        "chosen_error",
        "projected_error",
        "tile_center_error",
        "eval_ref_",
        "reference_x",
        "reference_y",
    )
    partial = sorted(
        column
        for column in lower
        if any(token in column for token in tokens)
    )
    leaked = sorted(set(exact) | set(partial))
    if leaked:
        raise RuntimeError(
            f"{label} contains forbidden reference/evaluation columns: {leaked}"
        )


def resolve_existing_image(
    value: Any,
    *,
    extra_roots: list[Path] | None = None,
) -> Path:
    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        raise FileNotFoundError(f"Invalid image path: {value!r}")

    path = Path(raw).expanduser()
    candidates: list[Path] = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(ROOT / path)
        for root in extra_roots or []:
            candidates.append(root / path)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not resolve image path. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def binary_auc_star(
    labels: pd.Series,
    scores: pd.Series,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            "label": labels.astype(bool),
            "score": pd.to_numeric(scores, errors="coerce"),
        }
    ).dropna()

    positives = int(frame["label"].sum())
    negatives = int((~frame["label"]).sum())

    if positives == 0 or negatives == 0:
        return {
            "count": int(len(frame)),
            "positives": positives,
            "negatives": negatives,
            "auc": None,
            "auc_star": None,
            "direction": None,
        }

    ranks = frame["score"].rank(
        method="average",
        ascending=True,
    )
    positive_rank_sum = float(
        ranks[frame["label"]].sum()
    )

    auc = (
        positive_rank_sum
        - positives * (positives + 1) / 2.0
    ) / float(positives * negatives)

    if auc >= 0.5:
        direction = "higher_favors_positive"
        auc_star = auc
    else:
        direction = "lower_favors_positive"
        auc_star = 1.0 - auc

    return {
        "count": int(len(frame)),
        "positives": positives,
        "negatives": negatives,
        "auc": float(auc),
        "auc_star": float(auc_star),
        "direction": direction,
    }


def numeric_summary(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if not len(values):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }

    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def rate_block(mask: pd.Series, valid: pd.Series) -> dict[str, Any]:
    mask = mask.astype(bool)
    valid = valid.astype(bool)
    count = int(mask.sum())
    hits = int((mask & valid).sum())
    return {
        "count": count,
        "valid_projection_count": hits,
        "valid_projection_rate": (
            float(hits / count)
            if count
            else None
        ),
    }


def verify_preflight(
    *,
    report_path: Path,
    manifest_path: Path,
    pool_path: Path,
) -> dict[str, Any]:
    if not report_path.exists():
        raise FileNotFoundError(report_path)

    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    if report.get("status") != "PASS_RG1_GEOMETRY_CONTRACT_PREFLIGHT":
        raise RuntimeError(
            "RG1.0 preflight has not passed."
        )

    manifest_sha = report[
        "trajectory"
    ][
        "canonical_manifest_sha256"
    ]
    pool_sha = report[
        "candidate_pool"
    ][
        "sha256"
    ]

    current_manifest_sha = sha256_file(
        manifest_path
    )
    current_pool_sha = sha256_file(
        pool_path
    )

    if current_manifest_sha != manifest_sha:
        raise RuntimeError(
            "Canonical manifest changed after RG1.0 preflight."
        )

    if current_pool_sha != pool_sha:
        raise RuntimeError(
            "Frozen candidate pool changed after RG1.0 preflight."
        )

    return report


def compute_blind_evidence(
    *,
    manifest: pd.DataFrame,
    pool: pd.DataFrame,
    tile_index: pd.DataFrame,
    tile_index_path: Path,
    progress_every: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_lookup = (
        manifest
        .set_index("query_id")
    )
    tile_lookup = (
        tile_index
        .set_index("tile_id")
    )

    detector = r411.create_detector()

    query_feature_cache: dict[int, Any] = {}
    tile_feature_cache: dict[str, Any] = {}

    def query_features(qid: int):
        if qid not in query_feature_cache:
            row = manifest_lookup.loc[qid]
            path = resolve_existing_image(
                row["image_path"]
            )
            query_feature_cache[qid] = (
                path,
                r411.compute_features(
                    path,
                    detector,
                ),
            )
        return query_feature_cache[qid]

    def tile_features(tile_id: str):
        if tile_id not in tile_feature_cache:
            row = tile_lookup.loc[tile_id]
            path = resolve_existing_image(
                row["tile_path"],
                extra_roots=[
                    tile_index_path.parent,
                ],
            )
            tile_feature_cache[tile_id] = (
                path,
                r411.compute_features(
                    path,
                    detector,
                ),
            )
        return tile_feature_cache[tile_id]

    rows: list[dict[str, Any]] = []
    ordered = pool.sort_values(
        ["query_id", "pool_rank"],
        kind="mergesort",
    ).reset_index(drop=True)

    started = time.perf_counter()

    for idx, row in ordered.iterrows():
        if (
            idx == 0
            or (idx + 1) % max(1, progress_every) == 0
            or idx + 1 == len(ordered)
        ):
            print(
                f"geometry pair {idx + 1}/{len(ordered)}",
                flush=True,
            )

        qid = int(row["query_id"])
        tile_id = str(row["tile_id"])

        q_path, qfeat = query_features(qid)
        t_path, tfeat = tile_features(tile_id)
        tile = tile_lookup.loc[tile_id]

        result = r411.verify_pair_with_geometry(
            qfeat,
            tfeat,
        )

        projected_easting = math.nan
        projected_northing = math.nan
        projected_inside = False

        if bool(result["homography_ok"]):
            u = float(
                result[
                    "projected_tile_u_px"
                ]
            )
            v = float(
                result[
                    "projected_tile_v_px"
                ]
            )

            projected_easting, projected_northing = (
                r411.tile_pixel_to_map(
                    u,
                    v,
                    tfeat.image_shape,
                    float(
                        tile[
                            "left_easting"
                        ]
                    ),
                    float(
                        tile[
                            "right_easting"
                        ]
                    ),
                    float(
                        tile[
                            "bottom_northing"
                        ]
                    ),
                    float(
                        tile[
                            "top_northing"
                        ]
                    ),
                )
            )

            th, tw = tfeat.image_shape
            projected_inside = bool(
                0.0 <= u < float(tw)
                and 0.0 <= v < float(th)
            )

        rows.append(
            {
                "query_id": qid,
                "pool_rank": int(
                    row[
                        "pool_rank"
                    ]
                ),
                "tile_id": tile_id,
                "source_view": str(
                    row[
                        "source_view"
                    ]
                ),
                "source_rank": int(
                    row[
                        "source_rank"
                    ]
                ),
                "selection_reason": str(
                    row[
                        "selection_reason"
                    ]
                ),
                "query_image_resolved": str(
                    q_path
                ),
                "tile_image_resolved": str(
                    t_path
                ),
                "good_matches": int(
                    result[
                        "good_matches"
                    ]
                ),
                "inliers": int(
                    result[
                        "inliers"
                    ]
                ),
                "inlier_ratio": float(
                    result[
                        "inlier_ratio"
                    ]
                ),
                "query_inlier_coverage": float(
                    result[
                        "query_inlier_coverage"
                    ]
                ),
                "tile_inlier_coverage": float(
                    result[
                        "sat_inlier_coverage"
                    ]
                ),
                "homography_ok": bool(
                    result[
                        "homography_ok"
                    ]
                ),
                "query_center_u_px": float(
                    result[
                        "query_center_u_px"
                    ]
                ),
                "query_center_v_px": float(
                    result[
                        "query_center_v_px"
                    ]
                ),
                "projected_tile_u_px": float(
                    result[
                        "projected_tile_u_px"
                    ]
                ),
                "projected_tile_v_px": float(
                    result[
                        "projected_tile_v_px"
                    ]
                ),
                "projected_easting": float(
                    projected_easting
                ),
                "projected_northing": float(
                    projected_northing
                ),
                "projected_inside_tile": bool(
                    projected_inside
                ),
                "projective_denominator": float(
                    result[
                        "projective_denominator"
                    ]
                ),
                "inlier_reprojection_rmse_px": float(
                    result[
                        "inlier_reprojection_rmse_px"
                    ]
                ),
                "geometry_error": (
                    result[
                        "error"
                    ]
                    if result[
                        "error"
                    ]
                    else ""
                ),
            }
        )

    elapsed = float(
        time.perf_counter()
        - started
    )

    evidence = pd.DataFrame(
        rows
    )

    summary = {
        "candidate_pairs": int(
            len(
                evidence
            )
        ),
        "queries": int(
            evidence[
                "query_id"
            ].nunique()
        ),
        "unique_tiles": int(
            evidence[
                "tile_id"
            ].nunique()
        ),
        "homography_success_count": int(
            evidence[
                "homography_ok"
            ].astype(bool).sum()
        ),
        "homography_success_rate": float(
            evidence[
                "homography_ok"
            ].astype(bool).mean()
        ),
        "inside_tile_projection_count": int(
            evidence[
                "projected_inside_tile"
            ].astype(bool).sum()
        ),
        "inliers": numeric_summary(
            evidence[
                "inliers"
            ]
        ),
        "inlier_ratio": numeric_summary(
            evidence[
                "inlier_ratio"
            ]
        ),
        "query_inlier_coverage": numeric_summary(
            evidence[
                "query_inlier_coverage"
            ]
        ),
        "tile_inlier_coverage": numeric_summary(
            evidence[
                "tile_inlier_coverage"
            ]
        ),
        "runtime_s": elapsed,
        "ms_per_pair": float(
            1000.0
            * elapsed
            / max(
                1,
                len(
                    evidence
                ),
            )
        ),
        "feature_cache": {
            "query_images": int(
                len(
                    query_feature_cache
                )
            ),
            "tile_images": int(
                len(
                    tile_feature_cache
                )
            ),
        },
    }

    return evidence, summary


def attach_postfreeze_reference(
    *,
    evidence: pd.DataFrame,
    reference_path: Path,
    tile_index: pd.DataFrame,
) -> pd.DataFrame:
    reference = load_reference_xy(
        reference_path
    )

    tile = (
        tile_index[
            [
                "tile_id",
                "left_easting",
                "right_easting",
                "bottom_northing",
                "top_northing",
                "center_easting",
                "center_northing",
            ]
        ]
        .copy()
    )
    tile[
        "tile_id"
    ] = tile[
        "tile_id"
    ].astype(str)

    merged = evidence.merge(
        tile,
        on="tile_id",
        how="left",
        validate="many_to_one",
    )

    ref = (
        reference[
            [
                "gt_x",
                "gt_y",
            ]
        ]
        .reset_index()
    )

    merged = merged.merge(
        ref,
        on="query_id",
        how="left",
        validate="many_to_one",
    )

    if merged[
        [
            "gt_x",
            "gt_y",
        ]
    ].isna().any().any():
        raise RuntimeError(
            "Reference attachment is missing query coordinates for RG1.1."
        )

    dx = (
        merged[
            "center_easting"
        ].astype(float)
        - merged[
            "gt_x"
        ].astype(float)
    )
    dy = (
        merged[
            "center_northing"
        ].astype(float)
        - merged[
            "gt_y"
        ].astype(float)
    )

    merged[
        "center_error_m"
    ] = np.hypot(
        dx,
        dy,
    )

    merged[
        "contains_query"
    ] = (
        (
            merged[
                "left_easting"
            ].astype(float)
            <= merged[
                "gt_x"
            ].astype(float)
        )
        &
        (
            merged[
                "gt_x"
            ].astype(float)
            <= merged[
                "right_easting"
            ].astype(float)
        )
        &
        (
            merged[
                "bottom_northing"
            ].astype(float)
            <= merged[
                "gt_y"
            ].astype(float)
        )
        &
        (
            merged[
                "gt_y"
            ].astype(float)
            <= merged[
                "top_northing"
            ].astype(float)
        )
    )

    merged[
        "le40"
    ] = (
        merged[
            "center_error_m"
        ]
        <= 40.0
    )
    merged[
        "le80"
    ] = (
        merged[
            "center_error_m"
        ]
        <= 80.0
    )

    valid_projection = (
        merged[
            "homography_ok"
        ].astype(bool)
        & np.isfinite(
            pd.to_numeric(
                merged[
                    "projected_easting"
                ],
                errors="coerce",
            )
        )
        & np.isfinite(
            pd.to_numeric(
                merged[
                    "projected_northing"
                ],
                errors="coerce",
            )
        )
    )

    merged[
        "valid_projection"
    ] = valid_projection

    merged[
        "projected_error_m"
    ] = math.nan

    px = (
        pd.to_numeric(
            merged.loc[
                valid_projection,
                "projected_easting",
            ],
            errors="coerce",
        )
        - merged.loc[
            valid_projection,
            "gt_x",
        ].astype(float)
    )
    py = (
        pd.to_numeric(
            merged.loc[
                valid_projection,
                "projected_northing",
            ],
            errors="coerce",
        )
        - merged.loc[
            valid_projection,
            "gt_y",
        ].astype(float)
    )

    merged.loc[
        valid_projection,
        "projected_error_m",
    ] = np.hypot(
        px,
        py,
    )

    merged[
        "projection_improves_center"
    ] = False

    merged.loc[
        valid_projection,
        "projection_improves_center",
    ] = (
        merged.loc[
            valid_projection,
            "projected_error_m",
        ].astype(float)
        <
        merged.loc[
            valid_projection,
            "center_error_m",
        ].astype(float)
    )

    return merged


def geometry_diagnostics(
    evaluation: pd.DataFrame,
) -> dict[str, Any]:
    features = [
        "good_matches",
        "inliers",
        "inlier_ratio",
        "query_inlier_coverage",
        "tile_inlier_coverage",
        "homography_ok",
        "projected_inside_tile",
    ]

    auc = {}

    for label in (
        "contains_query",
        "le80",
        "le40",
    ):
        auc[label] = {
            feature: binary_auc_star(
                evaluation[
                    label
                ],
                evaluation[
                    feature
                ].astype(float)
                if feature in {
                    "homography_ok",
                    "projected_inside_tile",
                }
                else evaluation[
                    feature
                ],
            )
            for feature in features
        }

    valid = evaluation[
        "valid_projection"
    ].astype(bool)

    valid_rate = {
        "all": {
            "count": int(
                len(
                    evaluation
                )
            ),
            "valid_projection_count": int(
                valid.sum()
            ),
            "valid_projection_rate": float(
                valid.mean()
            ),
        },
        "contains_true": rate_block(
            evaluation[
                "contains_query"
            ],
            valid,
        ),
        "contains_false": rate_block(
            ~evaluation[
                "contains_query"
            ].astype(bool),
            valid,
        ),
        "le80_true": rate_block(
            evaluation[
                "le80"
            ],
            valid,
        ),
        "le80_false": rate_block(
            ~evaluation[
                "le80"
            ].astype(bool),
            valid,
        ),
    }

    valid_rows = evaluation[
        valid
    ].copy()

    projection = {
        "valid_all": {
            "center_error_m": numeric_summary(
                valid_rows[
                    "center_error_m"
                ]
            ),
            "projected_error_m": numeric_summary(
                valid_rows[
                    "projected_error_m"
                ]
            ),
            "projection_improves_count": int(
                valid_rows[
                    "projection_improves_center"
                ].astype(bool).sum()
            ),
            "projection_improves_rate": (
                float(
                    valid_rows[
                        "projection_improves_center"
                    ].astype(bool).mean()
                )
                if len(
                    valid_rows
                )
                else None
            ),
        },
    }

    for label in (
        "contains_query",
        "le80",
        "le40",
    ):
        subset = valid_rows[
            valid_rows[
                label
            ].astype(bool)
        ]
        projection[
            label
        ] = {
            "count": int(
                len(
                    subset
                )
            ),
            "center_error_m": numeric_summary(
                subset[
                    "center_error_m"
                ]
            ),
            "projected_error_m": numeric_summary(
                subset[
                    "projected_error_m"
                ]
            ),
            "projection_improves_count": int(
                subset[
                    "projection_improves_center"
                ].astype(bool).sum()
            ),
            "projection_improves_rate": (
                float(
                    subset[
                        "projection_improves_center"
                    ].astype(bool).mean()
                )
                if len(
                    subset
                )
                else None
            ),
        }

    source_stats = {}

    for source, group in evaluation.groupby(
        "source_view"
    ):
        source_stats[
            str(
                source
            )
        ] = {
            "candidate_count": int(
                len(
                    group
                )
            ),
            "contains_count": int(
                group[
                    "contains_query"
                ].astype(bool).sum()
            ),
            "le80_count": int(
                group[
                    "le80"
                ].astype(bool).sum()
            ),
            "homography_success_count": int(
                group[
                    "homography_ok"
                ].astype(bool).sum()
            ),
            "homography_success_rate": float(
                group[
                    "homography_ok"
                ].astype(bool).mean()
            ),
            "inliers_median": float(
                pd.to_numeric(
                    group[
                        "inliers"
                    ],
                    errors="coerce",
                ).median()
            ),
        }

    return {
        "geometry_feature_auc_star": auc,
        "valid_projection_rate_by_candidate_class": valid_rate,
        "projection_quality": projection,
        "candidate_source_diagnostics": source_stats,
    }


def named_case_diagnostics(
    evaluation: pd.DataFrame,
    query_ids: list[int],
) -> dict[str, Any]:
    output = {}

    for qid in query_ids:
        group = (
            evaluation[
                evaluation[
                    "query_id"
                ].astype(int)
                == int(
                    qid
                )
            ]
            .sort_values(
                "pool_rank"
            )
            .copy()
        )

        if group.empty:
            output[
                f"q{qid}"
            ] = None
            continue

        strongest = (
            group.sort_values(
                [
                    "inliers",
                    "inlier_ratio",
                    "query_inlier_coverage",
                    "pool_rank",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    True,
                ],
                kind="mergesort",
            )
            .iloc[
                0
            ]
        )

        best_center = (
            group.sort_values(
                [
                    "center_error_m",
                    "pool_rank",
                ],
                kind="mergesort",
            )
            .iloc[
                0
            ]
        )

        valid = group[
            group[
                "valid_projection"
            ].astype(bool)
        ]

        best_projection = (
            valid.sort_values(
                [
                    "projected_error_m",
                    "pool_rank",
                ],
                kind="mergesort",
            )
            .iloc[
                0
            ]
            if len(
                valid
            )
            else None
        )

        rows = []

        for row in group.itertuples(
            index=False
        ):
            rows.append(
                {
                    "pool_rank": int(
                        row.pool_rank
                    ),
                    "tile_id": str(
                        row.tile_id
                    ),
                    "source_view": str(
                        row.source_view
                    ),
                    "source_rank": int(
                        row.source_rank
                    ),
                    "contains_query": bool(
                        row.contains_query
                    ),
                    "le40": bool(
                        row.le40
                    ),
                    "le80": bool(
                        row.le80
                    ),
                    "center_error_m": float(
                        row.center_error_m
                    ),
                    "good_matches": int(
                        row.good_matches
                    ),
                    "inliers": int(
                        row.inliers
                    ),
                    "inlier_ratio": float(
                        row.inlier_ratio
                    ),
                    "query_inlier_coverage": float(
                        row.query_inlier_coverage
                    ),
                    "tile_inlier_coverage": float(
                        row.tile_inlier_coverage
                    ),
                    "homography_ok": bool(
                        row.homography_ok
                    ),
                    "projected_inside_tile": bool(
                        row.projected_inside_tile
                    ),
                    "projected_error_m": (
                        float(
                            row.projected_error_m
                        )
                        if math.isfinite(
                            float(
                                row.projected_error_m
                            )
                        )
                        else None
                    ),
                }
            )

        output[
            f"q{qid}"
        ] = {
            "diagnostic_only": True,
            "pool_rank1": rows[
                0
            ],
            "strongest_geometry_evidence": {
                "pool_rank": int(
                    strongest[
                        "pool_rank"
                    ]
                ),
                "tile_id": str(
                    strongest[
                        "tile_id"
                    ]
                ),
                "inliers": int(
                    strongest[
                        "inliers"
                    ]
                ),
                "center_error_m": float(
                    strongest[
                        "center_error_m"
                    ]
                ),
                "contains_query": bool(
                    strongest[
                        "contains_query"
                    ]
                ),
            },
            "best_center_error_candidate": {
                "pool_rank": int(
                    best_center[
                        "pool_rank"
                    ]
                ),
                "tile_id": str(
                    best_center[
                        "tile_id"
                    ]
                ),
                "center_error_m": float(
                    best_center[
                        "center_error_m"
                    ]
                ),
                "inliers": int(
                    best_center[
                        "inliers"
                    ]
                ),
            },
            "best_valid_projection": (
                {
                    "pool_rank": int(
                        best_projection[
                            "pool_rank"
                        ]
                    ),
                    "tile_id": str(
                        best_projection[
                            "tile_id"
                        ]
                    ),
                    "projected_error_m": float(
                        best_projection[
                            "projected_error_m"
                        ]
                    ),
                    "center_error_m": float(
                        best_projection[
                            "center_error_m"
                        ]
                    ),
                }
                if best_projection is not None
                else None
            ),
            "all_candidates": rows,
        }

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RG1.1: compute frozen per-candidate ORB/homography evidence "
            "for all QV1.4 Top20 candidates, then attach reference only "
            "after the blind evidence table is frozen."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
    )
    args = parser.parse_args()

    config_path = resolve(
        args.config
    )
    cfg = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    dev = cfg[
        "development_trajectory"
    ]
    rg = cfg[
        "rg1_1"
    ]

    spec_path = resolve(
        dev[
            "spec"
        ]
    )
    manifest_path = resolve(
        dev[
            "canonical_manifest"
        ]
    )
    pool_path = resolve(
        dev[
            "frozen_candidate_pool"
        ]
    )
    preflight_path = resolve(
        rg[
            "preflight_report"
        ]
    )
    output_root = resolve(
        rg[
            "output_root"
        ]
    )

    preflight = verify_preflight(
        report_path=preflight_path,
        manifest_path=manifest_path,
        pool_path=pool_path,
    )

    spec = load_trajectory_spec(
        spec_path
    )

    map_variant = str(
        dev[
            "map_variant"
        ]
    )
    variant = spec.raw[
        "map"
    ][
        "variants"
    ][
        map_variant
    ]
    tile_index_path = resolve(
        variant[
            "tile_index"
        ]
    )

    manifest = pd.read_csv(
        manifest_path
    ).copy()
    pool = pd.read_csv(
        pool_path
    ).copy()
    tile_index = pd.read_csv(
        tile_index_path
    ).copy()

    assert_blind_safe(
        manifest,
        label="canonical manifest",
    )
    assert_blind_safe(
        pool,
        label="frozen candidate pool",
    )

    manifest[
        "query_id"
    ] = pd.to_numeric(
        manifest[
            "query_id"
        ],
        errors="raise",
    ).astype(int)
    pool[
        "query_id"
    ] = pd.to_numeric(
        pool[
            "query_id"
        ],
        errors="raise",
    ).astype(int)
    pool[
        "pool_rank"
    ] = pd.to_numeric(
        pool[
            "pool_rank"
        ],
        errors="raise",
    ).astype(int)
    pool[
        "source_rank"
    ] = pd.to_numeric(
        pool[
            "source_rank"
        ],
        errors="raise",
    ).astype(int)
    pool[
        "tile_id"
    ] = pool[
        "tile_id"
    ].astype(str)
    tile_index[
        "tile_id"
    ] = tile_index[
        "tile_id"
    ].astype(str)

    expected_rows = (
        int(
            dev[
                "expected_queries"
            ]
        )
        * int(
            dev[
                "expected_candidate_budget"
            ]
        )
    )

    if len(
        pool
    ) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} candidate rows, got {len(pool)}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    blind_root = (
        output_root
        / "blind_geometry"
    )
    post_root = (
        output_root
        / "postfreeze_evaluation"
    )
    blind_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    post_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 112)
    print(
        "RG1.1 — ALL-CANDIDATE GEOMETRY EVIDENCE AUDIT"
    )
    print("=" * 112)
    print(
        "queries:",
        int(
            dev[
                "expected_queries"
            ]
        ),
    )
    print(
        "candidate pairs:",
        expected_rows,
    )
    print(
        "candidate selection:",
        False,
    )
    print(
        "reference loaded before freeze:",
        False,
    )
    print()

    evidence, blind_summary = (
        compute_blind_evidence(
            manifest=manifest,
            pool=pool,
            tile_index=tile_index,
            tile_index_path=tile_index_path,
            progress_every=int(
                args.progress_every
            ),
        )
    )

    assert_blind_safe(
        evidence,
        label="RG1.1 blind geometry evidence",
    )

    if len(
        evidence
    ) != expected_rows:
        raise RuntimeError(
            "RG1.1 did not produce one blind evidence row per candidate pair."
        )

    blind_path = (
        blind_root
        / "rg1_1_all_candidate_geometry_evidence.csv"
    )
    blind_summary_path = (
        blind_root
        / "rg1_1_blind_geometry_summary.json"
    )

    evidence.to_csv(
        blind_path,
        index=False,
    )

    blind_summary_report = {
        "stage": "RG1.1_BLIND",
        "status": "PASS_RG1_1_BLIND_GEOMETRY_FREEZE",
        "created_at_utc": now_utc(),
        "preflight_report": str(
            preflight_path
        ),
        "preflight_status": preflight[
            "status"
        ],
        "candidate_selection_enabled": False,
        "hybrid_rank_prior_enabled": False,
        "reference_used": False,
        "bootstrap_enabled": False,
        "state_enabled": False,
        "geometry": blind_summary,
        "input_hashes": {
            "manifest": sha256_file(
                manifest_path
            ),
            "candidate_pool": sha256_file(
                pool_path
            ),
            "tile_index": sha256_file(
                tile_index_path
            ),
        },
    }

    blind_summary_path.write_text(
        json.dumps(
            blind_summary_report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    blind_hash = sha256_file(
        blind_path
    )
    blind_summary_hash = sha256_file(
        blind_summary_path
    )

    # --------------------------------------------------------
    # POST-FREEZE ONLY: reference attachment and diagnostics.
    # --------------------------------------------------------

    prepared_reference = (
        spec.reference.config.get(
            "prepared_attachment"
        )
    )

    if not prepared_reference:
        raise RuntimeError(
            "Development trajectory does not declare a prepared post-freeze "
            "reference attachment."
        )

    reference_path = resolve(
        prepared_reference
    )

    if not reference_path.exists():
        raise FileNotFoundError(
            reference_path
        )

    if sha256_file(
        blind_path
    ) != blind_hash:
        raise RuntimeError(
            "Blind geometry evidence changed before reference attachment."
        )

    evaluation = attach_postfreeze_reference(
        evidence=evidence,
        reference_path=reference_path,
        tile_index=tile_index,
    )

    diagnostics = geometry_diagnostics(
        evaluation
    )

    named = named_case_diagnostics(
        evaluation,
        [
            int(
                value
            )
            for value in cfg[
                "named_diagnostics"
            ]
        ],
    )

    evaluation_path = (
        post_root
        / "rg1_1_all_candidate_geometry_postfreeze_eval.csv"
    )
    evaluation.to_csv(
        evaluation_path,
        index=False,
    )

    if sha256_file(
        blind_path
    ) != blind_hash:
        raise RuntimeError(
            "Blind geometry evidence changed during post-freeze evaluation."
        )

    report = {
        "stage": "RG1.1",
        "status": "PASS_RG1_1_ALL_CANDIDATE_GEOMETRY_AUDIT",
        "created_at_utc": now_utc(),
        "scientific_question": (
            "Does geometry discriminate useful region candidates, and does "
            "valid homography projection improve within-region localization "
            "without using geometry as a selector?"
        ),
        "trajectory": {
            "id": spec.trajectory_id,
            "query_count": int(
                dev[
                    "expected_queries"
                ]
            ),
        },
        "candidate_pool": {
            "rows": expected_rows,
            "budget": int(
                dev[
                    "expected_candidate_budget"
                ]
            ),
            "path": str(
                pool_path
            ),
            "sha256": sha256_file(
                pool_path
            ),
        },
        "blind_freeze": {
            "evidence_path": str(
                blind_path
            ),
            "evidence_sha256": blind_hash,
            "summary_path": str(
                blind_summary_path
            ),
            "summary_sha256": blind_summary_hash,
            "reference_used": False,
        },
        "blind_geometry_summary": blind_summary,
        "postfreeze_reference": {
            "path": str(
                reference_path
            ),
            "sha256": sha256_file(
                reference_path
            ),
        },
        "diagnostics": diagnostics,
        "named_diagnostics": named,
        "scope_guarantees": {
            "all_candidate_pairs_processed": True,
            "candidate_selection_enabled": False,
            "geometry_rank_emitted": False,
            "hybrid_rank_prior_enabled": False,
            "bootstrap_enabled": False,
            "state_enabled": False,
            "temporal_fusion_enabled": False,
            "reference_used_before_blind_freeze": False,
            "blind_stress_trajectory_used_for_tuning": False,
        },
        "next_decision": (
            "Use RG1.1 evidence to decide whether a local geometry gate/refinement "
            "experiment is justified. Do not design a geometry selector from named "
            "cases alone."
        ),
    }

    report_path = (
        output_root
        / "rg1_1_all_candidate_geometry_audit_report.json"
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "blind homography success:",
        blind_summary[
            "homography_success_count"
        ],
        "/",
        blind_summary[
            "candidate_pairs"
        ],
        f"({blind_summary['homography_success_rate']:.4f})",
    )
    print(
        "blind median inliers:",
        blind_summary[
            "inliers"
        ][
            "median"
        ],
    )
    print(
        "blind median inlier ratio:",
        blind_summary[
            "inlier_ratio"
        ][
            "median"
        ],
    )

    for label in (
        "contains_query",
        "le80",
        "le40",
    ):
        ranked = sorted(
            diagnostics[
                "geometry_feature_auc_star"
            ][
                label
            ].items(),
            key=lambda kv: (
                kv[
                    1
                ][
                    "auc_star"
                ]
                if kv[
                    1
                ][
                    "auc_star"
                ]
                is not None
                else -1.0
            ),
            reverse=True,
        )
        print()
        print(
            label,
            "top geometry diagnostics:",
        )
        for feature, block in ranked[
            :5
        ]:
            print(
                f"  {feature:28s} "
                f"AUC*={block['auc_star']} "
                f"{block['direction']}"
            )

    projection = diagnostics[
        "projection_quality"
    ][
        "contains_query"
    ]

    print()
    print(
        "valid containing projections:",
        projection[
            "count"
        ],
    )
    print(
        "containing center-error median:",
        projection[
            "center_error_m"
        ][
            "median"
        ],
    )
    print(
        "containing projected-error median:",
        projection[
            "projected_error_m"
        ][
            "median"
        ],
    )
    print(
        "containing projection improvement rate:",
        projection[
            "projection_improves_rate"
        ],
    )

    print()
    print(
        "STATUS: PASS_RG1_1_ALL_CANDIDATE_GEOMETRY_AUDIT"
    )
    print(
        "report:",
        report_path,
    )


if __name__ == "__main__":
    main()
