#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from uavloc.data.trajectory_adapter import (
    FORBIDDEN_BLIND_REFERENCE_COLUMNS,
    load_trajectory_spec,
)


ROOT = Path.cwd().resolve()
DEFAULT_CONFIG = Path(
    "configs/research/region_candidate_geometry_v1.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/research_runs/region_candidate_geometry_v1/rg1_0"
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


def assert_no_reference_columns(
    frame: pd.DataFrame,
    *,
    label: str,
) -> None:
    lower = {
        str(column).strip().lower()
        for column in frame.columns
    }

    forbidden = {
        str(column).strip().lower()
        for column in FORBIDDEN_BLIND_REFERENCE_COLUMNS
    }

    leaked = sorted(
        lower & forbidden
    )

    additional_forbidden_tokens = (
        "ground_truth",
        "oracle",
        "chosen_error",
        "projected_error",
        "tile_center_error",
        "eval_ref_",
    )

    leaked_tokens = sorted(
        column
        for column in lower
        if any(
            token in column
            for token in additional_forbidden_tokens
        )
    )

    all_leaked = sorted(
        set(leaked)
        | set(leaked_tokens)
    )

    if all_leaked:
        raise RuntimeError(
            f"{label} contains reference/evaluation columns: {all_leaked}"
        )


def resolve_existing_image(
    value: Any,
    *,
    extra_roots: list[Path] | None = None,
) -> Path | None:
    if value is None:
        return None

    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return None

    path = Path(raw).expanduser()

    candidates = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(ROOT / path)

        for root in extra_roots or []:
            candidates.append(root / path)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    return None


def validate_pool_contract(
    *,
    pool: pd.DataFrame,
    manifest: pd.DataFrame,
    expected_queries: int,
    expected_budget: int,
    allowed_sources: set[str],
    require_center_top1: bool,
) -> dict[str, Any]:
    required_pool = {
        "query_id",
        "pool_rank",
        "tile_id",
        "source_view",
        "source_rank",
        "selection_reason",
    }

    required_manifest = {
        "trajectory_id",
        "query_id",
        "image_path",
        "image_width",
        "image_height",
        "reference_available",
    }

    missing_pool = sorted(
        required_pool
        - set(pool.columns)
    )
    missing_manifest = sorted(
        required_manifest
        - set(manifest.columns)
    )

    if missing_pool:
        raise RuntimeError(
            f"Candidate pool missing columns: {missing_pool}"
        )

    if missing_manifest:
        raise RuntimeError(
            f"Canonical manifest missing columns: {missing_manifest}"
        )

    qids = (
        manifest["query_id"]
        .astype(int)
        .tolist()
    )

    pool_qids = sorted(
        pool["query_id"]
        .astype(int)
        .drop_duplicates()
        .tolist()
    )

    if len(qids) != expected_queries:
        raise RuntimeError(
            f"Expected {expected_queries} manifest queries, got {len(qids)}"
        )

    if sorted(qids) != pool_qids:
        raise RuntimeError(
            "Candidate-pool query IDs do not match canonical manifest."
        )

    counts = (
        pool.groupby("query_id")
        .size()
    )

    if not (
        counts
        == expected_budget
    ).all():
        bad = counts[
            counts
            != expected_budget
        ]
        raise RuntimeError(
            "Candidate budget mismatch for queries: "
            + json.dumps(
                {
                    int(k): int(v)
                    for k, v in bad.items()
                },
                indent=2,
            )
        )

    if pool[
        ["query_id", "pool_rank"]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate (query_id, pool_rank) rows detected."
        )

    ranks = (
        pool.groupby("query_id")[
            "pool_rank"
        ]
        .apply(
            lambda s: sorted(
                s.astype(int).tolist()
            )
        )
    )

    expected_ranks = list(
        range(
            1,
            expected_budget + 1,
        )
    )

    if not ranks.apply(
        lambda values: (
            values == expected_ranks
        )
    ).all():
        raise RuntimeError(
            "Pool ranks are not exactly 1..budget for every query."
        )

    sources = set(
        pool[
            "source_view"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    unexpected = sorted(
        sources
        - allowed_sources
    )

    if unexpected:
        raise RuntimeError(
            f"Unexpected candidate source views: {unexpected}"
        )

    if require_center_top1:
        top1 = pool[
            pool[
                "pool_rank"
            ].astype(int)
            == 1
        ]

        if len(top1) != expected_queries:
            raise RuntimeError(
                "Expected exactly one Top1 row per query."
            )

        if not (
            (
                top1[
                    "source_view"
                ].astype(str)
                == "center_square"
            )
            & (
                top1[
                    "source_rank"
                ].astype(int)
                == 1
            )
        ).all():
            raise RuntimeError(
                "Frozen center-square Top1 authority is not preserved."
            )

    return {
        "query_count": int(
            expected_queries
        ),
        "candidate_budget": int(
            expected_budget
        ),
        "candidate_rows": int(
            len(pool)
        ),
        "source_counts": {
            str(k): int(v)
            for k, v in (
                pool[
                    "source_view"
                ]
                .value_counts()
                .to_dict()
                .items()
            )
        },
        "top1_center_preserved": bool(
            require_center_top1
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RG1.0: verify the frozen QV1.4 candidate-pool and image "
            "inputs before geometry research begins."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
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
    contract = cfg[
        "rg1_0"
    ]

    trajectory_spec_path = resolve(
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

    for path in (
        trajectory_spec_path,
        manifest_path,
        pool_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    spec = load_trajectory_spec(
        trajectory_spec_path
    )

    if spec.role != "development":
        raise RuntimeError(
            "RG1.0 development trajectory must have role='development'."
        )

    map_variant = str(
        dev[
            "map_variant"
        ]
    )

    variants = spec.raw[
        "map"
    ][
        "variants"
    ]

    if map_variant not in variants:
        raise RuntimeError(
            f"Trajectory spec does not declare required map variant {map_variant!r}."
        )

    tile_index_path = resolve(
        variants[
            map_variant
        ][
            "tile_index"
        ]
    )

    if not tile_index_path.exists():
        raise FileNotFoundError(
            tile_index_path
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

    if bool(
        contract[
            "forbid_reference_columns"
        ]
    ):
        assert_no_reference_columns(
            manifest,
            label="canonical manifest",
        )
        assert_no_reference_columns(
            pool,
            label="frozen candidate pool",
        )

    pool["query_id"] = pd.to_numeric(
        pool[
            "query_id"
        ],
        errors="raise",
    ).astype(int)
    pool["pool_rank"] = pd.to_numeric(
        pool[
            "pool_rank"
        ],
        errors="raise",
    ).astype(int)
    pool["source_rank"] = pd.to_numeric(
        pool[
            "source_rank"
        ],
        errors="raise",
    ).astype(int)
    pool["tile_id"] = pool[
        "tile_id"
    ].astype(str)

    manifest[
        "query_id"
    ] = pd.to_numeric(
        manifest[
            "query_id"
        ],
        errors="raise",
    ).astype(int)

    summary = validate_pool_contract(
        pool=pool,
        manifest=manifest,
        expected_queries=int(
            dev[
                "expected_queries"
            ]
        ),
        expected_budget=int(
            dev[
                "expected_candidate_budget"
            ]
        ),
        allowed_sources=set(
            contract[
                "allowed_candidate_sources"
            ]
        ),
        require_center_top1=bool(
            contract[
                "require_center_top1"
            ]
        ),
    )

    required_tile_columns = {
        "tile_id",
        "tile_path",
        "left_easting",
        "right_easting",
        "bottom_northing",
        "top_northing",
        "center_easting",
        "center_northing",
    }

    missing_tile = sorted(
        required_tile_columns
        - set(
            tile_index.columns
        )
    )

    if missing_tile:
        raise RuntimeError(
            f"768_s256 tile index missing geometry columns: {missing_tile}"
        )

    tile_index[
        "tile_id"
    ] = tile_index[
        "tile_id"
    ].astype(str)

    if tile_index[
        "tile_id"
    ].duplicated().any():
        raise RuntimeError(
            "Tile index has duplicate tile IDs."
        )

    tile_lookup = tile_index.set_index(
        "tile_id"
    )

    selected_ids = set(
        pool[
            "tile_id"
        ].tolist()
    )
    missing_ids = sorted(
        selected_ids
        - set(
            tile_lookup.index
        )
    )

    if missing_ids:
        raise RuntimeError(
            "Frozen candidate pool references tile IDs missing from tile index: "
            f"{missing_ids[:20]}"
        )

    query_missing = []

    if bool(
        contract[
            "require_all_query_images"
        ]
    ):
        for row in manifest.itertuples(
            index=False
        ):
            path = resolve_existing_image(
                row.image_path
            )
            if path is None:
                query_missing.append(
                    {
                        "query_id": int(
                            row.query_id
                        ),
                        "image_path": str(
                            row.image_path
                        ),
                    }
                )

    tile_missing = []

    if bool(
        contract[
            "require_all_tile_images"
        ]
    ):
        for tile_id in sorted(
            selected_ids
        ):
            row = tile_lookup.loc[
                tile_id
            ]
            path = resolve_existing_image(
                row[
                    "tile_path"
                ],
                extra_roots=[
                    tile_index_path.parent,
                ],
            )

            if path is None:
                tile_missing.append(
                    {
                        "tile_id": str(
                            tile_id
                        ),
                        "tile_path": str(
                            row[
                                "tile_path"
                            ]
                        ),
                    }
                )

    if query_missing:
        raise FileNotFoundError(
            "Missing query images: "
            + json.dumps(
                query_missing[
                    :20
                ],
                indent=2,
            )
        )

    if tile_missing:
        raise FileNotFoundError(
            "Missing tile images: "
            + json.dumps(
                tile_missing[
                    :20
                ],
                indent=2,
            )
        )

    geometry_cfg = cfg[
        "geometry_baseline"
    ]

    output_root = resolve(
        args.output_root
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "stage": "RG1.0",
        "status": "PASS_RG1_GEOMETRY_CONTRACT_PREFLIGHT",
        "created_at_utc": now_utc(),
        "trajectory": {
            "id": spec.trajectory_id,
            "role": spec.role,
            "spec_path": str(
                trajectory_spec_path
            ),
            "canonical_manifest_path": str(
                manifest_path
            ),
            "canonical_manifest_sha256": sha256_file(
                manifest_path
            ),
        },
        "candidate_pool": {
            "path": str(
                pool_path
            ),
            "sha256": sha256_file(
                pool_path
            ),
            **summary,
            "unique_tile_ids": int(
                len(
                    selected_ids
                )
            ),
        },
        "map": {
            "variant": map_variant,
            "tile_index_path": str(
                tile_index_path
            ),
            "tile_index_sha256": sha256_file(
                tile_index_path
            ),
            "selected_tile_ids_all_resolved": True,
            "selected_tile_images_all_resolved": True,
        },
        "query_images": {
            "all_resolved": True,
            "count": int(
                len(
                    manifest
                )
            ),
        },
        "geometry_contract": geometry_cfg,
        "scientific_boundary": {
            "candidate_pool_changed": False,
            "candidate_selection_enabled": False,
            "hybrid_rank_prior_enabled": False,
            "reference_loaded": False,
            "bootstrap_enabled": False,
            "state_enabled": False,
            "blind_stress_used_for_tuning": False,
        },
        "next_stage": {
            "stage": "RG1.1",
            "action": (
                "Compute ORB/homography evidence for every frozen Top20 "
                "candidate over all 403 development queries, freeze/hash the "
                "blind evidence table, then attach reference only for "
                "candidate-class and projection-quality diagnostics."
            ),
        },
    }

    report_path = (
        output_root
        / "rg1_0_geometry_contract_preflight_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("=" * 108)
    print(
        "RG1.0 — REGION-CANDIDATE GEOMETRY CONTRACT PREFLIGHT"
    )
    print("=" * 108)
    print(
        "trajectory:",
        spec.trajectory_id,
    )
    print(
        "queries:",
        summary[
            "query_count"
        ],
    )
    print(
        "candidate rows:",
        summary[
            "candidate_rows"
        ],
    )
    print(
        "candidate budget:",
        summary[
            "candidate_budget"
        ],
    )
    print(
        "unique selected tiles:",
        len(
            selected_ids
        ),
    )
    print(
        "source counts:",
        summary[
            "source_counts"
        ],
    )
    print(
        "center Top1 preserved:",
        summary[
            "top1_center_preserved"
        ],
    )
    print(
        "query images resolved:",
        len(
            manifest
        ),
    )
    print(
        "tile images resolved:",
        len(
            selected_ids
        ),
    )
    print(
        "reference loaded:",
        False,
    )
    print(
        "selection enabled:",
        False,
    )
    print()
    print(
        "STATUS: PASS_RG1_GEOMETRY_CONTRACT_PREFLIGHT"
    )
    print(
        "report:",
        report_path,
    )


if __name__ == "__main__":
    main()
