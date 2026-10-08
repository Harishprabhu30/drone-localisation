#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from uavloc.retrieval import (
    DinoV2CachedRetrievalBackend,
    cross_scale_duplicate,
)


ROOT = Path.cwd().resolve()
DEFAULT_CONFIG = Path(
    "configs/research/retrieval_candidate_pool_v2_r2.yaml"
)
DEFAULT_REPORT = Path(
    "outputs/research_runs/retrieval_candidate_pool_v2/"
    "r2_cross_scale_fusion/reports/r2_0_fusion_contract_preflight.json"
)

REQUIRED_INDEX_COLUMNS = {
    "tile_id",
    "tile_path",
    "grid_row",
    "grid_col",
    "pixel_col_off",
    "pixel_row_off",
    "tile_width_px",
    "tile_height_px",
    "left_easting",
    "bottom_northing",
    "right_easting",
    "top_northing",
    "center_easting",
    "center_northing",
    "ground_width_m",
    "ground_height_m",
}


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    config_path = resolve(args.config)
    cfg = yaml.safe_load(config_path.read_text())

    fusion = cfg["fusion"]
    if fusion["method"] != "physical_region_rrf_v1":
        raise RuntimeError("Unexpected R2.0 fusion method.")
    if float(fusion["rrf_k"]) < 0.0:
        raise RuntimeError("RRF k must be non-negative.")
    if int(fusion["per_scale_depth"]) < int(fusion["final_top_k"]):
        raise RuntimeError(
            "per_scale_depth must be >= final_top_k so dedup can refill budget."
        )
    if int(fusion["final_top_k"]) != 20:
        raise RuntimeError("R2 must preserve the unchanged ORB Top-20 budget.")

    region_cfg = fusion["physical_region"]
    if not bool(region_cfg["cross_scale_only"]):
        raise RuntimeError("R2.0 must not collapse same-scale neighbors.")
    if int(region_cfg["max_members_per_scale"]) != 1:
        raise RuntimeError("R2.0 allows at most one member per scale/region.")

    query_cache = resolve(cfg["control"]["query_cache"])
    if not query_cache.exists():
        raise FileNotFoundError(query_cache)

    backend = DinoV2CachedRetrievalBackend()
    query_rep = backend.load_representation(query_cache)

    if len(query_rep.ids) != int(cfg["control"]["query_count_expected"]):
        raise RuntimeError("Canonical query count mismatch.")

    expert_records = {}
    index_frames = {}

    for name in fusion["scale_order"]:
        spec = cfg["experts"][name]
        index_path = resolve(spec["index_csv"])
        cache_path = resolve(spec["map_cache"])

        for path in (index_path, cache_path):
            if not path.exists():
                raise FileNotFoundError(path)

        frame = pd.read_csv(index_path)
        missing = REQUIRED_INDEX_COLUMNS - set(frame.columns)
        if missing:
            raise RuntimeError(
                f"{name}: index missing physical provenance {sorted(missing)}"
            )

        frame["tile_id"] = frame["tile_id"].astype(str)
        map_rep = backend.load_representation(cache_path)
        backend.validate_pair(query_rep, map_rep)

        if not frame["tile_id"].to_numpy().tolist() == map_rep.ids.astype(str).tolist():
            raise RuntimeError(f"{name}: tile-index/cache order mismatch.")

        if not (frame["tile_width_px"].astype(int) == int(spec["tile_size_px"])).all():
            raise RuntimeError(f"{name}: tile-size mismatch.")
        if not (frame["tile_height_px"].astype(int) == int(spec["tile_size_px"])).all():
            raise RuntimeError(f"{name}: tile-size mismatch.")

        expert_records[name] = {
            "role": spec["role"],
            "tile_count": int(len(frame)),
            "tile_size_px": int(spec["tile_size_px"]),
            "stride_px": int(spec["stride_px"]),
            "ground_footprint_m": float(spec["ground_footprint_m"]),
            "index_csv": str(index_path),
            "index_sha256": sha256_file(index_path),
            "map_cache": str(cache_path),
            "map_cache_sha256": sha256_file(cache_path),
            "descriptor_shape": list(map_rep.descriptors.shape),
        }
        index_frames[name] = frame

    # Structural sanity check using the real center lattices: ensure at least
    # one cross-scale pair satisfies the declared duplicate relation.
    found_duplicate = False
    names = list(fusion["scale_order"])
    for a_name, b_name in zip(names[:-1], names[1:]):
        a = index_frames[a_name].iloc[len(index_frames[a_name]) // 2].to_dict()
        best = None
        for row in index_frames[b_name].to_dict(orient="records"):
            if cross_scale_duplicate(
                {**a, "variant": a_name},
                {**row, "variant": b_name},
                max_center_distance_m=float(
                    region_cfg["max_center_distance_m"]
                ),
                min_iou=float(region_cfg["minimum_bbox_iou"]),
            ):
                best = row
                break
        if best is not None:
            found_duplicate = True
            break

    if not found_duplicate:
        raise RuntimeError(
            "Declared physical-region duplicate rule matches no real "
            "cross-scale candidates; fusion contract is unusable."
        )

    experiments = cfg["experiments"]
    expected = {
        "pair_512_768": ["512_s256", "768_s256"],
        "triple_512_768_1024": [
            "512_s256",
            "768_s256",
            "1024_s256",
        ],
    }
    actual = {
        key: value["experts"]
        for key, value in experiments.items()
    }
    if actual != expected:
        raise RuntimeError(
            f"R2.0 experiment family changed unexpectedly: {actual}"
        )

    report = {
        "stage": "R2.0",
        "status": "PASS_R2_FUSION_CONTRACT_PREFLIGHT",
        "created_at_utc": now_utc(),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "query_cache": {
            "path": str(query_cache),
            "sha256": sha256_file(query_cache),
            "query_count": int(len(query_rep.ids)),
            "descriptor_shape": list(query_rep.descriptors.shape),
        },
        "experts": expert_records,
        "fusion_contract": fusion,
        "experiments": experiments,
        "blindness": {
            "query_coordinates_used": False,
            "oracle_used": False,
            "reference_used": False,
            "map_geometry_used_for_dedup": True,
        },
        "next_stage": {
            "stage": "R2.1",
            "action": (
                "Rank each expert, fuse physical regions with neutral RRF, "
                "freeze merged Top-20 lists, then attach evaluation only."
            ),
        },
    }

    report_path = resolve(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 88)
    print("R2.0 — CROSS-SCALE FUSION CONTRACT PREFLIGHT")
    print("=" * 88)
    print("status:", report["status"])
    print("query count:", len(query_rep.ids))
    print(
        "fusion:",
        f"RRF(k={fusion['rrf_k']})",
        f"depth={fusion['per_scale_depth']}",
        f"Top-{fusion['final_top_k']}",
    )
    print(
        "dedup:",
        f"center<={region_cfg['max_center_distance_m']}m",
        f"IoU>={region_cfg['minimum_bbox_iou']}",
        "cross-scale only",
    )
    for name, item in expert_records.items():
        print(
            f"{name:10s} tiles={item['tile_count']:3d} "
            f"shape={item['descriptor_shape']} role={item['role']}"
        )
    print("report:", report_path)


if __name__ == "__main__":
    main()
