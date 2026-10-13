#!/usr/bin/env python3
"""CV2.1: reference-free pinned SIFT+LightGlue 9-pair smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
import yaml

from uavloc.geometry.sift_lightglue_absolute_frontend import (
    DETECTION_THRESHOLD,
    EDGE_THRESHOLD,
    FEATURE_TYPE,
    LIGHTGLUE_DEPTH_CONFIDENCE,
    LIGHTGLUE_FILTER_THRESHOLD,
    LIGHTGLUE_WIDTH_CONFIDENCE,
    MAX_KEYPOINTS,
    MIN_HOMOGRAPHY_INLIERS,
    NMS_RADIUS,
    NUM_OCTAVES,
    PINNED_COMMIT,
    RANSAC_THRESHOLD_PX,
    RESIZE_LONG,
    ROOTSIFT,
    SIFT_BACKEND,
    SIFTLightGlueAbsoluteFrontend,
)

ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/crossview_scale_rotation_domain_cv2.yaml"


def resolve(v: str | Path) -> Path:
    p = Path(v).expanduser()
    return p if p.is_absolute() else ROOT / p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def deterministic_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.9f").encode()


def deterministic_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def freeze(files: dict[Path, bytes]) -> None:
    bad = [str(p) for p, d in files.items() if p.exists() and p.read_bytes() != d]
    if bad:
        raise RuntimeError("Refusing to overwrite differing CV2.1 outputs: " + ", ".join(bad))
    for p, d in files.items():
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("xb") as f:
                f.write(d)


def resolve_existing(value: Any, *roots: Path) -> Path:
    p = Path(str(value)).expanduser()
    candidates = [p] if p.is_absolute() else [ROOT / p, *[r / p for r in roots]]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(value)


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "CV2.0":
        raise RuntimeError("Expected CV2.0 config")
    x = cfg["lightglue"]
    expected = {
        "pinned_commit": PINNED_COMMIT,
        "feature_type": FEATURE_TYPE,
        "device": "cpu",
    }
    for k, v in expected.items():
        if x.get(k) != v:
            raise RuntimeError(f"CV2 contract changed: {k}")
    s = x["sift"]
    checks = {
        "backend": SIFT_BACKEND, "rootsift": ROOTSIFT, "max_keypoints": MAX_KEYPOINTS,
        "resize_long": RESIZE_LONG, "nms_radius": NMS_RADIUS,
        "detection_threshold": DETECTION_THRESHOLD, "edge_threshold": EDGE_THRESHOLD,
        "num_octaves": NUM_OCTAVES,
    }
    for k, v in checks.items():
        if s.get(k) != v:
            raise RuntimeError(f"SIFT contract changed: {k}")
    m = x["matcher"]
    for k, v in {
        "filter_threshold": LIGHTGLUE_FILTER_THRESHOLD,
        "depth_confidence": LIGHTGLUE_DEPTH_CONFIDENCE,
        "width_confidence": LIGHTGLUE_WIDTH_CONFIDENCE,
    }.items():
        if m.get(k) != v:
            raise RuntimeError(f"LightGlue contract changed: {k}")
    g = x["geometry"]
    if g["ransac_threshold_px"] != RANSAC_THRESHOLD_PX or g["min_homography_inliers"] != MIN_HOMOGRAPHY_INLIERS:
        raise RuntimeError("Geometry contract changed")
    for k, v in cfg["scientific_boundary"].items():
        if v is not False:
            raise RuntimeError(f"{k} must be false")


def main() -> None:
    ap = argparse.ArgumentParser(description="CV2.1 reference-free SIFT+LightGlue smoke")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args()
    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    verify_config(cfg)

    d = cfg["development"]
    if sha256_file(resolve(d["top20"])) != cfg["lineage"]["expected_top20_sha256"]:
        raise RuntimeError("Frozen Top20 changed")
    spec = yaml.safe_load(resolve(d["spec"]).read_text())
    manifest = pd.read_csv(resolve(d["manifest"]))
    pool = pd.read_csv(resolve(d["top20"]))
    tile_index_path = resolve(spec["map"]["variants"]["768_s256"]["tile_index"])
    tiles = pd.read_csv(tile_index_path)
    for frame in (manifest, pool, tiles):
        if "query_id" in frame.columns:
            frame["query_id"] = pd.to_numeric(frame["query_id"], errors="raise").astype(int)
    pool["pool_rank"] = pd.to_numeric(pool["pool_rank"], errors="raise").astype(int)
    pool["tile_id"] = pool["tile_id"].astype(str)
    tiles["tile_id"] = tiles["tile_id"].astype(str)

    refs = manifest["reference_available"].astype(str).str.lower().str.strip()
    if not refs.isin(["false", "0", "no", "n", "f"]).all():
        raise RuntimeError("Manifest exposes reference")

    qs = [int(x) for x in cfg["cv2_1_smoke"]["query_ids"]]
    ranks = [int(x) for x in cfg["cv2_1_smoke"]["pool_ranks"]]
    sample = pool[pool["query_id"].isin(qs) & pool["pool_rank"].isin(ranks)].copy()
    sample = sample.sort_values(["query_id", "pool_rank"], kind="mergesort")
    if len(sample) != int(cfg["cv2_1_smoke"]["expected_pairs"]):
        raise RuntimeError("CV2.1 smoke pair count changed")

    qlookup = manifest.set_index("query_id")
    tlookup = tiles.set_index("tile_id")
    frontend = SIFTLightGlueAbsoluteFrontend(resolve(cfg["lightglue"]["repo"]), device_name="cpu")
    meta = frontend.describe()
    rows = []
    for row in sample.itertuples(index=False):
        qid, tile_id = int(row.query_id), str(row.tile_id)
        qpath = resolve_existing(qlookup.loc[qid, "image_path"])
        tpath = resolve_existing(tlookup.loc[tile_id, "tile_path"], tile_index_path.parent)
        tile = tlookup.loc[tile_id]
        m = frontend.measure_pair(
            qpath, tpath,
            tile_bounds=(
                float(tile["left_easting"]), float(tile["right_easting"]),
                float(tile["bottom_northing"]), float(tile["top_northing"]),
            ),
        )
        rows.append({
            "query_id": qid,
            "pool_rank": int(row.pool_rank),
            "tile_id": tile_id,
            "good_matches": int(m.good_matches),
            "inliers": int(m.inliers),
            "homography_ok": bool(m.homography_ok),
            "finite_projection": bool(m.finite_projection),
            "projected_inside_tile": bool(m.projected_inside_tile),
            "projected_easting": float(m.projected_easting),
            "projected_northing": float(m.projected_northing),
            "query_keypoints": int(m.evidence["query_keypoints"]),
            "tile_keypoints": int(m.evidence["tile_keypoints"]),
            "query_scales_present": bool(m.evidence["query_scales_present"]),
            "query_orientations_present": bool(m.evidence["query_orientations_present"]),
            "tile_scales_present": bool(m.evidence["tile_scales_present"]),
            "tile_orientations_present": bool(m.evidence["tile_orientations_present"]),
            "total_ms": float(m.runtime.total_ms),
        })

    frame = pd.DataFrame(rows)
    if not frame[[
        "query_scales_present", "query_orientations_present",
        "tile_scales_present", "tile_orientations_present"
    ]].all().all():
        raise RuntimeError("SIFT scale/orientation fields missing in smoke")

    frozen = frame.drop(columns=["total_ms"])
    outroot = resolve(cfg["output_root"]) / "cv2_1"
    evidence_path = outroot / "cv2_1_sift_lightglue_smoke.csv"
    evidence_bytes = deterministic_csv(frozen)
    report = {
        "stage": "CV2.1",
        "status": "PASS_CV2_1_REFERENCE_FREE_SIFT_LIGHTGLUE_SMOKE",
        "frontend": meta,
        "sample": {"query_ids": qs, "pool_ranks": ranks, "pairs": int(len(frame))},
        "summary": {
            "homography_ok": int(frame["homography_ok"].sum()),
            "finite_projection": int(frame["finite_projection"].sum()),
            "inside_projection": int(frame["projected_inside_tile"].sum()),
            "median_query_keypoints": float(frame["query_keypoints"].median()),
            "median_tile_keypoints": float(frame["tile_keypoints"].median()),
            "median_matches": float(frame["good_matches"].median()),
            "median_inliers": float(frame["inliers"].median()),
            "median_total_ms_pair_not_frozen": float(frame["total_ms"].median()),
        },
        "scientific_boundary": {
            "reference_loaded": False, "oracle_labels_loaded": False,
            "scale_m2_applied_to_sift": False, "uniform40_executed": False,
            "full403_executed": False, "threshold_tuned": False,
            "candidate_reranked": False, "state_modified": False,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "top20_sha256": sha256_file(resolve(d["top20"])),
            "lightglue_weights_sha256": meta["lightglue_weights_sha256"],
            "sift_source_blob": meta["sift_source_blob"],
            "lightglue_source_blob": meta["lightglue_source_blob"],
            "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        },
    }
    report_path = outroot / "cv2_1_report.json"
    freeze({evidence_path: evidence_bytes, report_path: deterministic_json(report)})

    print("Status:", report["status"])
    print("LightGlue pinned commit:", meta["pinned_commit"])
    print("SIFT source blob:", meta["sift_source_blob"])
    print("LightGlue source blob:", meta["lightglue_source_blob"])
    print("SIFT-LightGlue weights SHA256:", meta["lightglue_weights_sha256"])
    print("Scale/orientation fields present: True")
    print("Homography ok:", report["summary"]["homography_ok"], "/ 9")
    print("Finite projection:", report["summary"]["finite_projection"], "/ 9")
    print("Median query/tile keypoints:", f"{report['summary']['median_query_keypoints']:.1f}/{report['summary']['median_tile_keypoints']:.1f}")
    print("Median matches/inliers:", f"{report['summary']['median_matches']:.1f}/{report['summary']['median_inliers']:.1f}")
    print("Median total ms/pair (not frozen):", f"{report['summary']['median_total_ms_pair_not_frozen']:.3f}")
    print("Reference loaded: False")
    print("scale_m2 applied to SIFT: False")
    print("Evidence SHA256:", report["provenance"]["evidence_sha256"])
    print("Report:", report_path)


if __name__ == "__main__":
    main()
