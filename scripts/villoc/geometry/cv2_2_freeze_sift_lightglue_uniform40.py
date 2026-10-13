#!/usr/bin/env python3
"""CV2.2: freeze reference-free uniform40 SIFT+LightGlue evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
import yaml

from uavloc.geometry.sift_lightglue_absolute_frontend import (
    PINNED_COMMIT,
    SIFTLightGlueAbsoluteFrontend,
)

ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/crossview_scale_rotation_domain_cv2_2.yaml"


def resolve(v: str | Path) -> Path:
    p = Path(v).expanduser()
    return p if p.is_absolute() else ROOT / p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.9f").encode()


def canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def freeze(files: dict[Path, bytes]) -> None:
    bad = [str(p) for p, d in files.items() if p.exists() and p.read_bytes() != d]
    if bad:
        raise RuntimeError("Refusing to overwrite differing CV2.2 outputs: " + ", ".join(bad))
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
    raise FileNotFoundError("Could not resolve: " + str(value))


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "CV2.2":
        raise RuntimeError("Expected CV2.2 config")
    for _, value in cfg["scientific_boundary"].items():
        if value is not False:
            raise RuntimeError("CV2.2 scientific boundary must remain false")
    g = cfg["generation"]
    if int(g["expected_queries"]) != 40 or int(g["candidates_per_query"]) != 20 or int(g["expected_pairs"]) != 800:
        raise RuntimeError("CV2.2 screen size changed")
    if g["pair_order"] != "query_id_then_pool_rank":
        raise RuntimeError("CV2.2 pair order changed")
    if g["retain_failed_pairs"] is not True or g["runtime_fields_frozen"] is not False:
        raise RuntimeError("CV2.2 generation contract changed")


def verify_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    block = cfg["cv2_1"]
    report = json.loads(resolve(block["report"]).read_text())
    if report.get("status") != block["expected_status"]:
        raise RuntimeError("CV2.1 report status mismatch")
    if report["provenance"]["evidence_sha256"] != block["expected_evidence_sha256"]:
        raise RuntimeError("CV2.1 smoke evidence hash mismatch")
    if report["provenance"]["lightglue_weights_sha256"] != block["expected_weights_sha256"]:
        raise RuntimeError("CV2.1 SIFT-LightGlue weights changed")
    if report["frontend"]["pinned_commit"] != PINNED_COMMIT:
        raise RuntimeError("CV2.1 pinned LightGlue commit changed")
    if report["scientific_boundary"]["reference_loaded"] is not False:
        raise RuntimeError("CV2.1 reference boundary violated")
    return report


def load_protocol(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = yaml.safe_load(resolve(cfg["protocol"]["config"]).read_text())
    if protocol.get("stage") != "CV2.0":
        raise RuntimeError("Expected frozen CV2.0 protocol")
    if protocol["lineage"]["expected_top20_sha256"] != cfg["lineage"]["expected_top20_sha256"]:
        raise RuntimeError("CV2.0/CV2.2 Top20 hash contract differs")
    if protocol["lineage"]["expected_cv1_uniform40_sha256"] != cfg["lineage"]["expected_cv1_uniform40_sha256"]:
        raise RuntimeError("CV1 control hash contract differs")
    return protocol, protocol["development"]


def main() -> None:
    ap = argparse.ArgumentParser(description="CV2.2 reference-free SIFT+LightGlue uniform40 freeze")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    verify_config(cfg)
    smoke = verify_smoke(cfg)
    protocol, d = load_protocol(cfg)

    top20_path = resolve(d["top20"])
    if sha256_file(top20_path) != cfg["lineage"]["expected_top20_sha256"]:
        raise RuntimeError("Frozen QV1.4 Top20 bytes changed")

    cv1_path = resolve(protocol["lineage"]["cv1_uniform40_evidence"])
    if sha256_file(cv1_path) != cfg["lineage"]["expected_cv1_uniform40_sha256"]:
        raise RuntimeError("Frozen CV1 uniform40 evidence bytes changed")

    spec = yaml.safe_load(resolve(d["spec"]).read_text())
    manifest = pd.read_csv(resolve(d["manifest"]))
    pool = pd.read_csv(top20_path)
    tile_index_path = resolve(spec["map"]["variants"]["768_s256"]["tile_index"])
    tiles = pd.read_csv(tile_index_path)

    manifest["query_id"] = pd.to_numeric(manifest["query_id"], errors="raise").astype(int)
    pool["query_id"] = pd.to_numeric(pool["query_id"], errors="raise").astype(int)
    pool["pool_rank"] = pd.to_numeric(pool["pool_rank"], errors="raise").astype(int)
    pool["source_rank"] = pd.to_numeric(pool["source_rank"], errors="raise").astype(int)
    pool["tile_id"] = pool["tile_id"].astype(str)
    tiles["tile_id"] = tiles["tile_id"].astype(str)

    refs = manifest["reference_available"].astype(str).str.lower().str.strip()
    if not refs.isin(["false", "0", "no", "n", "f"]).all():
        raise RuntimeError("Development manifest exposes reference")

    query_ids = [int(x) for x in protocol["uniform40"]["query_ids"]]
    if len(query_ids) != int(cfg["generation"]["expected_queries"]) or len(set(query_ids)) != len(query_ids):
        raise RuntimeError("Uniform40 query contract changed")

    manifest = manifest[manifest["query_id"].isin(query_ids)].copy()
    pool = pool[pool["query_id"].isin(query_ids)].copy()
    pool = pool.sort_values(["query_id", "pool_rank"], kind="mergesort").reset_index(drop=True)
    if len(manifest) != 40 or len(pool) != 800:
        raise RuntimeError("Uniform40 input count changed")
    for qid, group in pool.groupby("query_id", sort=True):
        ordered = group.sort_values("pool_rank")
        if len(ordered) != 20 or ordered["pool_rank"].tolist() != list(range(1, 21)):
            raise RuntimeError(f"q{qid} Top20 rank contract changed")

    qlookup = manifest.set_index("query_id")
    tlookup = tiles.set_index("tile_id")
    frontend = SIFTLightGlueAbsoluteFrontend(
        resolve(protocol["lightglue"]["repo"]),
        device_name=str(protocol["lightglue"]["device"]),
    )
    meta = frontend.describe()
    if meta["lightglue_weights_sha256"] != cfg["cv2_1"]["expected_weights_sha256"]:
        raise RuntimeError("SIFT-LightGlue weights changed since CV2.1")

    rows = []
    runtimes = []
    start = time.perf_counter()
    progress_every = int(cfg["generation"]["progress_every_pairs"])

    for index, row in enumerate(pool.itertuples(index=False), start=1):
        if index == 1 or index % progress_every == 0 or index == len(pool):
            print(f"CV2.2 SIFT+LightGlue pair {index}/{len(pool)}", flush=True)
        qid, tile_id = int(row.query_id), str(row.tile_id)
        qpath = resolve_existing(qlookup.loc[qid, "image_path"])
        tpath = resolve_existing(tlookup.loc[tile_id, "tile_path"], tile_index_path.parent)
        tile = tlookup.loc[tile_id]
        m = frontend.measure_pair(
            qpath,
            tpath,
            tile_bounds=(
                float(tile["left_easting"]), float(tile["right_easting"]),
                float(tile["bottom_northing"]), float(tile["top_northing"]),
            ),
        )
        runtimes.append(float(m.runtime.total_ms))
        out = {
            "trajectory_id": str(spec["trajectory"]["id"]),
            "query_id": qid,
            "pool_rank": int(row.pool_rank),
            "tile_id": tile_id,
            "source_view": str(row.source_view),
            "source_rank": int(row.source_rank),
            "frontend_name": m.frontend_name,
            "good_matches": int(m.good_matches),
            "inliers": int(m.inliers),
            "homography_ok": bool(m.homography_ok),
            "finite_projection": bool(m.finite_projection),
            "projected_inside_tile": bool(m.projected_inside_tile),
            "projected_easting": float(m.projected_easting),
            "projected_northing": float(m.projected_northing),
        }
        for name, value in sorted(m.evidence.items()):
            if name in {"query_feature_cache_hit", "tile_feature_cache_hit"}:
                continue
            out[f"evidence_{name}"] = value
        rows.append(out)

    wall_s = float(time.perf_counter() - start)
    frame = pd.DataFrame(rows).sort_values(["query_id", "pool_rank"], kind="mergesort").reset_index(drop=True)
    if len(frame) != 800 or frame["query_id"].nunique() != 40:
        raise RuntimeError("CV2.2 generated evidence count changed")

    outroot = resolve(cfg["output_root"])
    evidence_path = outroot / "cv2_2_sift_lightglue_uniform40_reference_free_evidence.csv"
    evidence_bytes = canonical_csv(frame)

    h = frame["homography_ok"].astype(bool)
    finite = frame["finite_projection"].astype(bool)
    perq_h = h.groupby(frame["query_id"]).sum()
    report = {
        "stage": "CV2.2",
        "status": "PASS_CV2_2_REFERENCE_FREE_SIFT_LIGHTGLUE_UNIFORM40_FREEZE",
        "frontend": meta,
        "summary": {
            "queries": int(frame["query_id"].nunique()),
            "pairs": int(len(frame)),
            "unique_tiles": int(frame["tile_id"].nunique()),
            "homography_ok_pairs": int(h.sum()),
            "finite_projection_pairs": int(finite.sum()),
            "inside_projection_pairs": int(frame["projected_inside_tile"].astype(bool).sum()),
            "queries_any_homography": int((perq_h > 0).sum()),
            "queries_all20_homographies": int((perq_h == 20).sum()),
            "median_matches": float(pd.to_numeric(frame["good_matches"]).median()),
            "median_inliers": float(pd.to_numeric(frame["inliers"]).median()),
            "median_runtime_ms_pair_not_frozen": float(np.median(runtimes)),
            "wall_s_not_frozen": wall_s,
        },
        "scientific_boundary": {
            "reference_loaded": False,
            "oracle_labels_loaded": False,
            "scale_m2_applied_to_sift": False,
            "candidate_selected": False,
            "reranking_executed": False,
            "threshold_tuned": False,
            "matcher_profile_sweep_executed": False,
            "full403_executed": False,
            "state_modified": False,
            "temporal_fusion_executed": False,
            "runtime_fields_frozen": False,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "protocol_config_sha256": sha256_file(resolve(cfg["protocol"]["config"])),
            "cv2_1_report_sha256": sha256_file(resolve(cfg["cv2_1"]["report"])),
            "cv2_1_smoke_evidence_sha256": smoke["provenance"]["evidence_sha256"],
            "top20_sha256": sha256_file(top20_path),
            "cv1_uniform40_evidence_sha256": sha256_file(cv1_path),
            "sift_lightglue_weights_sha256": meta["lightglue_weights_sha256"],
            "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        },
    }

    report_path = outroot / "cv2_2_freeze_report.json"
    freeze({evidence_path: evidence_bytes, report_path: canonical_json(report)})

    print("Status:", report["status"])
    print("Evidence SHA256:", report["provenance"]["evidence_sha256"])
    print("Homography ok pairs:", report["summary"]["homography_ok_pairs"], "/ 800")
    print("Finite projection pairs:", report["summary"]["finite_projection_pairs"], "/ 800")
    print("Queries all20 homographies:", report["summary"]["queries_all20_homographies"], "/ 40")
    print("Median matches/inliers:", f"{report['summary']['median_matches']:.1f}/{report['summary']['median_inliers']:.1f}")
    print("Median total ms/pair (not frozen):", f"{report['summary']['median_runtime_ms_pair_not_frozen']:.3f}")
    print("Observed wall s (not frozen):", f"{wall_s:.3f}")
    print("Reference loaded: False")
    print("scale_m2 applied to SIFT: False")
    print("Candidate selected/reranked: False")
    print("Accuracy claim: False")
    print("Report:", report_path)


if __name__ == "__main__":
    main()
