#!/usr/bin/env python3
"""LF2.1-A: reference-free ALIKED+LightGlue extractor-swap preflight/smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT=Path(__file__).resolve().parents[3]
SRC_ROOT=REPO_ROOT/"src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0,str(SRC_ROOT))

import pandas as pd
import yaml

from uavloc.geometry.aliked_lightglue_absolute_frontend import (
    ALIKED_MODEL_NAME,
    DETECTION_THRESHOLD,
    FEATURE_TYPE,
    MAX_KEYPOINTS,
    NMS_RADIUS,
    ALIKEDLightGlueAbsoluteFrontend,
)
from uavloc.geometry.lightglue_absolute_frontend import (
    LIGHTGLUE_DEPTH_CONFIDENCE,
    LIGHTGLUE_FILTER_THRESHOLD,
    LIGHTGLUE_WIDTH_CONFIDENCE,
    MIN_HOMOGRAPHY_INLIERS,
    PINNED_COMMIT,
    RANSAC_THRESHOLD_PX,
    RESIZE_LONG,
)

ROOT=REPO_ROOT
DEFAULT_CONFIG=ROOT/"configs/research/learned_geometric_frontend_lf2_1a.yaml"

MANIFEST_REQUIRED={"query_id","image_path","reference_available"}
POOL_REQUIRED={"query_id","pool_rank","tile_id"}
TILE_REQUIRED={
    "tile_id","tile_path","left_easting","right_easting",
    "bottom_northing","top_northing",
}


def resolve(v: str | Path)->Path:
    p=Path(v).expanduser()
    return p if p.is_absolute() else ROOT/p


def sha256_file(path: Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def deterministic_csv(frame: pd.DataFrame)->bytes:
    return frame.to_csv(index=False,lineterminator="\n",float_format="%.9f").encode("utf-8")


def deterministic_json(value: dict[str,Any])->bytes:
    return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode("utf-8")


def freeze(files: dict[Path,bytes])->None:
    bad=[str(p) for p,d in files.items() if p.exists() and p.read_bytes()!=d]
    if bad:
        raise RuntimeError("Refusing to overwrite differing LF2.1-A outputs: "+", ".join(bad))
    for p,d in files.items():
        if p.exists():
            continue
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open("xb") as f:
            f.write(d)


def require_columns(frame: pd.DataFrame, required: set[str], label: str)->None:
    missing=sorted(required-set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")


def resolve_existing(value: Any, *roots: Path)->Path:
    raw=Path(str(value)).expanduser()
    candidates=[raw] if raw.is_absolute() else [ROOT/raw,*[r/raw for r in roots]]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Could not resolve file; tried: "+", ".join(map(str,candidates)))


def verify_config(cfg: dict[str,Any])->None:
    if cfg.get("stage")!="LF2.1-A":
        raise RuntimeError("Expected LF2.1-A config")
    x=cfg["aliked_lightglue"]
    expected={
        "pinned_commit":PINNED_COMMIT,
        "feature_type":FEATURE_TYPE,
        "aliked_model_name":ALIKED_MODEL_NAME,
        "max_keypoints":MAX_KEYPOINTS,
        "resize_long":RESIZE_LONG,
        "detection_threshold":DETECTION_THRESHOLD,
        "nms_radius":NMS_RADIUS,
        "filter_threshold":LIGHTGLUE_FILTER_THRESHOLD,
        "depth_confidence":LIGHTGLUE_DEPTH_CONFIDENCE,
        "width_confidence":LIGHTGLUE_WIDTH_CONFIDENCE,
        "ransac_threshold_px":RANSAC_THRESHOLD_PX,
        "min_homography_inliers":MIN_HOMOGRAPHY_INLIERS,
    }
    for key,value in expected.items():
        if x.get(key)!=value:
            raise RuntimeError(f"ALIKED+LightGlue contract changed: {key}")
    for key,value in cfg["scientific_boundary"].items():
        if value is not False:
            raise RuntimeError(f"{key} must be false")


def verify_superpoint_baseline(cfg: dict[str,Any])->dict[str,Any]:
    block=cfg["baseline_superpoint_lightglue"]
    report_path=resolve(block["report"])
    report=json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status")!="PASS_LF2_0A_SUPERPOINT_LIGHTGLUE_ABSOLUTE_PREFLIGHT_SMOKE":
        raise RuntimeError("LF2.0-A SuperPoint baseline report not PASS")
    if report["outputs"]["smoke_evidence"]["sha256"]!=block["expected_smoke_sha256"]:
        raise RuntimeError("LF2.0-A baseline smoke hash mismatch")
    if report["frontend"]["superpoint_weights_sha256"]!=block["expected_superpoint_weights_sha256"]:
        raise RuntimeError("LF2.0-A SuperPoint weights hash mismatch")
    if report["frontend"]["lightglue_weights_sha256"]!=block["expected_lightglue_weights_sha256"]:
        raise RuntimeError("LF2.0-A LightGlue weights hash mismatch")
    return report


def load_smoke_inputs(cfg: dict[str,Any], key: str)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,Path]:
    block=cfg["frozen_inputs"][key]
    spec_path=resolve(block["spec"])
    manifest_path=resolve(block["manifest"])
    pool_path=resolve(block["top20"])
    if sha256_file(pool_path)!=block["expected_top20_sha256"]:
        raise RuntimeError(f"{key} Top20 hash changed")

    spec=yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    tile_index_path=resolve(spec["map"]["variants"]["768_s256"]["tile_index"])
    manifest=pd.read_csv(manifest_path).copy()
    pool=pd.read_csv(pool_path).copy()
    tiles=pd.read_csv(tile_index_path).copy()

    require_columns(manifest,MANIFEST_REQUIRED,f"{key} manifest")
    require_columns(pool,POOL_REQUIRED,f"{key} pool")
    require_columns(tiles,TILE_REQUIRED,f"{key} tiles")

    manifest["query_id"]=pd.to_numeric(manifest["query_id"],errors="raise").astype(int)
    pool["query_id"]=pd.to_numeric(pool["query_id"],errors="raise").astype(int)
    pool["pool_rank"]=pd.to_numeric(pool["pool_rank"],errors="raise").astype(int)
    pool["tile_id"]=pool["tile_id"].astype(str)
    tiles["tile_id"]=tiles["tile_id"].astype(str)

    refs=manifest["reference_available"].astype(str).str.strip().str.lower()
    if not refs.isin(["false","0","no","n","f"]).all():
        raise RuntimeError(f"{key} manifest exposes reference")

    query_ids=[int(v) for v in block["query_ids"]]
    ranks=[int(v) for v in cfg["frozen_inputs"]["sample_ranks"]]
    sample=pool[
        pool["query_id"].isin(query_ids)
        & pool["pool_rank"].isin(ranks)
    ].copy()
    expected=len(query_ids)*len(ranks)
    if len(sample)!=expected:
        raise RuntimeError(f"{key} sample pair count {len(sample)} != {expected}")
    return manifest,pool,tiles,tile_index_path


def main()->None:
    ap=argparse.ArgumentParser(description="LF2.1-A ALIKED+LightGlue extractor-swap preflight smoke")
    ap.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    args=ap.parse_args()
    cfg_path=resolve(args.config)
    cfg=yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verify_config(cfg)
    baseline_report=verify_superpoint_baseline(cfg)

    frontend=ALIKEDLightGlueAbsoluteFrontend(
        resolve(cfg["aliked_lightglue"]["repo"]),
        device_name=str(cfg["aliked_lightglue"]["device"]),
    )
    frontend_meta=frontend.describe()

    all_rows=[]
    trajectory_reports={}
    ranks=[int(v) for v in cfg["frozen_inputs"]["sample_ranks"]]

    for key in ("development","blind_stress"):
        block=cfg["frozen_inputs"][key]
        manifest,pool,tiles,tile_index_path=load_smoke_inputs(cfg,key)
        q_lookup=manifest.set_index("query_id")
        t_lookup=tiles.set_index("tile_id")
        query_ids=[int(v) for v in block["query_ids"]]
        sample=pool[
            pool["query_id"].isin(query_ids)
            & pool["pool_rank"].isin(ranks)
        ].sort_values(["query_id","pool_rank"],kind="mergesort")

        rows=[]
        for row in sample.itertuples(index=False):
            qid=int(row.query_id)
            tile_id=str(row.tile_id)
            qpath=resolve_existing(q_lookup.loc[qid,"image_path"])
            tpath=resolve_existing(
                t_lookup.loc[tile_id,"tile_path"],
                tile_index_path.parent,
            )
            tile=t_lookup.loc[tile_id]
            bounds=(
                float(tile["left_easting"]),
                float(tile["right_easting"]),
                float(tile["bottom_northing"]),
                float(tile["top_northing"]),
            )
            measurement=frontend.measure_pair(
                qpath,
                tpath,
                tile_bounds=bounds,
            )
            out={
                "trajectory_key":key,
                "query_id":qid,
                "pool_rank":int(row.pool_rank),
                "tile_id":tile_id,
                "frontend_name":measurement.frontend_name,
                "good_matches":int(measurement.good_matches),
                "inliers":int(measurement.inliers),
                "homography_ok":bool(measurement.homography_ok),
                "finite_projection":bool(measurement.finite_projection),
                "projected_inside_tile":bool(measurement.projected_inside_tile),
                "projected_easting":float(measurement.projected_easting),
                "projected_northing":float(measurement.projected_northing),
                "feature_ms":float(measurement.runtime.feature_ms),
                "match_ms":float(measurement.runtime.match_ms),
                "geometry_ms":float(measurement.runtime.geometry_ms),
                "total_ms":float(measurement.runtime.total_ms),
                "device":measurement.runtime.device,
                "backend":measurement.runtime.backend,
            }
            out.update({f"evidence_{k}":v for k,v in measurement.evidence.items()})
            rows.append(out)
            all_rows.append(out)

        frame=pd.DataFrame(rows)
        trajectory_reports[key]={
            "queries":int(frame["query_id"].nunique()),
            "pairs":int(len(frame)),
            "homography_ok":int(frame["homography_ok"].astype(bool).sum()),
            "finite_projection":int(frame["finite_projection"].astype(bool).sum()),
            "inside_projection":int(frame["projected_inside_tile"].astype(bool).sum()),
            "median_matches":float(frame["good_matches"].median()),
            "median_inliers":float(frame["inliers"].median()),
            "median_total_ms_pair":float(frame["total_ms"].median()),
        }

    evidence=pd.DataFrame(all_rows).sort_values(
        ["trajectory_key","query_id","pool_rank"],kind="mergesort"
    ).reset_index(drop=True)

    report={
        "stage":"LF2.1-A",
        "status":"PASS_LF2_1A_ALIKED_LIGHTGLUE_EXTRACTOR_SWAP_SMOKE",
        "frontend":frontend_meta,
        "sample_contract":{
            "development_query_ids":cfg["frozen_inputs"]["development"]["query_ids"],
            "blind_query_ids":cfg["frozen_inputs"]["blind_stress"]["query_ids"],
            "pool_ranks":ranks,
        },
        "trajectories":trajectory_reports,
        "scientific_boundary":{
            "reference_loaded":False,
            "oracle_labels_loaded":False,
            "full_top20_benchmark_executed":False,
            "candidate_reranked":False,
            "selector_executed":False,
            "threshold_tuned":False,
            "learned_model_trained":False,
            "bootstrap_state_modified":False,
        },
        "provenance":{
            "config_sha256":sha256_file(cfg_path),
            "script_sha256":sha256_file(Path(__file__).resolve()),
            "aliked_weights_sha256":frontend_meta["aliked_weights_sha256"],
            "lightglue_weights_sha256":frontend_meta["lightglue_weights_sha256"],
            "lightglue_pinned_commit":frontend_meta["pinned_commit"],
            "lightglue_resolved_git_commit":frontend_meta["resolved_git_commit"],
            "lightglue_source_mode":frontend_meta["source_mode"],
            "lightglue_source_root":frontend_meta["source_root"],
            "lightglue_verified_core_blobs":frontend_meta["verified_core_blobs"],
        },
        "outputs":{},
    }

    outroot=resolve(cfg["output_root"])
    evidence_path=outroot/"lf2_1a_aliked_lightglue_smoke.csv"
    evidence_bytes=deterministic_csv(evidence)
    report["outputs"]["smoke_evidence"]={
        "path":str(evidence_path.relative_to(ROOT)),
        "sha256":hashlib.sha256(evidence_bytes).hexdigest(),
        "rows":int(len(evidence)),
    }
    report_path=outroot/"lf2_1a_report.json"
    report_bytes=deterministic_json(report)
    freeze({evidence_path:evidence_bytes,report_path:report_bytes})

    print("Status:",report["status"])
    print("LightGlue pinned commit:",frontend_meta["pinned_commit"])
    print("LightGlue source mode:",frontend_meta["source_mode"])
    print("LightGlue source root:",frontend_meta["source_root"])
    print("LightGlue resolved git commit:",frontend_meta["resolved_git_commit"])
    print("ALIKED weights SHA256:",frontend_meta["aliked_weights_sha256"])
    print("LightGlue weights SHA256:",frontend_meta["lightglue_weights_sha256"])
    print("Device:",frontend_meta["device"])
    for key,r in trajectory_reports.items():
        print(
            f"{key}: queries={r['queries']} pairs={r['pairs']} "
            f"homography_ok={r['homography_ok']} finite={r['finite_projection']} "
            f"inside={r['inside_projection']} median_matches={r['median_matches']:.1f} "
            f"median_inliers={r['median_inliers']:.1f} "
            f"median_total_ms_pair={r['median_total_ms_pair']:.3f}"
        )
        base=baseline_report["trajectories"][key]
        print(
            f"  vs SuperPoint+LightGlue: "
            f"inside={base['inside_projection']}->{r['inside_projection']} "
            f"median_matches={base['median_matches']:.1f}->{r['median_matches']:.1f} "
            f"median_inliers={base['median_inliers']:.1f}->{r['median_inliers']:.1f} "
            f"median_total_ms_pair={base['median_total_ms_pair']:.3f}"
            f"->{r['median_total_ms_pair']:.3f}"
        )
    print("Reference loaded: False")
    print("Full Top20 benchmark executed: False")
    print("Smoke evidence SHA256:",report["outputs"]["smoke_evidence"]["sha256"])
    print("Report:",report_path)


if __name__=="__main__":
    main()
