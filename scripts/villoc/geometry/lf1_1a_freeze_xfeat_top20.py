#!/usr/bin/env python3
"""LF1.1-A: full reference-free XFeat evidence freeze on frozen QV1.4 Top20."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT=Path(__file__).resolve().parents[3]
SRC_ROOT=REPO_ROOT/"src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0,str(SRC_ROOT))

import numpy as np
import pandas as pd
import yaml

from uavloc.geometry.xfeat_absolute_frontend import (
    DETECTION_THRESHOLD,
    MIN_COSSIM,
    MIN_HOMOGRAPHY_INLIERS,
    PINNED_COMMIT,
    RANSAC_THRESHOLD_PX,
    RESIZE_LONG,
    TOP_K,
    XFeatAbsoluteFrontend,
)

ROOT=REPO_ROOT
DEFAULT_CONFIG=ROOT/"configs/research/learned_geometric_frontend_lf1_1a.yaml"

MANIFEST_REQUIRED={"query_id","image_path","reference_available"}
POOL_REQUIRED={
    "query_id","pool_rank","tile_id","source_view","source_rank",
}
TILE_REQUIRED={
    "tile_id","tile_path","left_easting","right_easting",
    "bottom_northing","top_northing",
}


def resolve(value: str | Path)->Path:
    path=Path(value).expanduser()
    return path if path.is_absolute() else ROOT/path


def sha256_file(path: Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def canonical_csv(frame: pd.DataFrame)->bytes:
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.9f",
    ).encode("utf-8")


def canonical_json(value: dict[str,Any])->bytes:
    return (
        json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n"
    ).encode("utf-8")


def freeze(files: dict[Path,bytes])->None:
    bad=[str(path) for path,data in files.items() if path.exists() and path.read_bytes()!=data]
    if bad:
        raise RuntimeError(
            "Refusing to overwrite differing LF1.1-A freeze: "+", ".join(bad)
        )
    for path,data in files.items():
        if path.exists():
            continue
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)


def require_columns(frame: pd.DataFrame, required: set[str], label: str)->None:
    missing=sorted(required-set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")


def resolve_existing(value: Any, *roots: Path)->Path:
    raw=Path(str(value)).expanduser()
    candidates=[raw] if raw.is_absolute() else [ROOT/raw,*[root/raw for root in roots]]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not resolve file; tried: "+", ".join(map(str,candidates))
    )


def bool_value(value: Any)->bool:
    if isinstance(value,(bool,np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true","1","yes","y","t"}


def finite_median(values: pd.Series)->float | None:
    array=pd.to_numeric(values,errors="coerce").to_numpy(dtype=float)
    array=array[np.isfinite(array)]
    return float(np.median(array)) if len(array) else None


def verify_config(cfg: dict[str,Any])->None:
    if cfg.get("stage")!="LF1.1-A":
        raise RuntimeError("Expected LF1.1-A config")
    x=cfg["xfeat"]
    expected={
        "pinned_commit":PINNED_COMMIT,
        "resize_long":RESIZE_LONG,
        "top_k":TOP_K,
        "detection_threshold":DETECTION_THRESHOLD,
        "min_cossim":MIN_COSSIM,
        "ransac_threshold_px":RANSAC_THRESHOLD_PX,
        "min_homography_inliers":MIN_HOMOGRAPHY_INLIERS,
    }
    for key,value in expected.items():
        if x.get(key)!=value:
            raise RuntimeError(f"XFeat contract changed: {key}")
    generation=cfg["generation"]
    if generation.get("pair_order")!="query_id_then_pool_rank":
        raise RuntimeError("Pair ordering changed")
    if generation.get("runtime_fields_frozen") is not False:
        raise RuntimeError("Runtime fields must not enter frozen evidence")
    if generation.get("retain_failed_pairs") is not True:
        raise RuntimeError("Failed pairs must be retained")
    for key,value in cfg["scientific_boundary"].items():
        if value is not False:
            raise RuntimeError(f"{key} must be false")


def verify_smoke(cfg: dict[str,Any])->dict[str,Any]:
    block=cfg["lf1_0"]
    report_path=resolve(block["report"])
    report=json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status")!="PASS_LF1_0_XFEAT_ABSOLUTE_FRONTEND_PREFLIGHT_SMOKE":
        raise RuntimeError("LF1.0 report not PASS")
    if report["outputs"]["smoke_evidence"]["sha256"]!=block["expected_smoke_sha256"]:
        raise RuntimeError("LF1.0 smoke evidence hash mismatch")
    if report["frontend"]["weights_sha256"]!=block["expected_weights_sha256"]:
        raise RuntimeError("LF1.0 weights hash mismatch")
    if report["frontend"]["resolved_commit"]!=PINNED_COMMIT:
        raise RuntimeError("LF1.0 XFeat commit mismatch")
    boundary=report.get("scientific_boundary",{})
    for key in (
        "reference_loaded",
        "oracle_labels_loaded",
        "full_top20_benchmark_executed",
        "candidate_reranked",
        "selector_executed",
        "threshold_tuned",
        "learned_model_trained",
        "bootstrap_state_modified",
    ):
        if boundary.get(key) is not False:
            raise RuntimeError(f"LF1.0 boundary violation: {key}")
    return report


def load_inputs(cfg: dict[str,Any], key: str)->dict[str,Any]:
    block=cfg["trajectories"][key]
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
    require_columns(pool,POOL_REQUIRED,f"{key} Top20")
    require_columns(tiles,TILE_REQUIRED,f"{key} tile index")

    manifest["query_id"]=pd.to_numeric(manifest["query_id"],errors="raise").astype(int)
    pool["query_id"]=pd.to_numeric(pool["query_id"],errors="raise").astype(int)
    pool["pool_rank"]=pd.to_numeric(pool["pool_rank"],errors="raise").astype(int)
    pool["source_rank"]=pd.to_numeric(pool["source_rank"],errors="raise").astype(int)
    pool["tile_id"]=pool["tile_id"].astype(str)
    tiles["tile_id"]=tiles["tile_id"].astype(str)

    refs=manifest["reference_available"].astype(str).str.strip().str.lower()
    if not refs.isin(["false","0","no","n","f"]).all():
        raise RuntimeError(f"{key} manifest exposes reference")
    if len(manifest)!=int(block["expected_queries"]) or manifest["query_id"].duplicated().any():
        raise RuntimeError(f"{key} manifest count/uniqueness mismatch")
    if len(pool)!=int(block["expected_pairs"]):
        raise RuntimeError(f"{key} Top20 pair count mismatch")
    if pool[["query_id","tile_id"]].duplicated().any():
        raise RuntimeError(f"{key} duplicate query/tile candidate")
    if set(pool["query_id"])!=set(manifest["query_id"]):
        raise RuntimeError(f"{key} query set mismatch")

    for qid,group in pool.groupby("query_id",sort=True):
        ordered=group.sort_values("pool_rank")
        if len(ordered)!=20 or ordered["pool_rank"].tolist()!=list(range(1,21)):
            raise RuntimeError(f"{key} q{qid} Top20 rank contract changed")
        top1=ordered.iloc[0]
        if str(top1["source_view"])!="center_square" or int(top1["source_rank"])!=1:
            raise RuntimeError(f"{key} q{qid} Top1 contract changed")

    missing_tiles=sorted(set(pool["tile_id"])-set(tiles["tile_id"]))
    if missing_tiles:
        raise RuntimeError(f"{key} missing tile IDs: {missing_tiles[:20]}")

    return {
        "spec_path":spec_path,
        "manifest_path":manifest_path,
        "pool_path":pool_path,
        "tile_index_path":tile_index_path,
        "manifest":manifest,
        "pool":pool.sort_values(["query_id","pool_rank"],kind="mergesort").reset_index(drop=True),
        "tiles":tiles,
        "trajectory_id":str(spec["trajectory"]["id"]),
    }


def preflight_paths(inputs: dict[str,Any])->tuple[dict[int,Path],dict[str,Path]]:
    query_paths={}
    for row in inputs["manifest"].itertuples(index=False):
        query_paths[int(row.query_id)]=resolve_existing(row.image_path)
    tile_lookup=inputs["tiles"].set_index("tile_id")
    tile_paths={}
    for tile_id in sorted(set(inputs["pool"]["tile_id"])):
        tile_paths[tile_id]=resolve_existing(
            tile_lookup.loc[tile_id,"tile_path"],
            inputs["tile_index_path"].parent,
        )
    return query_paths,tile_paths


def freeze_value(value: Any)->Any:
    if isinstance(value,(bool,np.bool_)):
        return bool(value)
    if isinstance(value,(int,np.integer)):
        return int(value)
    if isinstance(value,(float,np.floating)):
        return float(value)
    return value


def generate(
    frontend: XFeatAbsoluteFrontend,
    inputs: dict[str,Any],
    query_paths: dict[int,Path],
    tile_paths: dict[str,Path],
    *,
    progress_every: int,
)->tuple[pd.DataFrame,dict[str,Any],dict[str,float]]:
    tile_lookup=inputs["tiles"].set_index("tile_id")
    rows=[]
    observed_feature_ms=[]
    observed_match_ms=[]
    observed_geometry_ms=[]
    observed_total_ms=[]
    wall_start=time.perf_counter()

    pool=inputs["pool"]
    for index,row in enumerate(pool.itertuples(index=False),start=1):
        if index==1 or index%max(1,progress_every)==0 or index==len(pool):
            print(
                f"{inputs['trajectory_id']} XFeat Top20 pair {index}/{len(pool)}",
                flush=True,
            )
        qid=int(row.query_id)
        tile_id=str(row.tile_id)
        tile=tile_lookup.loc[tile_id]
        measurement=frontend.measure_pair(
            query_paths[qid],
            tile_paths[tile_id],
            tile_bounds=(
                float(tile["left_easting"]),
                float(tile["right_easting"]),
                float(tile["bottom_northing"]),
                float(tile["top_northing"]),
            ),
        )
        observed_feature_ms.append(float(measurement.runtime.feature_ms))
        observed_match_ms.append(float(measurement.runtime.match_ms))
        observed_geometry_ms.append(float(measurement.runtime.geometry_ms))
        observed_total_ms.append(float(measurement.runtime.total_ms))

        out={
            "trajectory_id":inputs["trajectory_id"],
            "query_id":qid,
            "pool_rank":int(row.pool_rank),
            "tile_id":tile_id,
            "source_view":str(row.source_view),
            "source_rank":int(row.source_rank),
            "frontend_name":measurement.frontend_name,
            "good_matches":int(measurement.good_matches),
            "inliers":int(measurement.inliers),
            "homography_ok":bool(measurement.homography_ok),
            "finite_projection":bool(measurement.finite_projection),
            "projected_easting":float(measurement.projected_easting),
            "projected_northing":float(measurement.projected_northing),
            "projected_inside_tile":bool(measurement.projected_inside_tile),
        }
        for name,value in sorted(measurement.evidence.items()):
            if name in {"query_feature_cache_hit","tile_feature_cache_hit"}:
                continue
            out[f"evidence_{name}"]=freeze_value(value)
        rows.append(out)

    wall_s=float(time.perf_counter()-wall_start)
    frame=pd.DataFrame(rows).sort_values(
        ["query_id","pool_rank"],kind="mergesort"
    ).reset_index(drop=True)

    h=frame["homography_ok"].map(bool_value)
    finite=frame["finite_projection"].map(bool_value)
    inside=frame["projected_inside_tile"].map(bool_value)
    h_per_query=h.groupby(frame["query_id"]).sum()

    summary={
        "queries":int(frame["query_id"].nunique()),
        "pairs":int(len(frame)),
        "unique_tiles":int(frame["tile_id"].nunique()),
        "homography_ok":int(h.sum()),
        "finite_projection":int(finite.sum()),
        "inside_projection":int(inside.sum()),
        "queries_any_homography":int((h_per_query>0).sum()),
        "queries_all20_homographies":int((h_per_query==20).sum()),
        "queries_no_homography":int((h_per_query==0).sum()),
        "median_matches":finite_median(frame["good_matches"]),
        "median_inliers":finite_median(frame["inliers"]),
    }
    runtime={
        "wall_s":wall_s,
        "feature_compute_ms_sum":float(np.sum(observed_feature_ms)),
        "match_ms_sum":float(np.sum(observed_match_ms)),
        "geometry_ms_sum":float(np.sum(observed_geometry_ms)),
        "median_total_ms_pair":float(np.median(observed_total_ms)),
        "mean_total_ms_pair":float(np.mean(observed_total_ms)),
    }
    return frame,summary,runtime


def main()->None:
    parser=argparse.ArgumentParser(
        description="LF1.1-A full reference-free XFeat Top20 evidence freeze"
    )
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    args=parser.parse_args()

    cfg_path=resolve(args.config)
    cfg=yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verify_config(cfg)
    verify_smoke(cfg)

    prepared={}
    for key in ("development","blind_stress"):
        inputs=load_inputs(cfg,key)
        qpaths,tpaths=preflight_paths(inputs)
        prepared[key]=(inputs,qpaths,tpaths)
        print(
            f"{key} preflight: queries={len(inputs['manifest'])} "
            f"pairs={len(inputs['pool'])} unique_tiles={len(tpaths)} "
            "reference_loaded=False"
        )

    frontend=XFeatAbsoluteFrontend(
        resolve(cfg["xfeat"]["repo"]),
        device_name=str(cfg["xfeat"]["device"]),
    )
    frontend_meta=frontend.describe()
    if frontend_meta["weights_sha256"]!=cfg["lf1_0"]["expected_weights_sha256"]:
        raise RuntimeError("XFeat weights changed since LF1.0")

    files={}
    trajectory_reports={}
    runtime_observed={}
    progress=int(cfg["generation"]["progress_every_pairs"])

    for key in ("development","blind_stress"):
        inputs,qpaths,tpaths=prepared[key]
        frame,summary,runtime=generate(
            frontend,
            inputs,
            qpaths,
            tpaths,
            progress_every=progress,
        )
        out=resolve(cfg["output_root"])/key/"lf1_1a_xfeat_top20_evidence.csv"
        payload=canonical_csv(frame)
        files[out]=payload
        trajectory_reports[key]={
            **summary,
            "output":{
                "path":str(out.relative_to(ROOT)),
                "sha256":hashlib.sha256(payload).hexdigest(),
                "rows":int(len(frame)),
            },
            "provenance":{
                "spec_sha256":sha256_file(inputs["spec_path"]),
                "manifest_sha256":sha256_file(inputs["manifest_path"]),
                "top20_sha256":sha256_file(inputs["pool_path"]),
                "tile_index_sha256":sha256_file(inputs["tile_index_path"]),
            },
        }
        runtime_observed[key]=runtime

    report={
        "stage":"LF1.1-A",
        "status":"PASS_LF1_1A_REFERENCE_FREE_XFEAT_TOP20_FREEZE",
        "frontend":frontend_meta,
        "trajectories":trajectory_reports,
        "scientific_boundary":{
            "reference_loaded":False,
            "oracle_labels_loaded":False,
            "candidate_reranked":False,
            "selector_executed":False,
            "threshold_tuned":False,
            "model_trained":False,
            "temporal_fusion_executed":False,
            "bootstrap_state_modified":False,
            "runtime_fields_frozen":False,
        },
        "interpretation":{
            "full_frozen_top20_processed":True,
            "failed_pairs_retained":True,
            "no_accuracy_claim":True,
            "runtime_printed_not_hashed":True,
        },
        "provenance":{
            "config_sha256":sha256_file(cfg_path),
            "script_sha256":sha256_file(Path(__file__).resolve()),
            "lf1_0_report_sha256":sha256_file(resolve(cfg["lf1_0"]["report"])),
            "xfeat_weights_sha256":frontend_meta["weights_sha256"],
            "xfeat_commit":frontend_meta["resolved_commit"],
        },
    }
    report_path=resolve(cfg["output_root"])/"lf1_1a_freeze_report.json"
    files[report_path]=canonical_json(report)
    freeze(files)

    print("Status:",report["status"])
    print("XFeat commit:",frontend_meta["resolved_commit"])
    print("XFeat weights SHA256:",frontend_meta["weights_sha256"])
    print("Device:",frontend_meta["device"])
    for key in ("development","blind_stress"):
        r=trajectory_reports[key]
        rt=runtime_observed[key]
        print(
            f"{key}: queries={r['queries']} pairs={r['pairs']} "
            f"unique_tiles={r['unique_tiles']} homography_ok={r['homography_ok']} "
            f"finite={r['finite_projection']} inside={r['inside_projection']} "
            f"queries_any_h={r['queries_any_homography']} "
            f"queries_all20_h={r['queries_all20_homographies']} "
            f"queries_no_h={r['queries_no_homography']} "
            f"median_matches={r['median_matches']:.3f} "
            f"median_inliers={r['median_inliers']:.3f}"
        )
        print("  evidence SHA256:",r["output"]["sha256"])
        print(
            f"  observed runtime (not frozen): wall_s={rt['wall_s']:.3f} "
            f"feature_compute_s={rt['feature_compute_ms_sum']/1000.0:.3f} "
            f"match_s={rt['match_ms_sum']/1000.0:.3f} "
            f"geometry_s={rt['geometry_ms_sum']/1000.0:.3f} "
            f"median_total_ms_pair={rt['median_total_ms_pair']:.3f}"
        )
    print("Reference loaded: False")
    print("Candidate reranked/selected: False")
    print("Accuracy claim: False")
    print("Freeze report:",report_path)


if __name__=="__main__":
    main()
