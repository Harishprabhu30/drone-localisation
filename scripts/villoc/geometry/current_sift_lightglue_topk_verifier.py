#!/usr/bin/env python3
"""Generic reference-free SIFT+LightGlue Top-K verifier for the current stack.

Inputs:
- reference-free query manifest
- DINO Top-K candidate CSV
- georeferenced tile index

Outputs under <run-root>/absolute/:
- geometric_candidate_evidence.csv
- selected_absolute_observations.csv
- absolute_query_summary.csv
- verifier_report.json

The query summary includes neutral canonical names plus legacy compatibility
columns required by older export/visualization helpers. New code should use the
neutral columns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parents[3]
SRC_ROOT=REPO_ROOT/"src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0,str(SRC_ROOT))

import numpy as np
import pandas as pd
import yaml

from uavloc.geometry.sift_lightglue_absolute_frontend import (
    PINNED_COMMIT, PINNED_LIGHTGLUE_BLOB, PINNED_SIFT_BLOB,
    SIFTLightGlueAbsoluteFrontend,
)

FORBIDDEN_EXACT={
    "latitude","longitude","lat","lon","reference_x_m","reference_y_m",
    "ground_truth_x_m","ground_truth_y_m","gt_easting","gt_northing",
    "oracle_tile_id","error_m","rmse","p95",
}
FORBIDDEN_PART=("hit_le_","ground_truth","oracle_","gt_error","reference_enu","postfreeze","eval_error")
TILE_REQUIRED={
    "tile_id","tile_path","left_easting","right_easting",
    "bottom_northing","top_northing",
}


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def assert_reference_free(df,label):
    bad=[]
    for c in df.columns:
        low=str(c).strip().lower()
        if low in FORBIDDEN_EXACT or any(x in low for x in FORBIDDEN_PART):
            bad.append(str(c))
    if bad:
        raise RuntimeError(f"{label} contains forbidden evaluation columns: {sorted(bad)}")
    if "reference_used" in df:
        used=df["reference_used"].astype(str).str.lower().isin({"true","1","yes","y","t"})
        if used.any():
            raise RuntimeError(f"{label}: reference_used=True")


def normid(s):
    return s.astype(str).str.strip().str.replace(r"\.0$","",regex=True)


def resolve_image(v,roots):
    p=Path(str(v)).expanduser()
    candidates=[p] if p.is_absolute() else [REPO_ROOT/p,*[r/p for r in roots]]
    for c in candidates:
        if c.is_file(): return c.resolve()
    raise FileNotFoundError(str(v))


def bool_value(v):
    return str(v).strip().lower() in {"true","1","yes","y","t"}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--query-csv",type=Path,required=True)
    ap.add_argument("--topk-csv",type=Path,required=True)
    ap.add_argument("--tile-index-csv",type=Path,required=True)
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--top-n",type=int,default=20)
    ap.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    args=ap.parse_args()

    started=time.perf_counter()
    cfg=yaml.safe_load(args.config.resolve().read_text())
    abs_cfg=cfg.get("absolute_localization",{})
    sift_cfg=abs_cfg.get("verifier",{})
    if sift_cfg.get("method")!="sift_lightglue":
        raise RuntimeError("Current verifier expects absolute_localization.verifier.method=sift_lightglue")
    if int(args.top_n)!=int(cfg["retrieval"]["top_k"]):
        raise RuntimeError("Verifier --top-n must equal retrieval.top_k")

    qpath=args.query_csv.resolve(); topk_path=args.topk_csv.resolve()
    tile_path=args.tile_index_csv.resolve(); run=args.run_root.resolve()
    for p in (qpath,topk_path,tile_path):
        if not p.is_file(): raise FileNotFoundError(p)

    q=pd.read_csv(qpath); topk=pd.read_csv(topk_path); tiles=pd.read_csv(tile_path)
    assert_reference_free(q,"query manifest")
    assert_reference_free(topk,"DINO Top-K")
    qid_col="query_id" if "query_id" in q.columns else "token0_id"
    if qid_col not in q or "image_path" not in q:
        raise RuntimeError("Query manifest requires query_id/token0_id and image_path")
    if {"query_id","rank","tile_id","score"}-set(topk):
        raise RuntimeError("Top-K schema changed")
    missing=TILE_REQUIRED-set(tiles)
    if missing: raise RuntimeError(f"Tile index missing {sorted(missing)}")

    q=q[[qid_col,"image_path"]].rename(columns={qid_col:"query_id"}).copy()
    q["query_id"]=normid(q["query_id"])
    topk["query_id"]=normid(topk["query_id"]); topk["tile_id"]=normid(topk["tile_id"])
    tiles["tile_id"]=normid(tiles["tile_id"])
    topk["rank"]=pd.to_numeric(topk["rank"],errors="raise").astype(int)
    topk=topk[topk["rank"]<=int(args.top_n)].copy()
    counts=topk.groupby("query_id").size()
    if not (counts==int(args.top_n)).all():
        raise RuntimeError(f"Each query must have exactly Top-{args.top_n} rows")
    if q["query_id"].duplicated().any(): raise RuntimeError("Duplicate query IDs")

    work=(topk.merge(q,on="query_id",validate="many_to_one")
          .merge(tiles[list(TILE_REQUIRED)],on="tile_id",validate="many_to_one"))
    if len(work)!=len(topk): raise RuntimeError("Candidate join changed row count")

    lg_repo=REPO_ROOT/str(sift_cfg.get("repo","third_party/LightGlue"))
    frontend=SIFTLightGlueAbsoluteFrontend(lg_repo,device_name=args.device)
    desc=frontend.describe()

    expected={
        "commit":str(sift_cfg.get("expected_commit",PINNED_COMMIT)),
        "sift_blob":str(sift_cfg.get("expected_sift_blob",PINNED_SIFT_BLOB)),
        "lightglue_blob":str(sift_cfg.get("expected_lightglue_blob",PINNED_LIGHTGLUE_BLOB)),
        "weights":str(sift_cfg.get("expected_weights_sha256",desc["lightglue_weights_sha256"])),
    }
    if desc["pinned_commit"]!=expected["commit"] or PINNED_COMMIT!=expected["commit"]:
        raise RuntimeError("LightGlue commit mismatch")
    if PINNED_SIFT_BLOB!=expected["sift_blob"] or PINNED_LIGHTGLUE_BLOB!=expected["lightglue_blob"]:
        raise RuntimeError("LightGlue source blob mismatch")
    if desc["lightglue_weights_sha256"]!=expected["weights"]:
        raise RuntimeError("LightGlue weights SHA mismatch")

    rows=[]; runtimes=[]; total=len(work)
    for i,row in enumerate(work.itertuples(index=False),1):
        if i==1 or i%100==0 or i==total:
            print(f"SIFT+LightGlue pair {i}/{total}",flush=True)
        qp=resolve_image(row.image_path,[run,qpath.parent])
        tp=resolve_image(row.tile_path,[tile_path.parent])
        m=frontend.measure_pair(
            qp,tp,
            tile_bounds=(
                float(row.left_easting),float(row.right_easting),
                float(row.bottom_northing),float(row.top_northing),
            ),
        )
        e=m.evidence; runtimes.append(float(m.runtime.total_ms))
        rows.append({
            "query_id":str(row.query_id),
            "rank":int(row.rank),
            "tile_id":str(row.tile_id),
            "score":float(row.score),
            "query_image":str(qp),
            "tile_path":str(tp),
            "left_easting":float(row.left_easting),
            "right_easting":float(row.right_easting),
            "bottom_northing":float(row.bottom_northing),
            "top_northing":float(row.top_northing),
            "matches":int(m.good_matches),
            "inliers":int(m.inliers),
            "inlier_ratio":float(e["inlier_ratio"]),
            "homography_ok":bool(m.homography_ok),
            "finite_projection":bool(m.finite_projection),
            "projected_inside_tile":bool(m.projected_inside_tile),
            "projected_tile_u_px":float(m.projected_tile_u_px),
            "projected_tile_v_px":float(m.projected_tile_v_px),
            "projected_easting":float(m.projected_easting),
            "projected_northing":float(m.projected_northing),
            "match_score_sum":float(e["match_score_sum"]),
            "match_score_mean":float(e["match_score_mean"]),
            "match_score_median":float(e["match_score_median"]),
            "symmetric_error_median_px":float(e["symmetric_error_median_px"]),
            "jacobian_anisotropy":float(e["jacobian_anisotropy"]),
            "footprint_area_fraction_tile":float(e["footprint_area_fraction_tile"]),
        })

    pair=pd.DataFrame(rows).sort_values(["query_id","rank"],kind="mergesort").reset_index(drop=True)
    assert_reference_free(pair,"SIFT pair evidence")
    ranked=pair.sort_values(
        ["query_id","inliers","rank","tile_id"],
        ascending=[True,False,True,True],
        kind="mergesort",
    ).copy()
    ranked["verifier_rank"]=ranked.groupby("query_id").cumcount()+1

    selected=[]; qsummary=[]
    for qid,g in ranked.groupby("query_id",sort=True):
        g=g.reset_index(drop=True); w=g.iloc[0]; runner=g.iloc[1]
        dino1=g.loc[g["rank"]==1].iloc[0]
        margin=int(w["inliers"]-runner["inliers"])
        selected.append({
            "query_id":str(qid),
            "selected_tile_id":str(w["tile_id"]),
            "selected_dino_rank":int(w["rank"]),
            "selected_dino_score":float(w["score"]),
            "selected_matches":int(w["matches"]),
            "selected_inliers":int(w["inliers"]),
            "selected_inlier_ratio":float(w["inlier_ratio"]),
            "runner_up_inliers":int(runner["inliers"]),
            "inlier_margin":margin,
            "selected_homography_ok":bool(w["homography_ok"]),
            "selected_finite_projection":bool(w["finite_projection"]),
            "selected_projected_inside_tile":bool(w["projected_inside_tile"]),
            "selected_projected_easting":float(w["projected_easting"]),
            "selected_projected_northing":float(w["projected_northing"]),
        })
        qsummary.append({
            "query_id":str(qid),
            "dino_top1_tile_id":str(dino1["tile_id"]),
            "dino_top1_score":float(dino1["score"]),
            "absolute_selected_tile_id":str(w["tile_id"]),
            "absolute_verifier":"sift_lightglue",
            "absolute_selected_dino_rank":int(w["rank"]),
            "absolute_selected_dino_score":float(w["score"]),
            "absolute_selected_matches":int(w["matches"]),
            "absolute_selected_inliers":int(w["inliers"]),
            "absolute_selected_inlier_ratio":float(w["inlier_ratio"]),
            "absolute_inlier_margin":margin,
            "absolute_confidence_score":float(w["inliers"]),
            # Legacy compatibility only. New code must use absolute_* names.
            "original_top1_tile_id":str(dino1["tile_id"]),
            "original_top1_dino_score":float(dino1["score"]),
            "reranked_top1_tile_id":str(w["tile_id"]),
            "reranked_top1_verifier_score":float(w["inliers"]),
            "reranked_top1_hybrid_score":float(w["inliers"]),
            "reranked_top1_inliers":int(w["inliers"]),
        })

    selected=pd.DataFrame(selected); qsummary=pd.DataFrame(qsummary)
    out=run/"absolute"; out.mkdir(parents=True,exist_ok=True)
    pair_path=out/"geometric_candidate_evidence.csv"
    selected_path=out/"selected_absolute_observations.csv"
    summary_path=out/"absolute_query_summary.csv"
    report_path=out/"verifier_report.json"
    ranked.to_csv(pair_path,index=False)
    selected.to_csv(selected_path,index=False)
    qsummary.to_csv(summary_path,index=False)

    report={
        "stage":"CURRENT_STACK_SIFT_LIGHTGLUE_TOPK",
        "status":"PASS_CURRENT_STACK_SIFT_LIGHTGLUE_TOPK",
        "reference_free":True,
        "queries":int(selected["query_id"].nunique()),
        "candidate_pairs":int(len(ranked)),
        "top_k":int(args.top_n),
        "selected_homography_ok":int(selected["selected_homography_ok"].sum()),
        "selected_finite_projection":int(selected["selected_finite_projection"].sum()),
        "median_selected_matches":float(selected["selected_matches"].median()),
        "median_selected_inliers":float(selected["selected_inliers"].median()),
        "median_selected_dino_rank":float(selected["selected_dino_rank"].median()),
        "median_inlier_margin":float(selected["inlier_margin"].median()),
        "runtime":{
            "stage_wall_s":float(time.perf_counter()-started),
            "median_pair_total_ms":float(np.median(runtimes)) if runtimes else None,
        },
        "provenance":{
            "config_sha256":sha256(args.config.resolve()),
            "query_manifest_sha256":sha256(qpath),
            "topk_sha256":sha256(topk_path),
            "tile_index_sha256":sha256(tile_path),
            "pair_evidence_sha256":sha256(pair_path),
            "selected_observations_sha256":sha256(selected_path),
            "query_summary_sha256":sha256(summary_path),
            "lightglue":desc,
        },
        "compatibility_note":"legacy reranked_* columns exist only for old downstream exporters/visuals; canonical names are absolute_*",
    }
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False))
    print("STATUS: PASS_CURRENT_STACK_SIFT_LIGHTGLUE_TOPK")
    print("queries/pairs:",f"{report['queries']}/{report['candidate_pairs']}")
    print("selected homography/finite:",f"{report['selected_homography_ok']}/{report['selected_finite_projection']}")
    print("median selected matches/inliers:",f"{report['median_selected_matches']:.1f}/{report['median_selected_inliers']:.1f}")
    print("pair evidence:",pair_path)
    print("selected observations:",selected_path)


if __name__=="__main__":
    main()
