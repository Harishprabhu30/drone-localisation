#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, math, sys, time
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
    PINNED_COMMIT, PINNED_LIGHTGLUE_BLOB, PINNED_SIFT_BLOB,
    SIFTLightGlueAbsoluteFrontend,
)

ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/blind_demo_sift_lightglue_transfer_bv1.yaml"
FORBIDDEN_EXACT = {
    "latitude","longitude","lat","lon","reference_x_m","reference_y_m",
    "ground_truth_x","ground_truth_y","ground_truth_x_m","ground_truth_y_m",
    "oracle_tile_id","oracle_best_tile_id","candidate_body_error_m",
    "reranked_top1_error_m","gt_easting","gt_northing","projected_error_m",
    "error_m","rmse","p95","final_error",
}
FORBIDDEN_PART = ("hit_le_","ground_truth","oracle_","gt_error","reference_enu","postfreeze","eval_error")
PROTOCOL_KEYS=("model_name","image_size","crop_mode","pooling","normalization","l2_normalize","descriptor_dtype")
TILE_REQUIRED={"tile_id","tile_path","left_easting","right_easting","bottom_northing","top_northing"}

def resolve(v):
    p=Path(v).expanduser()
    return p if p.is_absolute() else ROOT/p

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def canonical_csv(df):
    return df.to_csv(index=False,lineterminator="\n",float_format="%.9f").encode()

def safe(v):
    if isinstance(v,dict): return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [safe(x) for x in v]
    if isinstance(v,np.bool_): return bool(v)
    if isinstance(v,np.integer): return int(v)
    if isinstance(v,np.floating): v=float(v)
    if isinstance(v,float): return v if math.isfinite(v) else None
    return v

def canonical_json(v):
    return (json.dumps(safe(v),indent=2,sort_keys=True,allow_nan=False)+"\n").encode()

def freeze(files):
    bad=[str(p) for p,d in files.items() if p.exists() and p.read_bytes()!=d]
    if bad: raise RuntimeError("Refusing to overwrite differing BV1.0 outputs: "+", ".join(bad))
    for p,d in files.items():
        if not p.exists():
            p.parent.mkdir(parents=True,exist_ok=True)
            with p.open("xb") as f: f.write(d)

def assert_blind_safe(df,label):
    bad=[]
    for c in df.columns:
        low=str(c).strip().lower()
        if low in FORBIDDEN_EXACT or any(x in low for x in FORBIDDEN_PART): bad.append(str(c))
    bad=[x for x in bad if x!="reference_used"]
    if bad: raise RuntimeError(f"{label} contains forbidden columns: {sorted(bad)}")
    if "reference_used" in df:
        if df["reference_used"].astype(str).str.lower().isin({"true","1","yes","y","t"}).any():
            raise RuntimeError(f"{label}: reference_used=True")

def load_cache(path):
    d=np.load(path,allow_pickle=False)
    if {"descriptors","ids","meta_json"}-set(d.files): raise RuntimeError(f"Invalid cache: {path}")
    meta=d["meta_json"]
    if hasattr(meta,"item"): meta=meta.item()
    return {"descriptors":d["descriptors"].astype(np.float32),"ids":d["ids"].astype(str),"meta":json.loads(str(meta))}

def sig(meta): return {k:meta.get("protocol",{}).get(k) for k in PROTOCOL_KEYS}

def verify_cache_pair(q,m):
    qd,md=q["descriptors"],m["descriptors"]
    if qd.ndim!=2 or md.ndim!=2 or qd.shape[1]!=md.shape[1]: raise RuntimeError("DINO cache shape mismatch")
    if qd.shape[0]!=len(q["ids"]) or md.shape[0]!=len(m["ids"]): raise RuntimeError("DINO cache row mismatch")
    if len(set(q["ids"]))!=len(q["ids"]) or len(set(m["ids"]))!=len(m["ids"]): raise RuntimeError("Duplicate DINO IDs")
    if not np.isfinite(qd).all() or not np.isfinite(md).all(): raise RuntimeError("Non-finite DINO cache")
    if sig(q["meta"])!=sig(m["meta"]): raise RuntimeError(f"DINO protocol mismatch: query={sig(q['meta'])} map={sig(m['meta'])}")
    if not np.allclose(np.linalg.norm(qd,axis=1),1,atol=1e-3): raise RuntimeError("Query cache not L2 normalized")
    if not np.allclose(np.linalg.norm(md,axis=1),1,atol=1e-3): raise RuntimeError("Map cache not L2 normalized")

def normid(s): return s.astype(str).str.strip().str.replace(r"\.0$","",regex=True)

def resolve_image(v,roots):
    p=Path(str(v)).expanduser()
    candidates=[p] if p.is_absolute() else [ROOT/p,*[r/p for r in roots]]
    for c in candidates:
        if c.is_file(): return c.resolve()
    raise FileNotFoundError(str(v))

def verify_config(c):
    if c.get("stage")!="BV1.0": raise RuntimeError("Expected BV1.0")
    if c["map"]["variant"]!="768_s256": raise RuntimeError("BV1.0 must use 768_s256")
    if int(c["retrieval"]["top_k"])!=20 or int(c["retrieval"]["expected_pairs"])!=2460: raise RuntimeError("Top20 contract changed")
    if c["sift_lightglue"]["device"]!="cpu": raise RuntimeError("BV1.0 evidence device must be CPU")
    b=c["scientific_boundary"]
    for k in ("source_blind_run_immutable","map_variant_change_from_demo_declared","candidate_pool_change_from_demo_declared"):
        if b[k] is not True: raise RuntimeError(f"{k} must be true")
    for k in ("reference_allowed","srt_allowed","gps_allowed","oracle_allowed","evaluation_metrics_allowed",
              "sift_parameter_tuning_allowed","bootstrap_executed","fusion_executed",
              "postfreeze_truth_attachment_allowed_in_stage","oblique_experiment_allowed_in_stage"):
        if b[k] is not False: raise RuntimeError(f"{k} must be false")

def build_pool(q,m,k):
    sim=q["descriptors"]@m["descriptors"].T
    order=np.argsort(-sim,axis=1)[:,:k]
    rows=[]
    for i,qid in enumerate(q["ids"]):
        for rank,j in enumerate(order[i],1):
            rows.append({"variant":"768_s256","query_id":str(qid),"rank":rank,"tile_id":str(m["ids"][j]),"score":float(sim[i,j])})
    return pd.DataFrame(rows)

def select_winners(frame):
    x=frame.copy()
    x["query_id"]=normid(x["query_id"]); x["tile_id"]=normid(x["tile_id"])
    x["rank"]=pd.to_numeric(x["rank"]).astype(int); x["inliers"]=pd.to_numeric(x["inliers"]).astype(int)
    x=x.sort_values(["query_id","inliers","rank","tile_id"],ascending=[True,False,True,True],kind="mergesort")
    rows=[]
    for qid,g in x.groupby("query_id",sort=False):
        g=g.reset_index(drop=True); w=g.iloc[0]; r=g.iloc[1]
        rows.append({
            "query_id":str(qid),"selected_dino_rank":int(w["rank"]),"selected_tile_id":str(w["tile_id"]),
            "selected_dino_score":float(w["score"]),"selected_matches":int(w["matches"]),
            "selected_inliers":int(w["inliers"]),"runner_up_inliers":int(r["inliers"]),
            "inlier_margin":int(w["inliers"]-r["inliers"]),"selected_homography_ok":bool(w["homography_ok"]),
            "selected_finite_projection":bool(w["finite_projection"]),
            "selected_projected_inside_tile":bool(w["projected_inside_tile"]),
            "selected_projected_easting":w["projected_easting"],"selected_projected_northing":w["projected_northing"],
        })
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    args=ap.parse_args()
    started=time.perf_counter()
    cfg_path=resolve(args.config).resolve(); cfg=yaml.safe_load(cfg_path.read_text()); verify_config(cfg)
    s=cfg["source_demo"]; run=resolve(s["run_root"]).resolve()
    manifest_path=run/s["blind_manifest"]; qcache_path=run/s["query_cache"]; oldfreeze_path=run/s["blind_freeze"]
    source_cfg_path=resolve(s["config"]).resolve(); tile_index_path=resolve(cfg["map"]["tile_index"]).resolve()
    map_cache_path=resolve(cfg["map"]["descriptor_cache"]).resolve(); lg_repo=resolve(cfg["sift_lightglue"]["repo"]).resolve()
    for p in (manifest_path,qcache_path,oldfreeze_path,source_cfg_path,tile_index_path,map_cache_path):
        if not p.is_file(): raise FileNotFoundError(p)

    source_cfg=yaml.safe_load(source_cfg_path.read_text())
    if source_cfg["blind"]["enabled"] is not True or source_cfg["streams"]["V"]["reference_available"] is not False:
        raise RuntimeError("Historical demo blind contract changed")
    old=json.loads(oldfreeze_path.read_text())
    if old.get("status")!="PASS_BLIND_SUBMISSION_FROZEN": raise RuntimeError("Historical demo freeze not PASS")
    if old.get("localization_state")!=s["expected_localization_state"]: raise RuntimeError("Historical state changed")
    if old["submission"]["sha256"]!=s["expected_submission_sha256"]: raise RuntimeError("Historical submission SHA changed")

    manifest=pd.read_csv(manifest_path); assert_blind_safe(manifest,"blind manifest")
    if len(manifest)!=int(s["expected_queries"]): raise RuntimeError("Historical query count changed")
    qcache=load_cache(qcache_path); mcache=load_cache(map_cache_path); verify_cache_pair(qcache,mcache)
    if len(qcache["ids"])!=int(s["expected_queries"]): raise RuntimeError("Query cache count changed")
    if set(normid(manifest["query_id"]))!=set(normid(pd.Series(qcache["ids"]))): raise RuntimeError("Manifest/cache IDs differ")

    tiles=pd.read_csv(tile_index_path)
    missing=TILE_REQUIRED-set(tiles.columns)
    if missing: raise RuntimeError(f"Tile index missing {sorted(missing)}")
    tiles["tile_id"]=normid(tiles["tile_id"])
    if set(tiles["tile_id"])!=set(normid(pd.Series(mcache["ids"]))): raise RuntimeError("Map cache/tile index IDs differ")

    frontend=SIFTLightGlueAbsoluteFrontend(lg_repo,device_name="cpu"); desc=frontend.describe()
    if desc["pinned_commit"]!=cfg["sift_lightglue"]["expected_commit"] or PINNED_COMMIT!=cfg["sift_lightglue"]["expected_commit"]:
        raise RuntimeError("LightGlue commit mismatch")
    if PINNED_SIFT_BLOB!=cfg["sift_lightglue"]["expected_sift_blob"] or PINNED_LIGHTGLUE_BLOB!=cfg["sift_lightglue"]["expected_lightglue_blob"]:
        raise RuntimeError("LightGlue source blob mismatch")
    if desc["lightglue_weights_sha256"]!=cfg["sift_lightglue"]["expected_weights_sha256"]:
        raise RuntimeError("LightGlue weights SHA mismatch")

    print("Preflight: PASS")
    print("Historical blind run immutable: PASS")
    print("Reference/SRT/GPS loaded: False")
    print("Map variant: 768_s256")
    print("Map tiles:",len(tiles))

    pool=build_pool(qcache,mcache,int(cfg["retrieval"]["top_k"]))
    if len(pool)!=2460 or not (pool.groupby("query_id").size()==20).all(): raise RuntimeError("Top20 pool shape changed")
    q=manifest[["query_id","image_path"]].copy(); q["query_id"]=normid(q["query_id"])
    work=pool.copy(); work["query_id"]=normid(work["query_id"]); work["tile_id"]=normid(work["tile_id"])
    work=work.merge(q,on="query_id",how="left",validate="many_to_one").merge(tiles[list(TILE_REQUIRED)],on="tile_id",how="left",validate="many_to_one")
    if work[["image_path","tile_path","left_easting","right_easting","bottom_northing","top_northing"]].isna().any().any():
        raise RuntimeError("Candidate/query/tile join incomplete")

    rows=[]; runtimes=[]
    for i,row in enumerate(work.itertuples(index=False),1):
        if i==1 or i%100==0 or i==2460: print(f"SIFT+LightGlue pair {i}/2460",flush=True)
        qp=resolve_image(row.image_path,[run]); tp=resolve_image(row.tile_path,[tile_index_path.parent])
        m=frontend.measure_pair(qp,tp,tile_bounds=(float(row.left_easting),float(row.right_easting),float(row.bottom_northing),float(row.top_northing)))
        e=m.evidence; runtimes.append(float(m.runtime.total_ms))
        rows.append({
            "variant":"768_s256","query_id":str(row.query_id),"rank":int(row.rank),"tile_id":str(row.tile_id),"score":float(row.score),
            "query_image":str(qp),"tile_path":str(tp),"left_easting":float(row.left_easting),"right_easting":float(row.right_easting),
            "bottom_northing":float(row.bottom_northing),"top_northing":float(row.top_northing),
            "matches":int(m.good_matches),"inliers":int(m.inliers),"inlier_ratio":float(e["inlier_ratio"]),
            "homography_ok":bool(m.homography_ok),"finite_projection":bool(m.finite_projection),
            "projected_inside_tile":bool(m.projected_inside_tile),"projected_tile_u_px":float(m.projected_tile_u_px),
            "projected_tile_v_px":float(m.projected_tile_v_px),"projected_easting":float(m.projected_easting),
            "projected_northing":float(m.projected_northing),"match_score_sum":float(e["match_score_sum"]),
            "match_score_mean":float(e["match_score_mean"]),"match_score_median":float(e["match_score_median"]),
            "symmetric_error_median_px":float(e["symmetric_error_median_px"]),"jacobian_anisotropy":float(e["jacobian_anisotropy"]),
            "footprint_area_fraction_tile":float(e["footprint_area_fraction_tile"]),
        })

    pair=pd.DataFrame(rows).sort_values(["query_id","rank"],kind="mergesort").reset_index(drop=True)
    assert_blind_safe(pair,"pair evidence")
    sel=select_winners(pair); assert_blind_safe(sel,"selection evidence")
    candidate_bytes=canonical_csv(pool); pair_bytes=canonical_csv(pair); selection_bytes=canonical_csv(sel)
    out=resolve(cfg["output_root"]).resolve()
    cp=out/cfg["outputs"]["candidate_pool"]; pp=out/cfg["outputs"]["pair_evidence"]; sp=out/cfg["outputs"]["selection_evidence"]; rp=out/cfg["outputs"]["report"]
    summary={
        "queries":int(sel["query_id"].nunique()),"candidate_pairs":len(pair),"map_tiles":len(tiles),
        "candidate_homography_ok":int(pair["homography_ok"].astype(bool).sum()),
        "candidate_finite_projection":int(pair["finite_projection"].astype(bool).sum()),
        "selected_homography_ok":int(sel["selected_homography_ok"].astype(bool).sum()),
        "selected_finite_projection":int(sel["selected_finite_projection"].astype(bool).sum()),
        "median_pair_matches":float(pair["matches"].median()),"median_pair_inliers":float(pair["inliers"].median()),
        "median_selected_matches":float(sel["selected_matches"].median()),"median_selected_inliers":float(sel["selected_inliers"].median()),
        "median_selected_dino_rank":float(sel["selected_dino_rank"].median()),"median_inlier_margin":float(sel["inlier_margin"].median()),
        "median_pair_total_ms_not_frozen":float(np.median(runtimes)),
    }
    report={
        "stage":"BV1.0","status":"PASS_BV1_0_BLIND_768_S256_SIFT_LIGHTGLUE_TRANSFER_FREEZE","summary":summary,
        "change_from_historical_demo":{"historical_map_variant":"512_s256","bv1_0_map_variant":"768_s256",
          "historical_verifier":"ORB hybrid","bv1_0_verifier":"SIFT+LightGlue","dino_retrieval_rerun_against_new_map_geometry":True,
          "historical_demo_outputs_modified":False},
        "scientific_boundary":{"reference_loaded":False,"srt_loaded":False,"gps_loaded":False,"oracle_loaded":False,
          "evaluation_metrics_computed":False,"bootstrap_executed":False,"fusion_executed":False,"truth_attached":False},
        "provenance":{"config_sha256":sha256_file(cfg_path),"source_demo_config_sha256":sha256_file(source_cfg_path),
          "source_blind_manifest_sha256":sha256_file(manifest_path),"source_query_cache_sha256":sha256_file(qcache_path),
          "source_blind_freeze_sha256":sha256_file(oldfreeze_path),"historical_submission_sha256":old["submission"]["sha256"],
          "tile_index_768_s256_sha256":sha256_file(tile_index_path),"map_cache_768_s256_sha256":sha256_file(map_cache_path),
          "candidate_pool_sha256":hashlib.sha256(candidate_bytes).hexdigest(),"pair_evidence_sha256":hashlib.sha256(pair_bytes).hexdigest(),
          "selection_evidence_sha256":hashlib.sha256(selection_bytes).hexdigest(),"lightglue":desc},
        "runtime":{"total_stage_wall_seconds_not_frozen":float(time.perf_counter()-started)}
    }
    freeze({cp:candidate_bytes,pp:pair_bytes,sp:selection_bytes,rp:canonical_json(report)})
    print("Status:",report["status"]); print("Queries/pairs:",f"{summary['queries']}/{summary['candidate_pairs']}")
    print("Map variant: 768_s256"); print("Map tiles:",summary["map_tiles"])
    print("Candidate homography/finite:",f"{summary['candidate_homography_ok']}/{summary['candidate_finite_projection']}","/ 2460")
    print("Selected homography/finite:",f"{summary['selected_homography_ok']}/{summary['selected_finite_projection']}","/ 123")
    print("Median pair matches/inliers:",f"{summary['median_pair_matches']:.1f}/{summary['median_pair_inliers']:.1f}")
    print("Median selected matches/inliers:",f"{summary['median_selected_matches']:.1f}/{summary['median_selected_inliers']:.1f}")
    print("Median selected DINO rank:",f"{summary['median_selected_dino_rank']:.1f}")
    print("Median inlier margin:",f"{summary['median_inlier_margin']:.1f}")
    print("Candidate pool SHA256:",report["provenance"]["candidate_pool_sha256"])
    print("Pair evidence SHA256:",report["provenance"]["pair_evidence_sha256"])
    print("Selection evidence SHA256:",report["provenance"]["selection_evidence_sha256"])
    print("Reference/SRT/GPS used: False"); print("Bootstrap/fusion executed: False"); print("Report:",rp)

if __name__=="__main__": main()
