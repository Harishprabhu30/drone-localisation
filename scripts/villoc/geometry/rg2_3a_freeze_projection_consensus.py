#!/usr/bin/env python3
"""RG2.3-A reference-free consensus generation from frozen RG2.2 projections."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
import yaml

from uavloc.data.trajectory_adapter import FORBIDDEN_BLIND_REFERENCE_COLUMNS
from uavloc.geometry.projection_consensus import (
    coordinate_median,
    dispersion,
    dominant_radius_cluster_medoid,
    geometric_median,
    inlier_huber_centroid,
    medoid,
)

ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/local_map_neighborhood_rg2_3a.yaml"

REQ = {
    "trajectory_id","role","query_id","anchor_tile_id","tile_id","union_rank",
    "is_anchor","in_immediate","homography_ok","finite_projection",
    "projected_inside_tile","projected_easting","projected_northing","inliers",
}
FORBIDDEN_PARTIAL=("ground_truth","oracle","eval_ref_","reference_x","reference_y",
                   "projected_error","tile_center_error","chosen_error","gt_error")


def resolve(v):
    p=Path(v).expanduser()
    return p if p.is_absolute() else ROOT/p


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def assert_blind_safe(frame: pd.DataFrame, label: str):
    cols={str(c).strip().lower() for c in frame.columns}
    exact={str(c).strip().lower() for c in FORBIDDEN_BLIND_REFERENCE_COLUMNS
           if str(c).strip().lower()!="reference_available"}
    leaked=sorted((cols & exact) | {c for c in cols if any(t in c for t in FORBIDDEN_PARTIAL)})
    if leaked:
        raise RuntimeError(f"{label} contains forbidden reference/evaluation columns: {leaked}")


def bseries(s: pd.Series)->pd.Series:
    m={"true":True,"false":False,"1":True,"0":False,"yes":True,"no":False}
    if pd.api.types.is_bool_dtype(s): return s.astype(bool)
    out=s.astype(str).str.strip().str.lower().map(m)
    if out.isna().any(): raise RuntimeError("Non-boolean frozen field")
    return out.astype(bool)


def canonical_csv(rows):
    out=io.StringIO(newline="")
    w=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator="\n"); w.writeheader()
    for row in rows:
        cooked={}
        for k,v in row.items():
            if isinstance(v,bool): cooked[k]="true" if v else "false"
            elif isinstance(v,float): cooked[k]="" if not math.isfinite(v) else f"{v:.9f}"
            else: cooked[k]="" if v is None else v
        w.writerow(cooked)
    return out.getvalue().encode()


def canonical_json(v): return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()


def freeze(files):
    bad=[str(p) for p,d in files.items() if p.exists() and p.read_bytes()!=d]
    if bad: raise RuntimeError("Refusing to overwrite differing RG2.3-A freeze: "+", ".join(bad))
    for p,d in files.items():
        if not p.exists():
            p.parent.mkdir(parents=True,exist_ok=True)
            with p.open("xb") as f:f.write(d)


def verify_config(cfg):
    if cfg.get("stage")!="RG2.3-A": raise RuntimeError("Wrong stage")
    if cfg.get("map_variant")!="768_s256" or cfg.get("map_crs")!="EPSG:3346":
        raise RuntimeError("Map contract changed")
    boundary=cfg["scientific_boundary"]
    for k,v in boundary.items():
        if k=="reference_allowed":
            if v is not False: raise RuntimeError("Reference must be forbidden")
        elif v is not False:
            raise RuntimeError(f"{k} must be false")
    cc=cfg["consensus_contract"]
    if cc["cluster_radius_m"]!=25.6 or cc["huber_k"]!=1.345 or cc["no_reference_tuning"] is not True:
        raise RuntimeError("Consensus contract changed")


def verify_rg22(cfg):
    rp=resolve(cfg["rg2_2"]["report"])
    report=json.loads(rp.read_text())
    if report.get("status")!="PASS_RG2_2_REFERENCE_FREE_MULTI_TILE_PROJECTION_FREEZE":
        raise RuntimeError("RG2.2 report not PASS")
    if report.get("scientific_boundary",{}).get("reference_loaded") is not False:
        raise RuntimeError("RG2.2 reference boundary violated")
    return report,rp


def consensus_rows(frame: pd.DataFrame, cfg, trajectory_key: str):
    cc=cfg["consensus_contract"]
    rows=[]
    for qid,g in frame.groupby("query_id",sort=True):
        anchor=g[g["is_anchor"]]
        if len(anchor)!=1: raise RuntimeError(f"q{qid} anchor count !=1")
        ar=anchor.iloc[0]
        if not bool(ar["finite_projection"]): raise RuntimeError(f"q{qid} anchor projection not finite")
        anchor_e=float(ar["projected_easting"]); anchor_n=float(ar["projected_northing"])
        for subset in cc["subsets"]:
            base=g if subset=="full_overlap" else g[g["in_immediate"]]
            for view in cc["projection_views"]:
                use=base[base["finite_projection"]].copy()
                if view=="inside_only":
                    use=use[use["projected_inside_tile"]]
                if len(use)==0:
                    continue
                pts=use[["projected_easting","projected_northing"]].to_numpy(float)
                inliers=np.maximum(use["inliers"].to_numpy(float),1.0)
                disp=dispersion(pts)
                methods={}
                methods["coordinate_median"]=coordinate_median(pts)
                methods["geometric_median"]=geometric_median(pts)
                mi,mx,my=medoid(pts); methods["medoid"]=(mx,my)
                ci,support,cx,cy=dominant_radius_cluster_medoid(
                    pts,radius_m=float(cc["cluster_radius_m"])
                ); methods["radius_cluster_medoid"]=(cx,cy)
                methods["inlier_huber_centroid"]=inlier_huber_centroid(
                    pts,inlier_weights=inliers,huber_k=float(cc["huber_k"])
                )
                for method,(e,n) in methods.items():
                    rows.append({
                        "trajectory_key":trajectory_key,
                        "trajectory_id":str(ar["trajectory_id"]),
                        "query_id":int(qid),
                        "subset":subset,
                        "projection_view":view,
                        "method":method,
                        "point_count":int(len(use)),
                        "consensus_easting":float(e),
                        "consensus_northing":float(n),
                        "anchor_projection_easting":anchor_e,
                        "anchor_projection_northing":anchor_n,
                        "anchor_projected_inside_tile":bool(ar["projected_inside_tile"]),
                        "cluster_support_count":int(len(support)) if method=="radius_cluster_medoid" else "",
                        "radial_median_m":disp["radial_median_m"],
                        "radial_p90_m":disp["radial_p90_m"],
                        "radial_max_m":disp["radial_max_m"],
                        "pairwise_median_m":disp["pairwise_median_m"],
                        "pairwise_p90_m":disp["pairwise_p90_m"],
                        "pairwise_max_m":disp["pairwise_max_m"],
                        "bbox_diagonal_m":disp["bbox_diagonal_m"],
                    })
    return rows


def main():
    ap=argparse.ArgumentParser(description="RG2.3-A reference-free projection consensus freeze")
    ap.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    args=ap.parse_args()
    cfg=yaml.safe_load(resolve(args.config).read_text())
    verify_config(cfg)
    rg22,report_path=verify_rg22(cfg)

    files={}
    reports={}
    for key in ("development","blind_stress"):
        tc=cfg["rg2_2"][key]
        pair=resolve(tc["pair_evidence"]); summary=resolve(tc["query_summary"])
        if sha256_file(pair)!=tc["expected_pair_sha256"]: raise RuntimeError(f"{key} pair hash changed")
        if sha256_file(summary)!=tc["expected_summary_sha256"]: raise RuntimeError(f"{key} summary hash changed")
        frame=pd.read_csv(pair).copy()
        missing=sorted(REQ-set(frame.columns))
        if missing: raise RuntimeError(f"{key} missing columns {missing}")
        assert_blind_safe(frame,key)
        for c in ("is_anchor","in_immediate","homography_ok","finite_projection","projected_inside_tile"):
            frame[c]=bseries(frame[c])
        frame["query_id"]=pd.to_numeric(frame["query_id"],errors="raise").astype(int)
        frame["inliers"]=pd.to_numeric(frame["inliers"],errors="raise").astype(int)
        for c in ("projected_easting","projected_northing"):
            frame[c]=pd.to_numeric(frame[c],errors="coerce")
        if len(frame)!=int(tc["expected_pairs"]) or frame["query_id"].nunique()!=int(tc["expected_queries"]):
            raise RuntimeError(f"{key} frozen count mismatch")
        rows=consensus_rows(frame,cfg,key)
        out=resolve(cfg["output_root"])/key/"rg2_3a_consensus_coordinates.csv"
        payload=canonical_csv(rows); files[out]=payload
        rr=pd.DataFrame(rows)
        reports[key]={
            "queries":int(frame["query_id"].nunique()),
            "input_pairs":int(len(frame)),
            "consensus_rows":int(len(rows)),
            "method_counts":{str(k):int(v) for k,v in rr["method"].value_counts().sort_index().items()},
            "subset_view_counts":{
                f"{a}|{b}":int(v)
                for (a,b),v in rr.groupby(["subset","projection_view"]).size().items()
            },
            "point_count_histograms":{
                f"{a}|{b}":{str(int(k)):int(v) for k,v in g["point_count"].value_counts().sort_index().items()}
                for (a,b),g in rr.groupby(["subset","projection_view"])
            },
            "output":{"sha256":hashlib.sha256(payload).hexdigest(),"rows":len(rows)}
        }

    report={
        "stage":"RG2.3-A",
        "status":"PASS_RG2_3A_REFERENCE_FREE_CONSENSUS_FREEZE",
        "consensus_contract":cfg["consensus_contract"],
        "trajectories":reports,
        "scientific_boundary":{
            "reference_loaded":False,"srt_gps_loaded":False,"oracle_labels_loaded":False,
            "orb_rerun":False,"projection_rerun":False,"method_selected_by_accuracy":False,
            "accuracy_evaluation_executed":False
        },
        "provenance":{
            "config_sha256":sha256_file(resolve(args.config)),
            "script_sha256":sha256_file(Path(__file__).resolve()),
            "consensus_module_sha256":sha256_file(Path(sys.modules[coordinate_median.__module__].__file__).resolve()),
            "rg2_2_report_sha256":sha256_file(report_path)
        }
    }
    rp=resolve(cfg["output_root"])/"rg2_3a_freeze_report.json"
    files[rp]=canonical_json(report); freeze(files)

    print("Status:",report["status"])
    for key,r in reports.items():
        print(f"{key}: queries={r['queries']} input_pairs={r['input_pairs']} consensus_rows={r['consensus_rows']}")
        print("  subset/view counts:",r["subset_view_counts"])
        print("  consensus SHA256:",r["output"]["sha256"])
    print("Reference loaded: False")
    print("Accuracy evaluation executed: False")
    print("Freeze report:",rp)

if __name__=="__main__":
    main()
