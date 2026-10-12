#!/usr/bin/env python3
"""GD1.3-A: freeze query-relative evidence and Top20 physical-locality features."""
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

import numpy as np
import pandas as pd
import yaml

ROOT=REPO_ROOT
DEFAULT_CONFIG=ROOT/"configs/research/top20_geometric_discrimination_gd1_3a.yaml"


def resolve(v: str | Path)->Path:
    p=Path(v).expanduser()
    return p if p.is_absolute() else ROOT/p


def sha256_file(path: Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def numeric_feature(s: pd.Series)->pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    text=s.astype(str).str.strip().str.lower()
    boolish=text.map({"true":1.0,"false":0.0,"yes":1.0,"no":0.0,"y":1.0,"n":0.0})
    numeric=pd.to_numeric(s,errors="coerce")
    return numeric.where(numeric.notna(),boolish).astype(float)


def canonical_csv(frame: pd.DataFrame)->bytes:
    return frame.to_csv(index=False,lineterminator="\n",float_format="%.9f").encode("utf-8")


def canonical_json(value: dict[str,Any])->bytes:
    return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode("utf-8")


def canonical_json(value: dict[str,Any])->bytes:
    return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\\n").encode("utf-8")


def freeze(files: dict[Path,bytes])->None:
    bad=[str(p) for p,d in files.items() if p.exists() and p.read_bytes()!=d]
    if bad:
        raise RuntimeError("Refusing to overwrite differing GD1.3-A freeze: "+", ".join(bad))
    for p,d in files.items():
        if p.exists():
            continue
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open("xb") as f:
            f.write(d)


def verify_config(cfg: dict[str,Any])->None:
    if cfg.get("stage")!="GD1.3-A":
        raise RuntimeError("Expected GD1.3-A config")
    for key,value in cfg["scientific_boundary"].items():
        if value is not False:
            raise RuntimeError(f"{key} must be false")
    if cfg["query_relative"]["transforms"] != ["robust_z","percentile","missing"]:
        raise RuntimeError("Query-relative transform contract changed")
    if cfg["locality"]["graphs"] != ["nearby","overlaps"]:
        raise RuntimeError("Locality graph contract changed")


def verify_inputs(cfg: dict[str,Any])->tuple[dict[str,Any],pd.DataFrame]:
    report_path=resolve(cfg["gd1_0"]["report"])
    report=json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status")!="PASS_GD1_0_REFERENCE_FREE_TOP20_GEOMETRIC_EVIDENCE_FREEZE":
        raise RuntimeError("GD1.0 report not PASS")
    if report.get("scientific_boundary",{}).get("reference_loaded") is not False:
        raise RuntimeError("GD1.0 reference boundary violated")

    rg20_path=resolve(cfg["rg2_0"]["report"])
    rg20=json.loads(rg20_path.read_text(encoding="utf-8"))
    if rg20.get("status")!="PASS_RG2_0_MAP_TOPOLOGY_PREFLIGHT":
        raise RuntimeError("RG2.0 report not PASS")
    rel_path=resolve(cfg["rg2_0"]["relationships"])
    if sha256_file(rel_path)!=cfg["rg2_0"]["expected_relationships_sha256"]:
        raise RuntimeError("RG2.0 relationship bytes changed")
    if rg20["provenance"]["relationships_sha256"]!=cfg["rg2_0"]["expected_relationships_sha256"]:
        raise RuntimeError("RG2.0 relationship report hash mismatch")
    relationships=pd.read_csv(rel_path)
    return report,relationships


def connected_components(
    tile_ids: list[str],
    relationships: pd.DataFrame,
    predicate: str,
)->dict[str,int]:
    nodes=sorted(set(str(v) for v in tile_ids))
    allowed=set(nodes)
    adjacency={n:set() for n in nodes}
    rel=relationships[relationships[predicate].astype(str).str.lower().isin(["true","1","yes"])]
    for row in rel.itertuples(index=False):
        a=str(row.tile_id); b=str(row.neighbor_tile_id)
        if a in allowed and b in allowed:
            adjacency[a].add(b); adjacency[b].add(a)
    component={}
    cid=0
    for start in nodes:
        if start in component:
            continue
        cid+=1
        stack=[start]
        component[start]=cid
        while stack:
            cur=stack.pop()
            for nxt in sorted(adjacency[cur]):
                if nxt not in component:
                    component[nxt]=cid
                    stack.append(nxt)
    return component


def add_query_relative(frame: pd.DataFrame, features: list[str])->pd.DataFrame:
    out=frame.copy()
    for feature in features:
        values=numeric_feature(out[feature]).replace([np.inf,-np.inf],np.nan)
        out[f"{feature}__q_z"]=np.nan
        out[f"{feature}__q_pct"]=np.nan
        out[f"{feature}__missing"]=values.isna().astype(int)
        for _,idx in out.groupby("query_id",sort=True).groups.items():
            idx=list(idx)
            v=values.loc[idx]
            finite=v.dropna()
            if finite.empty:
                continue
            med=float(finite.median())
            scale=float(finite.quantile(0.75)-finite.quantile(0.25))
            if not math.isfinite(scale) or scale<=1e-12:
                scale=float(finite.std(ddof=0))
            if not math.isfinite(scale) or scale<=1e-12:
                scale=1.0
            out.loc[finite.index,f"{feature}__q_z"]=(finite-med)/scale
            if len(finite)==1:
                out.loc[finite.index,f"{feature}__q_pct"]=0.5
            else:
                ranks=finite.rank(method="average",ascending=True)
                out.loc[finite.index,f"{feature}__q_pct"]=(ranks-1.0)/(len(finite)-1.0)
    return out


def add_locality_features(
    frame: pd.DataFrame,
    relationships: pd.DataFrame,
    predicate: str,
)->pd.DataFrame:
    out=frame.copy()
    prefix=f"{predicate}_component"
    columns=[
        "id","size","min_pool_rank","mean_pool_rank","reciprocal_rank_sum",
        "unique_source_views","inliers_max","inliers_median","good_matches_max",
        "orientation_resultant_median","jacobian_anisotropy_median",
    ]
    for c in columns:
        out[f"{prefix}_{c}"]=np.nan

    for _,idx in out.groupby("query_id",sort=True).groups.items():
        idx=list(idx)
        group=out.loc[idx].copy()
        comp=connected_components(group["tile_id"].astype(str).tolist(),relationships,predicate)
        group["_cid"]=group["tile_id"].astype(str).map(comp).astype(int)
        for cid,cg in group.groupby("_cid",sort=True):
            ranks=pd.to_numeric(cg["pool_rank"],errors="raise").astype(float)
            stats={
                "id":int(cid),
                "size":int(len(cg)),
                "min_pool_rank":float(ranks.min()),
                "mean_pool_rank":float(ranks.mean()),
                "reciprocal_rank_sum":float((1.0/ranks).sum()),
                "unique_source_views":int(cg["source_view"].astype(str).nunique()),
                "inliers_max":float(numeric_feature(cg["inliers"]).max()),
                "inliers_median":float(numeric_feature(cg["inliers"]).median()),
                "good_matches_max":float(numeric_feature(cg["good_matches"]).max()),
                "orientation_resultant_median":float(numeric_feature(cg["orientation_resultant"]).median()),
                "jacobian_anisotropy_median":float(numeric_feature(cg["jacobian_anisotropy"]).replace([np.inf,-np.inf],np.nan).median()),
            }
            target=cg.index
            for name,value in stats.items():
                out.loc[target,f"{prefix}_{name}"]=value
    return out


def main()->None:
    ap=argparse.ArgumentParser(description="GD1.3-A reference-free query-relative and locality feature freeze")
    ap.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    args=ap.parse_args()
    cfg_path=resolve(args.config)
    cfg=yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verify_config(cfg)
    gd10,relationships=verify_inputs(cfg)

    feature_groups=gd10["evidence_groups"]
    raw_features=[]
    for group in sorted(feature_groups):
        for feature in feature_groups[group]:
            if feature not in raw_features:
                raw_features.append(feature)

    files={}
    reports={}
    for key in ("development","blind_stress"):
        block=cfg["gd1_0"][key]
        path=resolve(block["evidence"])
        if sha256_file(path)!=block["expected_sha256"]:
            raise RuntimeError(f"{key} GD1.0 evidence hash changed")
        if gd10["trajectories"][key]["output"]["sha256"]!=block["expected_sha256"]:
            raise RuntimeError(f"{key} GD1.0 report hash mismatch")
        frame=pd.read_csv(path).copy()
        if len(frame)!=int(block["expected_pairs"]) or frame["query_id"].nunique()!=int(block["expected_queries"]):
            raise RuntimeError(f"{key} GD1.0 count mismatch")
        missing=[f for f in raw_features if f not in frame.columns]
        if missing:
            raise RuntimeError(f"{key} missing raw evidence columns: {missing}")

        enriched=add_query_relative(frame,raw_features)
        for predicate in cfg["locality"]["graphs"]:
            enriched=add_locality_features(enriched,relationships,predicate)
        enriched=enriched.sort_values(["query_id","pool_rank"],kind="mergesort").reset_index(drop=True)

        out=resolve(cfg["output_root"])/key/"gd1_3a_query_relative_locality_features.csv"
        payload=canonical_csv(enriched)
        files[out]=payload

        component_summary={}
        for predicate in cfg["locality"]["graphs"]:
            ccol=f"{predicate}_component_id"
            counts=enriched.groupby("query_id")[ccol].nunique()
            component_summary[predicate]={
                "median_components_per_query":float(counts.median()),
                "min_components_per_query":int(counts.min()),
                "max_components_per_query":int(counts.max()),
            }
        reports[key]={
            "queries":int(enriched["query_id"].nunique()),
            "rows":int(len(enriched)),
            "raw_features_transformed":int(len(raw_features)),
            "component_summary":component_summary,
            "output":{"sha256":hashlib.sha256(payload).hexdigest(),"rows":int(len(enriched))},
        }

    report={
        "stage":"GD1.3-A",
        "status":"PASS_GD1_3A_REFERENCE_FREE_QUERY_RELATIVE_LOCALITY_FREEZE",
        "query_relative_contract":cfg["query_relative"],
        "locality_contract":cfg["locality"],
        "trajectories":reports,
        "scientific_boundary":{
            "reference_loaded":False,
            "oracle_labels_loaded":False,
            "reranking_executed":False,
            "selector_executed":False,
            "threshold_tuned":False,
            "orb_rerun":False,
            "learned_frontend_executed":False,
            "temporal_fusion_executed":False,
            "bootstrap_state_modified":False,
        },
        "interpretation":{
            "features_are_reference_free_transformations_of_frozen_gd1_0":True,
            "locality_ids_are_top20_induced_map_components_not_truth_labels":True,
            "no_accuracy_claim":True,
        },
        "provenance":{
            "config_sha256":sha256_file(cfg_path),
            "script_sha256":sha256_file(Path(__file__).resolve()),
            "gd1_0_report_sha256":sha256_file(resolve(cfg["gd1_0"]["report"])),
            "rg2_0_report_sha256":sha256_file(resolve(cfg["rg2_0"]["report"])),
            "relationships_sha256":sha256_file(resolve(cfg["rg2_0"]["relationships"])),
        },
    }
    report_path=resolve(cfg["output_root"])/"gd1_3a_freeze_report.json"
    files[report_path]=canonical_json(report)
    freeze(files)

    print("Status:",report["status"])
    for key,r in reports.items():
        print(f"{key}: queries={r['queries']} rows={r['rows']} raw_features={r['raw_features_transformed']}")
        print("  components:",r["component_summary"])
        print("  evidence SHA256:",r["output"]["sha256"])
    print("Reference loaded: False")
    print("Reranking/selector executed: False")
    print("Freeze report:",report_path)


if __name__=="__main__":
    main()
