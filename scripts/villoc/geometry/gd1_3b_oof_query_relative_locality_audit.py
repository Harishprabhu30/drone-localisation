#!/usr/bin/env python3
"""GD1.3-B: post-freeze OOF audit of query-relative and locality features."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT=Path(__file__).resolve().parents[3]
SRC_ROOT=REPO_ROOT/"src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0,str(SRC_ROOT))

import pandas as pd
import yaml

from uavloc.evaluation.oof_ranking import (
    binary_auc,
    query_rank_hits,
    run_contiguous_oof,
)

ROOT=REPO_ROOT
DEFAULT_CONFIG=ROOT/"configs/research/top20_geometric_discrimination_gd1_3b.yaml"


def resolve(v: str | Path)->Path:
    p=Path(v).expanduser()
    return p if p.is_absolute() else ROOT/p


def sha256_file(path: Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def load_json_compat(path: Path)->dict[str,Any]:
    """Read JSON, tolerating only GD1.3-A's historical trailing literal \\n defect."""
    raw=path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        stripped=raw.rstrip()
        if stripped.endswith("\\n"):
            candidate=stripped[:-2]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        raise RuntimeError(
            f"Invalid JSON in {path}; compatibility recovery did not apply"
        ) from exc


def deterministic_csv(frame: pd.DataFrame)->bytes:
    return frame.to_csv(index=False,lineterminator="\n",float_format="%.9f").encode("utf-8")


def deterministic_json(value: dict[str,Any])->bytes:
    return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode("utf-8")


def write_immutable(files: dict[Path,bytes])->None:
    bad=[str(p) for p,d in files.items() if p.exists() and p.read_bytes()!=d]
    if bad:
        raise RuntimeError("Refusing to overwrite differing GD1.3-B outputs: "+", ".join(bad))
    for p,d in files.items():
        if p.exists():
            continue
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open("xb") as f:
            f.write(d)


def verify_config(cfg: dict[str,Any])->None:
    if cfg.get("stage")!="GD1.3-B":
        raise RuntimeError("Expected GD1.3-B config")
    if cfg["evaluation"].get("fold_scheme")!="contiguous_query_blocks":
        raise RuntimeError("Fold scheme changed")
    if cfg["evaluation"].get("no_hyperparameter_tuning") is not True:
        raise RuntimeError("Hyperparameter tuning must remain disabled")
    if float(cfg["evaluation"].get("logistic_l2"))!=1.0:
        raise RuntimeError("Fixed L2 contract changed")
    for key,value in cfg["scientific_boundary"].items():
        expected=True if key=="development_only" else False
        if value is not expected:
            raise RuntimeError(f"Scientific boundary mismatch: {key}")


def load_and_verify(cfg: dict[str,Any])->pd.DataFrame:
    block=cfg["gd1_3a"]
    report_path=resolve(block["report"])
    features_path=resolve(block["development_features"])
    report=load_json_compat(report_path)
    if report.get("status")!="PASS_GD1_3A_REFERENCE_FREE_QUERY_RELATIVE_LOCALITY_FREEZE":
        raise RuntimeError("GD1.3-A report not PASS")
    if report.get("scientific_boundary",{}).get("reference_loaded") is not False:
        raise RuntimeError("GD1.3-A reference boundary violated")
    if report["trajectories"]["development"]["output"]["sha256"]!=block["expected_development_sha256"]:
        raise RuntimeError("GD1.3-A development report hash mismatch")
    if report["trajectories"]["blind_stress"]["output"]["sha256"]!=block["expected_blind_sha256"]:
        raise RuntimeError("GD1.3-A blind report hash mismatch")
    if sha256_file(features_path)!=block["expected_development_sha256"]:
        raise RuntimeError("GD1.3-A development bytes changed")

    features=pd.read_csv(features_path).copy()
    if len(features)!=int(block["expected_rows"]) or features["query_id"].nunique()!=int(block["expected_queries"]):
        raise RuntimeError("GD1.3-A development count mismatch")

    gd11_report_path=resolve(cfg["gd1_1"]["report"])
    gd11=load_json_compat(gd11_report_path)
    labels_path=resolve(cfg["gd1_1"]["reference_labels_source"])
    if gd11.get("status")!="PASS_GD1_1_POSTFREEZE_GEOMETRIC_DISCRIMINATION_AUDIT":
        raise RuntimeError("GD1.1 report not PASS")
    if sha256_file(labels_path)!=gd11["outputs"]["labeled_evidence"]["sha256"]:
        raise RuntimeError("GD1.1 labeled evidence changed")
    labels=pd.read_csv(labels_path)[
        ["query_id","pool_rank","tile_id","contains_query","center_le40","center_le80"]
    ].copy()

    keys=["query_id","pool_rank","tile_id"]
    if features[keys].duplicated().any() or labels[keys].duplicated().any():
        raise RuntimeError("Duplicate candidate key")
    merged=features.merge(labels,on=keys,how="left",validate="one_to_one")
    if merged[list(cfg["labels"])].isna().any().any():
        raise RuntimeError("Reference-label attachment failed")
    return merged


def feature_families(frame: pd.DataFrame)->dict[str,list[str]]:
    q_relative=sorted(
        c for c in frame.columns
        if c.endswith("__q_z") or c.endswith("__q_pct") or c.endswith("__missing")
    )
    nearby=sorted(
        c for c in frame.columns
        if c.startswith("nearby_component_") and not c.endswith("_id")
    )
    overlaps=sorted(
        c for c in frame.columns
        if c.startswith("overlaps_component_") and not c.endswith("_id")
    )
    inlier_relative=[
        c for c in ("inliers","inliers__q_z","inliers__q_pct","inliers__missing")
        if c in frame.columns
    ]
    if not q_relative or not nearby or not overlaps or len(inlier_relative)<3:
        raise RuntimeError("Expected GD1.3-A feature families are incomplete")
    return {
        "inliers_only":["inliers"],
        "inliers_query_relative":inlier_relative,
        "all_query_relative_geometry":q_relative,
        "nearby_locality_only":nearby,
        "overlap_locality_only":overlaps,
        "query_relative_plus_nearby":q_relative+nearby,
        "query_relative_plus_overlap":q_relative+overlaps,
        "query_relative_plus_both_localities":q_relative+nearby+overlaps,
        "query_relative_plus_both_localities_plus_retrieval":q_relative+nearby+overlaps+["pool_rank"],
    }


def retrieval_scores(frame: pd.DataFrame)->pd.Series:
    return -pd.to_numeric(frame["pool_rank"],errors="raise").astype(float)


def main()->None:
    ap=argparse.ArgumentParser(
        description="GD1.3-B post-freeze OOF query-relative/locality selector audit"
    )
    ap.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    args=ap.parse_args()
    cfg_path=resolve(args.config)
    cfg=yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verify_config(cfg)
    frame=load_and_verify(cfg)
    families=feature_families(frame)

    labels=list(cfg["labels"])
    model_families=list(cfg["model_families"])
    topks=[int(v) for v in cfg["evaluation"]["topk"]]
    rows=[]
    results={}

    for label in labels:
        results[label]={}
        for family in model_families:
            if family=="retrieval_rank_only":
                scores=retrieval_scores(frame)
                folds=[]
            else:
                scores,folds=run_contiguous_oof(
                    frame,
                    features=families[family],
                    label_col=label,
                    folds=int(cfg["evaluation"]["folds"]),
                    clip_value=float(cfg["evaluation"]["standardized_clip"]),
                    l2=float(cfg["evaluation"]["logistic_l2"]),
                    max_iterations=int(cfg["evaluation"]["logistic_max_iterations"]),
                    tolerance=float(cfg["evaluation"]["logistic_tolerance"]),
                    balanced=bool(cfg["evaluation"]["balanced_class_weights"]),
                )
            tmp=frame[["query_id","pool_rank","tile_id",label]].copy()
            tmp["score"]=scores
            auc=binary_auc(tmp[label],tmp["score"])
            hits=query_rank_hits(
                tmp,score_col="score",label_col=label,topks=topks
            )
            results[label][family]={
                "pair_auc_oof":auc,
                "query_hits":hits,
                "folds":folds,
                "feature_count":0 if family=="retrieval_rank_only" else len(families[family]),
            }
            for row in tmp.itertuples(index=False):
                rows.append({
                    "label":label,
                    "model_family":family,
                    "query_id":int(row.query_id),
                    "pool_rank":int(row.pool_rank),
                    "tile_id":str(row.tile_id),
                    "target":bool(getattr(row,label)),
                    "oof_score":float(row.score),
                })

    report={
        "stage":"GD1.3-B",
        "status":"PASS_GD1_3B_OOF_QUERY_RELATIVE_LOCALITY_AUDIT",
        "scope":"development_only_postfreeze",
        "protocol":cfg["evaluation"],
        "results":results,
        "scientific_boundary":{
            "gd1_3a_generated_before_reference":True,
            "blind_stress_loaded":False,
            "blind_stress_reference_used":False,
            "hyperparameter_tuning_executed":False,
            "frontend_changed":False,
            "candidate_pool_changed":False,
            "deployable_model_exported":False,
            "bootstrap_state_modified":False,
        },
        "interpretation":{
            "decision_gate_against_inliers_baseline":True,
            "no_selector_promoted":True,
            "no_final_model_fit":True,
            "external_validation_still_required":True,
        },
        "provenance":{
            "config_sha256":sha256_file(cfg_path),
            "script_sha256":sha256_file(Path(__file__).resolve()),
            "gd1_3a_report_sha256":sha256_file(resolve(cfg["gd1_3a"]["report"])),
            "gd1_3a_development_sha256":sha256_file(resolve(cfg["gd1_3a"]["development_features"])),
            "gd1_1_report_sha256":sha256_file(resolve(cfg["gd1_1"]["report"])),
            "gd1_1_labeled_evidence_sha256":sha256_file(resolve(cfg["gd1_1"]["reference_labels_source"])),
        },
        "outputs":{},
    }

    scores=pd.DataFrame(rows).sort_values(
        ["label","model_family","query_id","pool_rank"],kind="mergesort"
    )
    out=resolve(cfg["output_root"])
    scores_path=out/"gd1_3b_oof_candidate_scores.csv"
    score_bytes=deterministic_csv(scores)
    report["outputs"]["oof_scores"]={
        "path":str(scores_path.relative_to(ROOT)),
        "sha256":hashlib.sha256(score_bytes).hexdigest(),
        "rows":int(len(scores)),
    }
    report_path=out/"gd1_3b_report.json"
    report_bytes=deterministic_json(report)
    write_immutable({scores_path:score_bytes,report_path:report_bytes})

    print("Status:",report["status"])
    for label in labels:
        print(f"{label}:")
        for family in model_families:
            r=results[label][family]
            h=r["query_hits"]
            print(
                f"  {family}: auc={r['pair_auc_oof'] if r['pair_auc_oof'] is not None else 'NA'} "
                f"top1={h['top1']}/403 top3={h['top3']}/403 "
                f"top5={h['top5']}/403 top10={h['top10']}/403 "
                f"features={r['feature_count']}"
            )
    print("Blind stress loaded: False")
    print("Hyperparameter tuning executed: False")
    print("Deployable model exported: False")
    print("Report:",report_path)
    print("OOF score SHA256:",report["outputs"]["oof_scores"]["sha256"])


if __name__=="__main__":
    main()
