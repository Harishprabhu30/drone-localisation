#!/usr/bin/env python3
"""LF1.2-C: post-freeze ORB vs sparse XFeat vs XFeat* comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT=Path(__file__).resolve().parents[3]
SRC_ROOT=REPO_ROOT/"src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0,str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0,str(SRC_ROOT))

import pandas as pd
import yaml

from uavloc.evaluation.reference import load_reference_xy
from scripts.villoc.geometry.lf1_1b_postfreeze_orb_xfeat_comparison import (
    attach_reference,
    diagnostic_feature_ranking,
    homography_retention,
    oracle_projection_summary,
    projection_summary,
    ranking_summary,
    selected_projection_rows,
    verify_tile_geometry,
)

ROOT=REPO_ROOT
DEFAULT_CONFIG=ROOT/"configs/research/learned_geometric_frontend_lf1_2c.yaml"


def resolve(value: str|Path)->Path:
    p=Path(value).expanduser()
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


def write_immutable(files: dict[Path,bytes])->None:
    changed=[str(p) for p,d in files.items() if p.exists() and p.read_bytes()!=d]
    if changed:
        raise RuntimeError("Refusing to overwrite differing LF1.2-C outputs: "+", ".join(changed))
    for p,d in files.items():
        if p.exists():
            continue
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open("xb") as f:
            f.write(d)


def verify_config(cfg: dict[str,Any])->None:
    if cfg.get("stage")!="LF1.2-C":
        raise RuntimeError("Expected LF1.2-C config")
    if cfg.get("labels")!=["contains_query","center_le40","center_le80"]:
        raise RuntimeError("Label contract changed")
    if cfg["ranking"].get("selectors")!=[
        "retrieval_rank","inliers","valid_then_inliers"
    ]:
        raise RuntimeError("Ranking selector contract changed")
    boundary=cfg["scientific_boundary"]
    if boundary.get("development_reference_allowed") is not True:
        raise RuntimeError("Development reference must be explicitly allowed")
    if boundary.get("reference_after_all_freezes_only") is not True:
        raise RuntimeError("Reference-after-all-freezes contract missing")
    for key in (
        "blind_stress_loaded",
        "blind_stress_reference_allowed",
        "evidence_mutable",
        "threshold_tuning_enabled",
        "selector_training_enabled",
        "frontend_parameters_mutable",
        "bootstrap_enabled",
        "state_enabled",
    ):
        if boundary.get(key) is not False:
            raise RuntimeError(f"{key} must be false")


def verify_frontend(
    cfg: dict[str,Any],
    key: str,
    expected_status: str,
    *,
    verify_blind_hash: bool=False,
)->tuple[Path,dict[str,Any]]:
    block=cfg[key]
    report_path=resolve(block["report"])
    evidence_path=resolve(block["development_evidence"])
    report=json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status")!=expected_status:
        raise RuntimeError(f"{key} report not PASS")
    boundary=report.get("scientific_boundary",{})
    if boundary.get("reference_loaded") is not False:
        raise RuntimeError(f"{key} reference boundary violated")
    if report["trajectories"]["development"]["output"]["sha256"]!=block["expected_development_sha256"]:
        raise RuntimeError(f"{key} development report hash mismatch")
    if verify_blind_hash and report["trajectories"]["blind_stress"]["output"]["sha256"]!=block["expected_blind_sha256"]:
        raise RuntimeError(f"{key} blind report hash mismatch")
    if sha256_file(evidence_path)!=block["expected_development_sha256"]:
        raise RuntimeError(f"{key} development evidence bytes changed")
    return evidence_path,report


def main()->None:
    ap=argparse.ArgumentParser(
        description="LF1.2-C post-freeze ORB vs sparse XFeat vs XFeat* comparison"
    )
    ap.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    args=ap.parse_args()

    cfg_path=resolve(args.config)
    cfg=yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verify_config(cfg)

    orb_path,orb_report=verify_frontend(
        cfg,"orb","PASS_GD1_0_REFERENCE_FREE_TOP20_GEOMETRIC_EVIDENCE_FREEZE"
    )
    sparse_path,sparse_report=verify_frontend(
        cfg,"xfeat_sparse","PASS_LF1_1A_REFERENCE_FREE_XFEAT_TOP20_FREEZE"
    )
    star_path,star_report=verify_frontend(
        cfg,"xfeat_star","PASS_LF1_2B_REFERENCE_FREE_XFEAT_STAR_TOP20_FREEZE",
        verify_blind_hash=True,
    )
    tile_path=verify_tile_geometry({
        "map":{
            "rg2_0_report":cfg["map"]["rg2_0_report"],
            "tile_geometry":cfg["map"]["tile_geometry"],
        }
    })

    raw={
        "orb":pd.read_csv(orb_path),
        "xfeat_sparse":pd.read_csv(sparse_path),
        "xfeat_star":pd.read_csv(star_path),
    }
    for key,frame in raw.items():
        block=cfg[key]
        if len(frame)!=int(block["expected_pairs"]):
            raise RuntimeError(f"{key} pair count mismatch")
        if frame["query_id"].nunique()!=int(block["expected_queries"]):
            raise RuntimeError(f"{key} query count mismatch")

    keys=["query_id","pool_rank","tile_id"]
    base=raw["orb"][keys].reset_index(drop=True)
    for key in ("xfeat_sparse","xfeat_star"):
        if not base.equals(raw[key][keys].reset_index(drop=True)):
            raise RuntimeError(f"{key} frozen Top20 ordering differs from ORB")

    for spec in cfg["common_features"]:
        for frontend in ("orb","xfeat_sparse","xfeat_star"):
            column=spec[frontend]
            if column not in raw[frontend].columns:
                raise RuntimeError(
                    f"{frontend} missing common feature {column}"
                )

    # Critical boundary: all three freezes verified before reference is opened.
    reference_path=resolve(cfg["development"]["reference_attachment"])
    print("ORB, sparse XFeat, and XFeat* frozen evidence verified before reference attachment.")
    print("Attaching development reference for LF1.2-C diagnostics only.")
    reference=load_reference_xy(reference_path)
    tile=pd.read_csv(tile_path)

    frames={
        "orb":attach_reference(raw["orb"],tile,reference,frontend="orb"),
        "xfeat_sparse":attach_reference(
            raw["xfeat_sparse"],tile,reference,frontend="xfeat_sparse"
        ),
        "xfeat_star":attach_reference(
            raw["xfeat_star"],tile,reference,frontend="xfeat_star"
        ),
    }

    labels=list(cfg["labels"])
    topks=[int(v) for v in cfg["ranking"]["topk"]]
    selectors=list(cfg["ranking"]["selectors"])
    selected_selector=str(cfg["projection"]["selected_selector"])

    report={
        "stage":"LF1.2-C",
        "status":"PASS_LF1_2C_POSTFREEZE_THREE_FRONTEND_COMPARISON",
        "scope":"development_only_postfreeze",
        "frontends":{},
        "common_feature_diagnostics":{},
        "scientific_boundary":{
            "all_three_freezes_verified_before_reference":True,
            "blind_stress_loaded":False,
            "blind_stress_reference_used":False,
            "threshold_tuning_executed":False,
            "selector_training_executed":False,
            "frontend_parameters_changed":False,
            "frontend_winner_promoted":False,
            "bootstrap_state_modified":False,
        },
        "interpretation":{
            "feature_auc_direction_is_development_diagnostic_only":True,
            "oracle_projection_is_diagnostic_upper_bound_only":True,
            "no_deployment_rule_selected":True,
        },
        "provenance":{
            "config_sha256":sha256_file(cfg_path),
            "script_sha256":sha256_file(Path(__file__).resolve()),
            "orb_report_sha256":sha256_file(resolve(cfg["orb"]["report"])),
            "orb_evidence_sha256":sha256_file(orb_path),
            "xfeat_sparse_report_sha256":sha256_file(resolve(cfg["xfeat_sparse"]["report"])),
            "xfeat_sparse_evidence_sha256":sha256_file(sparse_path),
            "xfeat_star_report_sha256":sha256_file(resolve(cfg["xfeat_star"]["report"])),
            "xfeat_star_evidence_sha256":sha256_file(star_path),
            "tile_geometry_sha256":sha256_file(tile_path),
            "reference_attachment_sha256":sha256_file(reference_path),
        },
        "outputs":{},
    }

    labeled_frames=[]
    selected_frames=[]
    for frontend,frame in frames.items():
        fb={
            "pair_counts":{
                "pairs":int(len(frame)),
                "queries":int(frame["query_id"].nunique()),
                "homography_ok":int(frame["homography_ok"].astype(bool).sum()),
                "finite_projection":int(frame["finite_projection"].astype(bool).sum()),
            },
            "labels":{},
            "projection":{},
        }
        for label in labels:
            fb["labels"][label]={
                "homography_retention":homography_retention(frame,label),
                "rankings":{
                    selector:ranking_summary(frame,label,selector,topks)
                    for selector in selectors
                },
            }
        selected=selected_projection_rows(frame,selected_selector)
        selected_frames.append(selected)
        fb["projection"]={
            "selected_valid_then_inliers":projection_summary(
                selected["projection_error_m"]
            ),
            "oracle_best_finite_projection":oracle_projection_summary(frame),
        }
        report["frontends"][frontend]=fb
        labeled_frames.append(frame)

    feature_rows=[]
    for spec in cfg["common_features"]:
        name=str(spec["name"])
        report["common_feature_diagnostics"][name]={}
        for frontend,frame in frames.items():
            column=str(spec[frontend])
            block={}
            for label in labels:
                diag=diagnostic_feature_ranking(
                    frame,column,label,topks
                )
                block[label]=diag
                auc=diag["pair_auc"]
                hits=diag["query_hits"] or {}
                feature_rows.append({
                    "feature":name,
                    "frontend":frontend,
                    "source_column":column,
                    "label":label,
                    "auc":auc["auc"],
                    "auc_star":auc["auc_star"],
                    "direction":auc["direction"],
                    "top1":hits.get("top1"),
                    "top3":hits.get("top3"),
                    "top5":hits.get("top5"),
                    "top10":hits.get("top10"),
                    "top20":hits.get("top20"),
                    "queries_with_finite_feature":(
                        diag["queries_with_finite_feature"]
                        if diag["query_hits"] is not None else 0
                    ),
                })
            report["common_feature_diagnostics"][name][frontend]=block

    labeled=pd.concat(labeled_frames,ignore_index=True).sort_values(
        ["frontend","query_id","pool_rank"],kind="mergesort"
    )
    selected=pd.concat(selected_frames,ignore_index=True).sort_values(
        ["frontend","query_id"],kind="mergesort"
    )
    feature_diag=pd.DataFrame(feature_rows).sort_values(
        ["label","feature","frontend"],kind="mergesort"
    )

    out=resolve(cfg["output_root"])
    labeled_path=out/"lf1_2c_labeled_pairs.csv"
    selected_path=out/"lf1_2c_selected_projection.csv"
    feature_path=out/"lf1_2c_common_feature_diagnostics.csv"
    files={
        labeled_path:deterministic_csv(labeled),
        selected_path:deterministic_csv(selected),
        feature_path:deterministic_csv(feature_diag),
    }
    for name,path,frame in (
        ("labeled_pairs",labeled_path,labeled),
        ("selected_projection",selected_path,selected),
        ("common_feature_diagnostics",feature_path,feature_diag),
    ):
        report["outputs"][name]={
            "path":str(path.relative_to(ROOT)),
            "sha256":hashlib.sha256(files[path]).hexdigest(),
            "rows":int(len(frame)),
        }
    report_path=out/"lf1_2c_comparison_report.json"
    files[report_path]=deterministic_json(report)
    write_immutable(files)

    print("Status:",report["status"])
    for frontend in ("orb","xfeat_sparse","xfeat_star"):
        fb=report["frontends"][frontend]
        print(
            f"{frontend}: pairs={fb['pair_counts']['pairs']} "
            f"homography_ok={fb['pair_counts']['homography_ok']} "
            f"finite={fb['pair_counts']['finite_projection']}"
        )
        for label in labels:
            retention=fb["labels"][label]["homography_retention"]
            rankings=fb["labels"][label]["rankings"]
            print(
                f"  {label}: positive_pair_retention="
                f"{retention['positive_pair_retention']:.4f} "
                f"negative_pair_rejection="
                f"{retention['negative_pair_rejection']:.4f} "
                f"query_positive_retention="
                f"{retention['query_positive_retention']:.4f}"
            )
            for selector in selectors:
                h=rankings[selector]["hits"]
                print(
                    f"    {selector}: top1={h['top1']}/403 "
                    f"top3={h['top3']}/403 top5={h['top5']}/403 "
                    f"top10={h['top10']}/403"
                )
        sel=fb["projection"]["selected_valid_then_inliers"]
        oracle=fb["projection"]["oracle_best_finite_projection"]
        print(
            f"  selected projection: median={sel['median_m']:.3f} "
            f"rmse={sel['rmse_m']:.3f} p95={sel['p95_m']:.3f} "
            f"le40={sel['le40']}/{sel['count']} "
            f"le80={sel['le80']}/{sel['count']} "
            f"gt200={sel['gt200']}/{sel['count']}"
        )
        print(
            f"  oracle projection: median={oracle['median_m']:.3f} "
            f"le40={oracle['le40']}/{oracle['count']} "
            f"le80={oracle['le80']}/{oracle['count']}"
        )

    for label in labels:
        print(f"Top common features for {label}:")
        subset=feature_diag[feature_diag["label"]==label]
        for frontend in ("orb","xfeat_sparse","xfeat_star"):
            top=subset[subset["frontend"]==frontend].sort_values(
                ["auc_star","feature"],
                ascending=[False,True],
                kind="mergesort",
            ).head(5)
            print(f"  {frontend}:")
            for row in top.itertuples(index=False):
                print(
                    f"    {row.feature}: auc*={row.auc_star:.4f} "
                    f"dir={row.direction} top1={int(row.top1)}/403 "
                    f"top5={int(row.top5)}/403"
                )

    print("Blind stress loaded: False")
    print("Frontend winner promoted: False")
    print("Report:",report_path)
    print(
        "Comparison diagnostics SHA256:",
        report["outputs"]["common_feature_diagnostics"]["sha256"],
    )


if __name__=="__main__":
    main()
