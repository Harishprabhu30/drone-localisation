#!/usr/bin/env python3
"""CV2.5: full403 post-freeze validation of SIFT+LightGlue vs frozen ORB controls."""
from __future__ import annotations

import argparse, hashlib, json, math, sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
import yaml

from uavloc.evaluation.reference import load_reference_xy

ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/crossview_scale_rotation_domain_cv2_5.yaml"
REQ = {"query_id","pool_rank","tile_id","inliers","homography_ok","finite_projection","projected_easting","projected_northing"}
TILE_REQ = {"tile_id","left_easting","right_easting","bottom_northing","top_northing","center_easting","center_northing"}


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


def deterministic_json(v: dict[str, Any]) -> bytes:
    return (json.dumps(v, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_immutable(files: dict[Path, bytes]) -> None:
    bad = [str(p) for p,d in files.items() if p.exists() and p.read_bytes() != d]
    if bad:
        raise RuntimeError("Refusing to overwrite differing CV2.5 outputs: " + ", ".join(bad))
    for p,d in files.items():
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("xb") as f:
                f.write(d)


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    m = {"true":True,"false":False,"1":True,"0":False,"yes":True,"no":False,"y":True,"n":False,"t":True,"f":False}
    out = s.astype(str).str.strip().str.lower().map(m)
    if out.isna().any():
        raise RuntimeError("Unparseable boolean")
    return out.astype(bool)


def require(frame: pd.DataFrame, cols: set[str], label: str) -> None:
    missing = sorted(cols - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "CV2.5":
        raise RuntimeError("Expected CV2.5 config")
    if cfg["ranking"]["selector"] != "inliers" or cfg["ranking"]["tie_break"] != ["pool_rank_asc","tile_id_asc"]:
        raise RuntimeError("Ranking contract changed")
    b = cfg["scientific_boundary"]
    if b["all_reference_free_freezes_verified_before_truth"] is not True or b["development_reference_allowed"] is not True:
        raise RuntimeError("Reference boundary missing")
    for k in ("frontend_evidence_mutable","scale_m2_applied_to_sift","threshold_tuning_allowed","selector_formula_mutable","jetson_cuda_evidence_used","blind_stress_loaded","state_changes_allowed","temporal_fusion_allowed"):
        if b[k] is not False:
            raise RuntimeError(f"{k} must be false")


def verify_inputs(cfg: dict[str, Any]) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    blocks = [("orb_original",cfg["original_orb"]),("orb_scale_m2",cfg["scale_m2_orb"]),("sift_lightglue",cfg["sift_lightglue"])]
    out = {}
    for name, block in blocks:
        p = resolve(block["evidence"])
        if sha256_file(p) != block["expected_sha256"]:
            raise RuntimeError(f"{name} evidence hash mismatch")
        f = pd.read_csv(p)
        if len(f) != int(block["expected_pairs"]) or f["query_id"].nunique() != int(block["expected_queries"]):
            raise RuntimeError(f"{name} evidence count mismatch")
        out[name] = f
    sblock = cfg["sift_lightglue"]
    rep = json.loads(resolve(sblock["report"]).read_text())
    if rep.get("status") != sblock["expected_status"]:
        raise RuntimeError("CV2.4 report status mismatch")
    if rep["provenance"]["evidence_sha256"] != sblock["expected_sha256"]:
        raise RuntimeError("CV2.4 report evidence hash mismatch")
    if rep["scientific_boundary"]["reference_loaded"] is not False:
        raise RuntimeError("CV2.4 reference boundary violated")

    key = ["query_id","pool_rank","tile_id"]
    for k in out:
        out[k]["query_id"] = pd.to_numeric(out[k]["query_id"], errors="raise").astype(int)
        out[k]["pool_rank"] = pd.to_numeric(out[k]["pool_rank"], errors="raise").astype(int)
        out[k]["tile_id"] = out[k]["tile_id"].astype(str)
        out[k] = out[k].sort_values(key, kind="mergesort").reset_index(drop=True)
    if not out["orb_original"][key].equals(out["orb_scale_m2"][key]) or not out["orb_original"][key].equals(out["sift_lightglue"][key]):
        raise RuntimeError("Frozen pair identities differ")
    return out["orb_original"], out["orb_scale_m2"], out["sift_lightglue"]


def verify_tiles(cfg: dict[str, Any]) -> Path:
    rp = resolve(cfg["map"]["rg2_0_report"])
    tp = resolve(cfg["map"]["tile_geometry"])
    rep = json.loads(rp.read_text())
    if rep.get("status") != "PASS_RG2_0_MAP_TOPOLOGY_PREFLIGHT":
        raise RuntimeError("RG2.0 topology report not PASS")
    if sha256_file(tp) != rep["provenance"]["tile_geometry_sha256"]:
        raise RuntimeError("Tile geometry changed")
    return tp


def attach(frame: pd.DataFrame, tiles: pd.DataFrame, ref: pd.DataFrame, name: str) -> pd.DataFrame:
    require(frame, REQ, name)
    require(tiles, TILE_REQ, "tile geometry")
    x = frame.copy()
    x["homography_ok"] = bool_series(x["homography_ok"])
    x["finite_projection"] = bool_series(x["finite_projection"])
    x["frontend"] = name
    t = tiles[list(TILE_REQ)].copy()
    t["tile_id"] = t["tile_id"].astype(str)
    r = ref[["gt_x","gt_y"]].reset_index()
    out = x.merge(t,on="tile_id",how="left",validate="many_to_one").merge(r,on="query_id",how="left",validate="many_to_one")
    needed = list(TILE_REQ-{"tile_id"}) + ["gt_x","gt_y"]
    if out[needed].isna().any().any():
        raise RuntimeError(f"{name} reference join incomplete")
    out["contains_query"] = ((out["gt_x"]>=out["left_easting"])&(out["gt_x"]<=out["right_easting"])&(out["gt_y"]>=out["bottom_northing"])&(out["gt_y"]<=out["top_northing"]))
    out["center_error_m"] = np.hypot(out["center_easting"]-out["gt_x"], out["center_northing"]-out["gt_y"])
    out["center_le40"] = out["center_error_m"] <= 40.0
    out["center_le80"] = out["center_error_m"] <= 80.0
    px = pd.to_numeric(out["projected_easting"],errors="coerce")
    py = pd.to_numeric(out["projected_northing"],errors="coerce")
    finite = np.isfinite(px)&np.isfinite(py)
    out["projection_error_m"] = np.nan
    out.loc[finite,"projection_error_m"] = np.hypot(px[finite]-out.loc[finite,"gt_x"], py[finite]-out.loc[finite,"gt_y"])
    return out


def rank(g: pd.DataFrame) -> pd.DataFrame:
    x = g.copy()
    x["_score"] = pd.to_numeric(x["inliers"],errors="coerce").fillna(-1.0)
    return x.sort_values(["_score","pool_rank","tile_id"],ascending=[False,True,True],kind="mergesort")


def proj_summary(values: pd.Series, total: int) -> dict[str,Any]:
    a = pd.to_numeric(values,errors="coerce").to_numpy(dtype=float)
    finite = a[np.isfinite(a)]
    nonfinite = total-len(finite)
    return {
        "finite_count": int(len(finite)),
        "nonfinite_count": int(nonfinite),
        "median_m": float(np.median(finite)) if len(finite) else None,
        "rmse_m": float(np.sqrt(np.mean(finite*finite))) if len(finite) else None,
        "p95_m": float(np.quantile(finite,.95)) if len(finite) else None,
        "le40": int((finite<=40).sum()),
        "le80": int((finite<=80).sum()),
        "gt200": int((finite>200).sum()),
        "unsafe_tail": int(nonfinite + (finite>200).sum()),
    }


def summarize(frame: pd.DataFrame, topks: list[int]) -> tuple[dict[str,Any],pd.DataFrame]:
    hits = {lab:{f"top{k}":0 for k in topks} for lab in ("contains_query","center_le40","center_le80")}
    rows=[]
    for qid,g in frame.groupby("query_id",sort=True):
        o=rank(g).reset_index(drop=True)
        s=o.iloc[0]
        for lab in hits:
            for k in topks:
                hits[lab][f"top{k}"] += int(o.head(k)[lab].astype(bool).any())
        rows.append({
            "query_id":int(qid),"tile_id":str(s["tile_id"]),"pool_rank":int(s["pool_rank"]),
            "inliers":float(s["inliers"]),"contains_query":bool(s["contains_query"]),
            "center_le40":bool(s["center_le40"]),"center_le80":bool(s["center_le80"]),
            "projection_error_m":float(s["projection_error_m"]) if pd.notna(s["projection_error_m"]) else np.nan,
            "top1_top2_inlier_margin": float(o.iloc[0]["inliers"])-float(o.iloc[1]["inliers"]) if len(o)>1 else np.nan,
        })
    sel=pd.DataFrame(rows)
    oracle=(frame.replace([np.inf,-np.inf],np.nan).dropna(subset=["projection_error_m"]).groupby("query_id")["projection_error_m"].min())
    return {
        "queries":int(len(sel)),
        "ranking":hits,
        "selected_top1":{"contains_query":int(sel["contains_query"].sum()),"center_le40":int(sel["center_le40"].sum()),"center_le80":int(sel["center_le80"].sum())},
        "selected_projection":proj_summary(sel["projection_error_m"],len(sel)),
        "oracle_best_finite_projection":proj_summary(oracle,len(sel)),
        "median_top1_top2_inlier_margin":float(pd.to_numeric(sel["top1_top2_inlier_margin"],errors="coerce").median()),
    }, sel


def compare(scale: pd.DataFrame, sift: pd.DataFrame) -> tuple[dict[str,Any],pd.DataFrame]:
    a=scale.add_prefix("scale_").rename(columns={"scale_query_id":"query_id"})
    b=sift.add_prefix("sift_").rename(columns={"sift_query_id":"query_id"})
    m=a.merge(b,on="query_id",validate="one_to_one")
    m["winner_changed"]=m["scale_tile_id"]!=m["sift_tile_id"]
    m["containment_rescue"]=~m["scale_contains_query"] & m["sift_contains_query"]
    m["containment_damage"]=m["scale_contains_query"] & ~m["sift_contains_query"]
    m["le80_rescue"]=~m["scale_center_le80"] & m["sift_center_le80"]
    m["le80_damage"]=m["scale_center_le80"] & ~m["sift_center_le80"]
    se=pd.to_numeric(m["scale_projection_error_m"],errors="coerce")
    fe=pd.to_numeric(m["sift_projection_error_m"],errors="coerce")
    scale_unsafe=(~np.isfinite(se)) | (se>200)
    sift_unsafe=(~np.isfinite(fe)) | (fe>200)
    m["unsafe_rescue"]=scale_unsafe & ~sift_unsafe
    m["unsafe_damage"]=~scale_unsafe & sift_unsafe
    return {
        "winner_changed":int(m["winner_changed"].sum()),
        "containment_rescues":int(m["containment_rescue"].sum()),
        "containment_damages":int(m["containment_damage"].sum()),
        "center_le80_rescues":int(m["le80_rescue"].sum()),
        "center_le80_damages":int(m["le80_damage"].sum()),
        "unsafe_tail_rescues":int(m["unsafe_rescue"].sum()),
        "unsafe_tail_damages":int(m["unsafe_damage"].sum()),
    },m


def evaluate_gate(scale: dict[str,Any], sift: dict[str,Any], contract: dict[str,Any]) -> dict[str,Any]:
    top1_delta={k:int(sift["selected_top1"][k]-scale["selected_top1"][k]) for k in ("contains_query","center_le40","center_le80")}
    le40_delta=int(sift["selected_projection"]["le40"]-scale["selected_projection"]["le40"])
    unsafe_reduction=int(scale["selected_projection"]["unsafe_tail"]-sift["selected_projection"]["unsafe_tail"])
    preservation=(
        all(v>=-int(contract["max_top1_regression_queries_each_target"]) for v in top1_delta.values())
        and le40_delta>=-int(contract["max_selected_projection_le40_regression_queries"])
        and unsafe_reduction>=-int(contract["max_unsafe_tail_increase_queries"])
    )
    mg=contract["meaningful_gain"]
    meaningful=(
        max(top1_delta.values())>=int(mg["min_top1_improvement_queries_any_target"])
        or le40_delta>=int(mg["min_selected_projection_le40_improvement_queries"])
        or unsafe_reduction>=int(mg["min_unsafe_tail_reduction_queries"])
    )
    return {
        "top1_deltas_sift_minus_scale_m2":top1_delta,
        "selected_projection_le40_delta":le40_delta,
        "unsafe_tail_reduction":unsafe_reduction,
        "preservation_pass":bool(preservation),
        "meaningful_gain_pass":bool(meaningful),
        "pass":bool(preservation and meaningful),
    }


def main() -> None:
    ap=argparse.ArgumentParser(description="CV2.5 full403 post-freeze SIFT+LightGlue validation")
    ap.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    args=ap.parse_args()
    cfg_path=resolve(args.config)
    cfg=yaml.safe_load(cfg_path.read_text())
    verify_config(cfg)
    orig_raw,scale_raw,sift_raw=verify_inputs(cfg)
    tile_path=verify_tiles(cfg)

    print("All frozen full403 evidence sets verified before reference attachment.")
    ref_path=resolve(cfg["development"]["reference_attachment"])
    print("Attaching development reference for CV2.5 only.")
    ref=load_reference_xy(ref_path)
    tiles=pd.read_csv(tile_path)

    frames={
        "orb_original":attach(orig_raw,tiles,ref,"orb_original"),
        "orb_scale_m2":attach(scale_raw,tiles,ref,"orb_scale_m2"),
        "sift_lightglue":attach(sift_raw,tiles,ref,"sift_lightglue"),
    }
    topks=[int(x) for x in cfg["ranking"]["topk"]]
    summaries={}
    selected={}
    for name,frame in frames.items():
        summaries[name],selected[name]=summarize(frame,topks)

    comparison,audit=compare(selected["orb_scale_m2"],selected["sift_lightglue"])
    gate=evaluate_gate(summaries["orb_scale_m2"],summaries["sift_lightglue"],cfg["validation_gate_vs_scale_m2"])
    if gate["pass"]:
        decision="PROMOTE_SIFT_LIGHTGLUE_AS_STRONGER_DEVELOPMENT_FRONTEND"
    elif gate["preservation_pass"]:
        decision="HOLD_SCALE_M2_ORB_PRIMARY_SIFT_LIGHTGLUE_ALTERNATIVE"
    else:
        decision="CLOSE_CV2_SIFT_LIGHTGLUE_NONROBUST_FULL403"

    rows=[]
    for name in ("orb_original","orb_scale_m2","sift_lightglue"):
        s=summaries[name]; p=s["selected_projection"]; o=s["oracle_best_finite_projection"]
        rows.append({
            "frontend":name,
            "top1_contains_query":s["selected_top1"]["contains_query"],
            "top1_center_le40":s["selected_top1"]["center_le40"],
            "top1_center_le80":s["selected_top1"]["center_le80"],
            "projection_finite":p["finite_count"],"projection_nonfinite":p["nonfinite_count"],
            "projection_median_m":p["median_m"],"projection_rmse_m":p["rmse_m"],"projection_p95_m":p["p95_m"],
            "projection_le40":p["le40"],"projection_le80":p["le80"],"projection_gt200":p["gt200"],"projection_unsafe_tail":p["unsafe_tail"],
            "oracle_projection_median_m":o["median_m"],"oracle_projection_le40":o["le40"],"oracle_projection_le80":o["le80"],
            "median_top1_top2_inlier_margin":s["median_top1_top2_inlier_margin"],
        })
    summary=pd.DataFrame(rows)

    report={
        "stage":"CV2.5",
        "status":"PASS_CV2_5_POSTFREEZE_SIFT_LIGHTGLUE_FULL403_VALIDATION",
        "summaries":summaries,
        "sift_vs_scale_m2_selected_comparison":comparison,
        "validation_gate":{"contract":cfg["validation_gate_vs_scale_m2"],"result":gate},
        "decision":decision,
        "scientific_boundary":{
            "freezes_verified_before_reference":True,"development_reference_used_postfreeze":True,
            "scale_m2_applied_to_sift":False,"threshold_tuned":False,"selector_formula_changed":False,
            "jetson_cuda_evidence_used":False,"blind_stress_loaded":False,"state_modified":False,
        },
        "provenance":{
            "config_sha256":sha256_file(cfg_path),
            "original_orb_sha256":sha256_file(resolve(cfg["original_orb"]["evidence"])),
            "scale_m2_orb_sha256":sha256_file(resolve(cfg["scale_m2_orb"]["evidence"])),
            "sift_lightglue_sha256":sha256_file(resolve(cfg["sift_lightglue"]["evidence"])),
            "reference_attachment_sha256":sha256_file(ref_path),
            "tile_geometry_sha256":sha256_file(tile_path),
        },
    }

    outroot=resolve(cfg["output_root"])
    summary_path=outroot/"cv2_5_full403_frontend_summary.csv"
    audit_path=outroot/"cv2_5_sift_vs_scale_selected_audit.csv"
    report_path=outroot/"cv2_5_validation_report.json"
    write_immutable({
        summary_path:deterministic_csv(summary),
        audit_path:deterministic_csv(audit),
        report_path:deterministic_json(report),
    })

    print("Status:",report["status"])
    print()
    print(summary.to_string(index=False))
    print()
    print("Winner changed:",comparison["winner_changed"])
    print("Containment rescues/damages:",f"{comparison['containment_rescues']}/{comparison['containment_damages']}")
    print("Center<=80 rescues/damages:",f"{comparison['center_le80_rescues']}/{comparison['center_le80_damages']}")
    print("Unsafe-tail rescues/damages:",f"{comparison['unsafe_tail_rescues']}/{comparison['unsafe_tail_damages']}")
    print("Validation gate:","PASS" if gate["pass"] else "FAIL")
    print("Gate result:",json.dumps(gate,sort_keys=True))
    print("Decision:",decision)
    print("Jetson CUDA evidence used: False")
    print("Threshold tuned: False")
    print("Report:",report_path)


if __name__=="__main__":
    main()
