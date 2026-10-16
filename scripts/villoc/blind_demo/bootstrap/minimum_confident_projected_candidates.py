#!/usr/bin/env python3
"""Current-stack projected-candidate implementation of minimum_confident_v2.

Consumes already-projected reference-free absolute candidates from the promoted
SIFT+LightGlue frontend. It does NOT recompute ORB/SIFT geometry.

The temporal/bootstrap mathematics is intentionally the same transferred policy:
- Top4 candidates per query
- four spatially separated causal evidence frames
- least-squares 2D similarity
- >=3 unique tile IDs
- visual span >=100 px
- map span >=50 m
- median residual <=51.2 m
- maximum residual <=102.4 m
- quarter/quarter policy selected later by the final-policy adapter

Outputs preserve the R5.1 file contract so the existing canonical map-state
adapters and temporal-authority routers can be reused unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FORBIDDEN_EXACT = {
    "latitude","longitude","lat","lon","reference_x_m","reference_y_m",
    "ground_truth_x_m","ground_truth_y_m","oracle_tile_id","gt_easting",
    "gt_northing","projected_error_m","tile_center_error_m","error_m","rmse","p95",
}
FORBIDDEN_PART = (
    "hit_le_","ground_truth","oracle_","gt_error","reference_enu",
    "postfreeze","eval_error",
)


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def assert_reference_free(df: pd.DataFrame,label: str) -> None:
    bad=[]
    for c in df.columns:
        low=str(c).strip().lower()
        if low in FORBIDDEN_EXACT or any(x in low for x in FORBIDDEN_PART):
            bad.append(str(c))
    if bad:
        raise RuntimeError(f"{label} contains forbidden evaluation columns: {sorted(bad)}")


def dynamic_import(name: str,path: Path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper: {path}")
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod


def make_hypothesis(helper,update_qid,evidence_qids,visual_xy,combo,cfg,hid):
    map_xy=np.asarray(
        [[float(c["projected_easting"]),float(c["projected_northing"])] for c in combo],
        dtype=float,
    )
    if not np.all(np.isfinite(map_xy)):
        return None
    tile_ids=[str(c["tile_id"]) for c in combo]
    if len(set(tile_ids)) < int(cfg["minimum_unique_tile_ids"]):
        return None
    map_span=helper.span(map_xy)
    if map_span < float(cfg["minimum_map_span_m"]):
        return None
    try:
        model=helper.fit_similarity(visual_xy,map_xy)
    except ValueError:
        return None
    residual=np.linalg.norm(helper.apply_similarity(visual_xy,model)-map_xy,axis=1)
    med=float(np.median(residual)); mx=float(np.max(residual))
    if med > float(cfg["median_measurement_residual_limit_m"]):
        return None
    if mx > float(cfg["maximum_measurement_residual_limit_m"]):
        return None
    return {
        "hypothesis_id":int(hid),
        "update_query_id":int(update_qid),
        "evidence_query_ids":[int(x) for x in evidence_qids],
        "tile_ids":tile_ids,
        "candidate_choice_ranks":[int(c["candidate_choice_rank"]) for c in combo],
        "map_observations":[(float(x),float(y)) for x,y in map_xy],
        "map_unique_tile_ids":int(len(set(tile_ids))),
        "map_span_m":float(map_span),
        "median_projected_residual_m":med,
        "max_projected_residual_m":mx,
        "sum_dino_rank":float(sum(float(c["rank"]) for c in combo)),
        "sum_hybrid_rank":float(sum(float(c["candidate_choice_rank"]) for c in combo)),
        "all_projected_inside_tile":bool(all(bool(c["projected_inside_tile"]) for c in combo)),
        **model,
    }


def pareto(rows):
    keys=("median_projected_residual_m","sum_hybrid_rank","sum_dino_rank")
    out=[]
    for i,a in enumerate(rows):
        dominated=False
        for j,b in enumerate(rows):
            if i==j: continue
            if all(float(b[k])<=float(a[k]) for k in keys) and any(float(b[k])<float(a[k]) for k in keys):
                dominated=True; break
        if not dominated: out.append(a)
    return out


def model_payload(h):
    return {k:float(h[k]) for k in (
        "a_real","a_imag","b_real","b_imag","scale_m_per_visual_px","rotation_deg"
    )}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo-root",type=Path,required=True)
    p.add_argument("--run-root",type=Path,required=True)
    p.add_argument("--architecture-contract",type=Path,required=True)
    p.add_argument("--expected-contract-sha256",required=True)
    p.add_argument("--out-root",type=Path,required=True)
    p.add_argument("--candidate-csv",type=Path,required=True)
    p.add_argument("--relative-csv",type=Path,required=True)
    p.add_argument("--manifest-csv",type=Path,required=True)
    args=p.parse_args()

    repo=args.repo_root.resolve()
    out=args.out_root.resolve(); out.mkdir(parents=True,exist_ok=True)
    contract_path=args.architecture_contract.resolve()
    actual_sha=sha256(contract_path)
    if actual_sha != str(args.expected_contract_sha256):
        raise RuntimeError(f"Architecture contract SHA mismatch: {actual_sha}")
    contract=json.loads(contract_path.read_text())
    if contract.get("status")!="PASS_CURRENT_STACK_PROJECTED_CANDIDATE_AUTHORITY_CONTRACT":
        raise RuntimeError("Unexpected current-stack architecture contract")

    top_m=int(contract["absolute_observation"]["candidate_count"])
    evidence_frames=int(contract["bootstrap_hypothesis"]["evidence_queries"])
    if top_m!=4 or evidence_frames!=4:
        raise RuntimeError("Current stack expects Top4 / four evidence frames")
    cfg=contract["bootstrap_hypothesis"]
    helper_path=repo/"scripts/villoc/blind_demo/bootstrap/minimum_confident_v2_backend.py"
    helper=dynamic_import("current_stack_minconf_helpers",helper_path)

    cand=pd.read_csv(args.candidate_csv.resolve())
    rel=pd.read_csv(args.relative_csv.resolve())
    manifest=pd.read_csv(args.manifest_csv.resolve())
    assert_reference_free(cand,"absolute candidates")
    assert_reference_free(rel,"relative trajectory")
    assert_reference_free(manifest,"localization manifest")

    required_c={
        "query_id","rank","tile_id","inliers","homography_ok","finite_projection",
        "projected_inside_tile","projected_easting","projected_northing",
    }
    if required_c-set(cand):
        raise RuntimeError(f"candidate table missing {sorted(required_c-set(cand))}")
    if {"token0_id","visual_x_px","visual_y_px"}-set(rel):
        raise RuntimeError("relative trajectory schema changed")
    if {"query_id","timestamp_s"}-set(manifest):
        raise RuntimeError("manifest schema changed")

    cand["query_id"]=pd.to_numeric(cand["query_id"],errors="raise").astype(int)
    cand["rank"]=pd.to_numeric(cand["rank"],errors="raise").astype(int)
    cand["inliers"]=pd.to_numeric(cand["inliers"],errors="raise").astype(int)
    cand["tile_id"]=cand["tile_id"].astype(str)
    for c in ("homography_ok","finite_projection","projected_inside_tile"):
        cand[c]=cand[c].astype(str).str.lower().isin({"true","1","yes","y","t"})
    cand=cand.sort_values(
        ["query_id","inliers","rank","tile_id"],
        ascending=[True,False,True,True],
        kind="mergesort",
    )
    cand["candidate_choice_rank"]=cand.groupby("query_id").cumcount()+1
    cand=cand[cand["candidate_choice_rank"]<=top_m].copy()

    rel["query_id"]=pd.to_numeric(rel["token0_id"],errors="raise").astype(int)
    rel["visual_x_px"]=pd.to_numeric(rel["visual_x_px"],errors="raise")
    rel["visual_y_px"]=pd.to_numeric(rel["visual_y_px"],errors="raise")
    manifest["query_id"]=pd.to_numeric(manifest["query_id"],errors="raise").astype(int)
    manifest["timestamp_s"]=pd.to_numeric(manifest["timestamp_s"],errors="coerce")
    rel=rel.merge(
        manifest[["query_id","timestamp_s"]].drop_duplicates("query_id"),
        on="query_id",how="left",validate="one_to_one",
    )

    common=sorted(set(cand["query_id"]) & set(rel["query_id"]))
    if not common: raise RuntimeError("No common query IDs")
    short=cand.groupby("query_id").size()
    short=short[short<top_m]
    if len(short): raise RuntimeError(f"Queries with fewer than Top4: {short.head().to_dict()}")

    observations=cand[[
        "query_id","rank","tile_id","candidate_choice_rank","matches","inliers",
        "homography_ok","finite_projection","projected_inside_tile",
        "projected_easting","projected_northing"
    ]].copy()
    ts=manifest.set_index("query_id")["timestamp_s"]
    observations.insert(1,"timestamp_s",observations["query_id"].map(ts))
    observations=observations.rename(columns={"matches":"stored_matches","inliers":"stored_inliers"})
    observations["valid_projected_observation"]=(
        observations["homography_ok"] & observations["finite_projection"]
        & np.isfinite(pd.to_numeric(observations["projected_easting"],errors="coerce"))
        & np.isfinite(pd.to_numeric(observations["projected_northing"],errors="coerce"))
    )

    by_q={
        int(q):g[g["valid_projected_observation"]].to_dict("records")
        for q,g in observations.groupby("query_id",sort=True)
    }
    rel=rel[rel["query_id"].isin(common)].sort_values("query_id").reset_index(drop=True)
    rel_by_q=rel.set_index("query_id")

    updates=[]; hypotheses=[]; leaders={}; hid=1
    for q in common:
        prefix=rel[rel["query_id"]<=q]
        row={
            "update_query_id":int(q),"timestamp_s":float(ts.get(q,np.nan)),
            "evidence_query_ids":"","visual_span_px":math.nan,
            "enumerated_hypotheses":0,"geometric_pass_hypotheses":0,
            "pareto_hypotheses":0,"blind_leader_hypothesis_id":math.nan,
            "blind_leader_scale":math.nan,"blind_leader_rotation_deg":math.nan,
            "action":"",
        }
        evidence=helper.select_evidence(prefix,evidence_frames)
        if len(evidence)!=evidence_frames:
            row["action"]="WAIT_INSUFFICIENT_EVIDENCE_FRAMES"; updates.append(row); continue
        visual_xy=rel_by_q.loc[evidence,["visual_x_px","visual_y_px"]].to_numpy(float)
        visual_span=helper.span(visual_xy)
        row["evidence_query_ids"]=",".join(map(str,evidence)); row["visual_span_px"]=visual_span
        if visual_span<float(cfg["minimum_visual_span_px"]):
            row["action"]="WAIT_VISUAL_SPAN"; updates.append(row); continue
        lists=[by_q.get(int(eq),[]) for eq in evidence]
        if any(len(x)==0 for x in lists):
            row["action"]="NO_VALID_SUBTILE_CANDIDATE"; updates.append(row); continue
        row["enumerated_hypotheses"]=int(np.prod([len(x) for x in lists]))
        admissible=[]
        for combo in itertools.product(*lists):
            h=make_hypothesis(helper,q,evidence,visual_xy,combo,cfg,hid); hid+=1
            if h is not None: admissible.append(h)
        row["geometric_pass_hypotheses"]=len(admissible)
        if not admissible:
            row["action"]="NO_GEOMETRICALLY_ADMISSIBLE_HYPOTHESIS"; updates.append(row); continue
        front=pareto(admissible); row["pareto_hypotheses"]=len(front)
        leader=min(front,key=lambda h:(
            h["median_projected_residual_m"],h["sum_hybrid_rank"],
            h["sum_dino_rank"],tuple(h["tile_ids"])
        ))
        leaders[int(q)]=leader
        row["blind_leader_hypothesis_id"]=int(leader["hypothesis_id"])
        row["blind_leader_scale"]=float(leader["scale_m_per_visual_px"])
        row["blind_leader_rotation_deg"]=float(leader["rotation_deg"])
        row["action"]="BLIND_LEADER_AVAILABLE"
        front_ids={int(h["hypothesis_id"]) for h in front}
        for h in admissible:
            hypotheses.append(helper.serial_hypothesis(
                h,int(h["hypothesis_id"]) in front_ids,
                int(h["hypothesis_id"])==int(leader["hypothesis_id"])
            ))
        updates.append(row)

    updates_df=pd.DataFrame(updates)
    seeds=[q for q in common if q in leaders]
    if not seeds: raise RuntimeError("No projected-candidate blind leader produced")
    seed_q=int(seeds[0]); seed=leaders[seed_q]

    family=contract["candidate_policy_family"]
    support_required=int(family["maturity_support_required"])
    policies={
        f"activate_{a.replace('_spacing','')}_track_{t.replace('_spacing','')}":{
            "activation_threshold_m":float(av),"tracking_threshold_m":float(tv)
        }
        for a,av in family["activation_thresholds"].items()
        for t,tv in family["tracking_thresholds"].items()
    }

    policy_rows=[]; policy_results={}
    for policy_name,policy in sorted(policies.items()):
        mode="ACQUISITION"; active=seed; active_source_q=seed_q
        streak=0; matured_at=None; maturity_model=None
        policy_rows.append({
            "policy":policy_name,
            "activation_threshold_m":policy["activation_threshold_m"],
            "tracking_threshold_m":policy["tracking_threshold_m"],
            "support_required":support_required,"update_query_id":seed_q,
            "mode_before":"ACQUISITION","minimum_innovation_m":math.nan,
            "consistency_streak_before":0,"consistency_streak_after":0,
            "action":"SEED","mode_after":"ACQUISITION","matured_now":False,
            "matured_at_query_id":math.nan,"innovation_best_tile_id":"",
            "innovation_best_choice_rank":math.nan,
            "active_source_update_q_before":math.nan,
            "active_source_update_q_after":seed_q,
            "active_hypothesis_id_after":int(active["hypothesis_id"]),
        })
        for q in [x for x in common if x>seed_q]:
            mode_before=mode; source_before=active_source_q; streak_before=streak
            current_leader=leaders.get(int(q))
            current_visual=rel_by_q.loc[[q],["visual_x_px","visual_y_px"]].to_numpy(float)
            prior_prediction=helper.apply_similarity(current_visual,active)[0]
            current=observations[
                (observations["query_id"]==q)&observations["valid_projected_observation"]
            ].copy()
            if len(current):
                current["innovation_m"]=np.hypot(
                    current["projected_easting"]-float(prior_prediction[0]),
                    current["projected_northing"]-float(prior_prediction[1]),
                )
                best=current.sort_values(
                    ["innovation_m","candidate_choice_rank"],kind="mergesort"
                ).iloc[0]
                min_innovation=float(best["innovation_m"])
                best_tile=str(best["tile_id"]); best_choice=int(best["candidate_choice_rank"])
            else:
                min_innovation=math.inf; best_tile=""; best_choice=math.nan
            matured_now=False
            if mode=="ACQUISITION":
                if current_leader is None:
                    action="ACQUISITION_NO_LEADER"; streak=0
                else:
                    active=current_leader; active_source_q=int(q)
                    streak=streak+1 if min_innovation<=policy["activation_threshold_m"] else 0
                    action="ACQUISITION_ACCEPT"
                    if streak>=support_required:
                        mode="TRACKING"; matured_at=int(q); maturity_model=dict(active); matured_now=True
            else:
                if current_leader is None:
                    action="TRACKING_HOLD_NO_LEADER"
                elif min_innovation<=policy["tracking_threshold_m"]:
                    action="TRACKING_ACCEPT"; active=current_leader; active_source_q=int(q)
                else:
                    action="TRACKING_HOLD_INNOVATION"
            policy_rows.append({
                "policy":policy_name,
                "activation_threshold_m":policy["activation_threshold_m"],
                "tracking_threshold_m":policy["tracking_threshold_m"],
                "support_required":support_required,"update_query_id":int(q),
                "mode_before":mode_before,"minimum_innovation_m":min_innovation,
                "consistency_streak_before":streak_before,
                "consistency_streak_after":streak,"action":action,
                "mode_after":mode,"matured_now":matured_now,
                "matured_at_query_id":matured_at if matured_at is not None else math.nan,
                "innovation_best_tile_id":best_tile,
                "innovation_best_choice_rank":best_choice,
                "active_source_update_q_before":source_before,
                "active_source_update_q_after":active_source_q,
                "active_hypothesis_id_after":int(active["hypothesis_id"]),
            })
        state="PROVISIONAL_ABSOLUTE_LOCK" if matured_at is not None else "NO_PROVISIONAL_LOCK"
        policy_results[policy_name]={
            "localization_state":state,
            "activation_threshold_m":policy["activation_threshold_m"],
            "tracking_threshold_m":policy["tracking_threshold_m"],
            "maturity_support_required":support_required,
            "matured_at_query_id":matured_at,
            "maturity_map_state":model_payload(maturity_model) if maturity_model else None,
            "final_active_source_update_q":int(active_source_q),
            "final_active_map_state":model_payload(active),
            "forbidden_state_emitted":False,
        }

    policy_timeline=pd.DataFrame(policy_rows)
    paths={
        "observations":out/"r5_1_blind_top4_subtile_observations.csv",
        "hypotheses":out/"r5_1_blind_subtile_hypotheses.csv",
        "leader_updates":out/"r5_1_blind_leader_updates.csv",
        "policy_timeline":out/"r5_1_blind_policy_timeline.csv",
        "policy_results":out/"r5_1_blind_policy_results.json",
    }
    observations.to_csv(paths["observations"],index=False)
    pd.DataFrame(hypotheses).to_csv(paths["hypotheses"],index=False)
    updates_df.to_csv(paths["leader_updates"],index=False)
    policy_timeline.to_csv(paths["policy_timeline"],index=False)
    paths["policy_results"].write_text(json.dumps({
        "stage":"CURRENT_STACK_MINIMUM_CONFIDENT_V2",
        "policy_family_selected":False,
        "policy_results":policy_results,
        "blind_contract":{
            "gps_used":False,"srt_used":False,"reference_used":False,
            "oracle_used":False,"evaluation_error_used":False,
            "absolute_locked_emitted":False,
        },
    },indent=2))

    freeze_path=out/"r5_1_blind_implementation_freeze_manifest.json"
    freeze={
        "stage":"CURRENT_STACK_PROJECTED_CANDIDATE_MINCONF_FREEZE",
        "architecture_contract_sha256":actual_sha,
        "candidate_interface":"projected_absolute_candidates_v1",
        "verifier":"SIFT+LightGlue",
        "configuration":{
            "top_m":top_m,"evidence_frames":evidence_frames,
            "map_spacing_m":float(family["map_spacing_m"]),
            "maturity_support_required":support_required,
            "policies":policies,**cfg,
        },
        "inputs":{
            "candidate_csv":str(args.candidate_csv.resolve()),
            "candidate_sha256":sha256(args.candidate_csv.resolve()),
            "relative_csv":str(args.relative_csv.resolve()),
            "relative_sha256":sha256(args.relative_csv.resolve()),
            "manifest_csv":str(args.manifest_csv.resolve()),
            "manifest_sha256":sha256(args.manifest_csv.resolve()),
        },
        "counts":{
            "queries":len(common),"top4_candidate_rows":len(observations),
            "valid_projected_observations":int(observations["valid_projected_observation"].sum()),
            "leader_updates":int(updates_df["blind_leader_hypothesis_id"].notna().sum()),
            "geometric_pass_hypotheses":len(hypotheses),
        },
        "blind_contract":{
            "gt_used":False,"gps_used":False,"srt_used":False,
            "reference_used":False,"oracle_used":False,
            "evaluation_error_used":False,"policy_selected":False,
            "absolute_locked_emitted":False,
        },
        "outputs":{},
    }
    for name,path in paths.items():
        freeze["outputs"][name]={"path":str(path),"sha256":sha256(path)}
    freeze_path.write_text(json.dumps(freeze,indent=2))

    report=out/"r5_1_minimum_confident_bootstrap_v2_report.json"
    report.write_text(json.dumps({
        "stage":"CURRENT_STACK_MINIMUM_CONFIDENT_V2",
        "status":"PASS_CURRENT_STACK_PROJECTED_CANDIDATE_MINIMUM_CONFIDENT_V2",
        "architecture_contract_sha256":actual_sha,
        "blind_freeze_manifest_sha256":sha256(freeze_path),
        "policy_family_selected":False,
        "policy_results":policy_results,
        "blind_contract":freeze["blind_contract"],
    },indent=2))

    print("STATUS: PASS_CURRENT_STACK_PROJECTED_CANDIDATE_MINIMUM_CONFIDENT_V2")
    print("queries:",len(common))
    print("Top4 observation rows:",len(observations))
    print("valid projected observations:",int(observations["valid_projected_observation"].sum()))
    print("first leader query:",seed_q)
    print("GT/reference/SRT/GPS used: False")


if __name__=="__main__":
    main()
