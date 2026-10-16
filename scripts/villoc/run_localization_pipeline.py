#!/usr/bin/env python3
"""One-command current VILLOC localization pipeline.

This is the canonical successor to the historical recorded-flight demo
orchestrator. The historical script remains untouched for reproducibility.

Current stack:
  video -> reference-free manifest -> XFeat relative -> DINOv2 img518 Top20
  -> SIFT+LightGlue projected candidates -> minimum_confident_v2
  -> causal map-state timeline -> estimated trajectory -> visuals -> freeze
  -> neutral canonical artifact aliases/provenance.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def resolve(repo: Path, value) -> Path:
    p=Path(str(value)).expanduser()
    return p.resolve() if p.is_absolute() else (repo/p).resolve()


def run_stage(index, name, command, repo, run_root, env):
    log_dir=run_root/"logs"; log_dir.mkdir(parents=True,exist_ok=True)
    log_path=log_dir/f"pipeline_{index:02d}_{name}.log"
    print("\n"+"="*100)
    print(f"CURRENT-STACK STAGE {index:02d} — {name}")
    print("="*100)
    print(shlex.join([str(x) for x in command]))
    with log_path.open("w",encoding="utf-8") as log:
        process=subprocess.Popen(
            [str(x) for x in command],cwd=repo,env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("Could not capture stage output")
        for line in process.stdout:
            print(line,end="",flush=True); log.write(line); log.flush()
        code=process.wait()
    if code!=0:
        raise RuntimeError(f"Stage {index:02d} {name} failed; see {log_path}")


def main():
    p=argparse.ArgumentParser(description="Run the current reference-free VILLOC localization stack.")
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--video",type=Path,default=None)
    p.add_argument("--run-id",default=None)
    p.add_argument("--dino-device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--xfeat-device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--sift-device",choices=["cpu"],default="cpu",
                   help="CPU only for the currently frozen SIFT+LightGlue evidence contract.")
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()

    repo=Path.cwd().resolve()
    cfg_path=args.config.expanduser().resolve()
    require(cfg_path.exists(),f"Config not found: {cfg_path}")
    cfg=yaml.safe_load(cfg_path.read_text())
    require(cfg.get("reference_free",{}).get("enabled") is True,
            "Current pipeline requires reference_free.enabled=true")
    require(cfg["streams"]["V"].get("reference_available") is False,
            "Localization config must declare reference_available=false")
    require(cfg["map"]["primary_variant"]=="768_s256",
            "Current promoted stack requires 768_s256")
    require(cfg["retrieval"]["top_k"]==20,
            "Current promoted stack requires DINO Top20")
    require(cfg["absolute_localization"]["verifier"]["method"]=="sift_lightglue",
            "Current promoted stack requires SIFT+LightGlue")
    require(cfg["bootstrap"]["backend"]=="minimum_confident_v2",
            "Current promoted stack requires minimum_confident_v2")

    dataset=cfg["dataset"]; rf=cfg["reference_free"]; map_cfg=cfg["map"]
    retrieval=cfg["retrieval"]; export=cfg["export"]
    raw_root=resolve(repo,dataset["raw_root"])
    configured_root=resolve(repo,dataset["output_root"])
    run_root=(configured_root.parent/args.run_id).resolve() if args.run_id else configured_root
    video=(resolve(repo,args.video) if args.video is not None
           else (raw_root/str(cfg["streams"]["V"]["video"])).resolve())
    require(video.exists(),f"Video not found: {video}")

    variant=map_cfg["primary_variant"]; tile_cfg=map_cfg["tile_variants"][variant]
    tile_index=resolve(repo,tile_cfg["index_csv"])
    tile_dir=resolve(repo,tile_cfg["tiles_dir"])
    map_cache=resolve(repo,retrieval["map_descriptor_cache"])
    map_aoi=resolve(repo,map_cfg["canonical_aoi_tif"])
    for label,path in [
        ("tile index",tile_index),("tile directory",tile_dir),
        ("map cache",map_cache),("map AOI",map_aoi),
    ]:
        require(path.exists(),f"{label} not found: {path}")

    tag=str(retrieval["descriptor_tag"])
    require("_img518_" in tag and "center_square" in tag and "avgpatch" in tag,
            "Current stack requires DINO img518 center_square avgpatch")
    dino_query_tag=tag[:-4]+f"_{args.dino_device}" if tag.endswith("_cpu") else tag
    map_cache_root=map_cache.parent
    map_root=map_cache.parent.parent
    xfeat_repo=(repo/"third_party/accelerated_features").resolve()
    require(xfeat_repo.exists(),f"XFeat repository not found: {xfeat_repo}")

    runtime_cfg=copy.deepcopy(cfg)
    runtime_cfg["dataset"]["output_root"]=str(run_root)
    runtime_cfg["runtime_execution"]={
        "dino_query_device":args.dino_device,
        "dino_query_descriptor_tag":dino_query_tag,
        "map_descriptor_tag":tag,
        "xfeat_device":args.xfeat_device,
        "sift_lightglue_device":args.sift_device,
    }
    if args.video is not None:
        runtime_cfg["dataset"]["raw_root"]=str(video.parent)
        runtime_cfg["streams"]["V"]["video"]=video.name

    runtime_cfg_path=run_root/"config_resolved.yaml"
    manifest=run_root/"metadata/blind_query_manifest.csv"
    query_cache=run_root/"descriptors"/f"s8_11c_dinov2_queries_v_1fps_{dino_query_tag}.npz"
    relative=run_root/"metadata/s8_xfeat_relative_frontend/s8r4_xfeat_relative_trajectory_blind_raw.csv"
    topk=run_root/"retrieval/s8_11d"/f"s8_11d_topk_{variant}_{tag}.csv"
    candidate_evidence=run_root/"absolute/geometric_candidate_evidence.csv"
    absolute_summary=run_root/"absolute/absolute_query_summary.csv"
    bootstrap_report=run_root/"reports/blind_map_bootstrap/blind_map_bootstrap_report.json"
    map_traj=run_root/"trajectories/blind_map_aligned_relative_trajectory.csv"
    temporal_manifest=run_root/"metadata/blind_temporal_fusion/blind_temporal_correction_manifest.csv"
    fused=run_root/"trajectories/blind_temporal_fused_trajectory.csv"
    temporal_report=run_root/"reports/blind_temporal_fusion/blind_temporal_fusion_report.json"
    legacy_output=run_root/"trajectories/submission_estimated_trajectory.csv"
    addon9_report=run_root/"reports/addon9_estimated_latlon/estimated_latlon_export_report.json"

    py=sys.executable
    commands=[
        ("environment_preflight",[
            py,repo/"scripts/villoc/blind_demo/addon12_environment_checker.py",
            "--repo-root",repo,"--run-root",run_root,"--config",runtime_cfg_path,
            "--video",video,"--map-root",map_root,"--variant",variant,"--tag",tag,
            "--tile-index",tile_index,"--map-cache",map_cache,
            "--xfeat-path",xfeat_repo,"--require-lightglue",
        ]),
        ("reference_free_query_manifest",[
            py,repo/"scripts/villoc/blind_demo/addon7_blind_query_manifest.py",
            "--video",video,"--run-root",run_root,
            "--sample-rate-fps",str(rf["sample_rate_fps"]),
            "--assumed-rel-alt-m",str(rf["assumed_rel_alt_m"]),
            "--assumed-gimbal-pitch-deg",str(rf["assumed_gimbal_pitch_deg"]),
            "--view-assumption",str(rf.get("view_assumption","near_nadir")),
            "--map-aoi",map_aoi,"--map-tile-index",tile_index,
        ]),
        ("xfeat_relative_frontend",[
            py,repo/"scripts/villoc/s8_r4_xfeat_relative_frontend.py",
            "--repo-root",repo,"--manifest",manifest,"--output-root",run_root,
            "--xfeat-repo",xfeat_repo,"--sequence",str(dataset.get("sequence_name","recorded_flight")),
            "--device",args.xfeat_device,"--blind-only",
        ]),
        ("dino_query_cache",[
            py,repo/"scripts/villoc/s8_11bc_build_dinov2_caches.py",
            "--config",runtime_cfg_path,"--query-csv",manifest,
            "--reuse-map-caches","--map-cache-root",map_cache_root,
            "--map-cache-tag",tag,"--device",args.dino_device,
            "--batch-size","1","--image-size","518",
            "--crop-mode","center_square","--pooling","avgpatch",
        ]),
        ("dino_top20_retrieval",[
            py,repo/"scripts/villoc/blind_demo/stage6_blind_dino_topk_retrieval.py",
            "--query-cache",query_cache,"--map-cache",map_cache,"--run-root",run_root,
            "--variant",variant,"--tag",tag,"--top-k","20",
        ]),
        ("sift_lightglue_top20",[
            py,repo/"scripts/villoc/geometry/current_sift_lightglue_topk_verifier.py",
            "--config",runtime_cfg_path,"--query-csv",manifest,"--topk-csv",topk,
            "--tile-index-csv",tile_index,"--run-root",run_root,
            "--top-n","20","--device",args.sift_device,
        ]),
        ("minimum_confident_v2",[
            py,repo/"scripts/villoc/blind_demo/stage10b2_bootstrap_backend.py",
            "--repo-root",repo,"--config",runtime_cfg_path,"--run-root",run_root,
            "--candidate-csv",candidate_evidence,"--relative-csv",relative,
            "--manifest-csv",manifest,
        ]),
        ("causal_map_alignment",[
            py,repo/"scripts/villoc/blind_demo/stage10b3_map_alignment_router.py",
            "--repo-root",repo,"--config",runtime_cfg_path,"--blind-manifest",manifest,
            "--raw-relative",relative,"--bootstrap-report",bootstrap_report,
            "--run-root",run_root,"--map-crs",str(map_cfg["crs"]),
        ]),
        ("temporal_authority",[
            py,repo/"scripts/villoc/blind_demo/stage10b4_temporal_router.py",
            "--repo-root",repo,"--config",runtime_cfg_path,"--map-trajectory",map_traj,
            "--bootstrap-report",bootstrap_report,"--absolute-query-summary",absolute_summary,
            "--absolute-candidate-scores",candidate_evidence,"--run-root",run_root,
        ]),
        ("estimated_output_export",[
            py,repo/"scripts/villoc/blind_demo/addon9_estimated_output_router.py",
            "--repo-root",repo,"--config",runtime_cfg_path,"--fused-trajectory",fused,
            "--fusion-report",temporal_report,"--absolute-query-summary",absolute_summary,
            "--run-root",run_root,"--tile-index",tile_index,
            "--source-crs",str(map_cfg["crs"]),"--target-crs",str(export["target_crs"]),
        ]),
        ("reference_free_visuals",[
            py,repo/"scripts/villoc/blind_demo/addon10_no_reference_visuals.py",
            "--relative-trajectory",map_traj,"--submission",legacy_output,
            "--temporal-manifest",temporal_manifest,"--run-root",run_root,
            "--tile-index",tile_index,"--ortho-tif",map_aoi,
        ]),
        ("freeze_reference_free_output",[
            py,repo/"scripts/villoc/blind_demo/stage10b5d_freeze_blind_submission.py",
            "--submission",legacy_output,"--addon9-report",addon9_report,"--run-root",run_root,
        ]),
        ("canonical_project_artifacts",[
            py,repo/"scripts/villoc/pipeline/package_current_localization_run.py",
            "--run-root",run_root,"--config",runtime_cfg_path,
        ]),
    ]

    print("="*100)
    print("CURRENT VILLOC LOCALIZATION PIPELINE")
    print("="*100)
    print("video:",video)
    print("run root:",run_root)
    print("map variant:",variant)
    print("absolute verifier: SIFT+LightGlue")
    print("reference used by localization: False")
    print("\nExecution order:")
    for i,(name,cmd) in enumerate(commands,1):
        print(f"{i:02d}. {name:34s} {Path(str(cmd[1])).relative_to(repo)}")
    if args.dry_run:
        print("\nSTATUS: PASS_CURRENT_STACK_PIPELINE_DRY_RUN")
        return

    require(not run_root.exists(),f"Refusing to execute into existing run root: {run_root}")
    run_root.mkdir(parents=True,exist_ok=False)
    runtime_cfg_path.write_text(yaml.safe_dump(runtime_cfg,sort_keys=False))

    env=os.environ.copy()
    env["PYTHONPATH"]=str(repo/"src")+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"]="1"

    started=time.perf_counter(); completed=[]
    try:
        for i,(name,cmd) in enumerate(commands,1):
            run_stage(i,name,cmd,repo,run_root,env); completed.append(name)
        status="PASS"
    except BaseException:
        status="FAIL"
        raise
    finally:
        metrics=run_root/"metrics"; metrics.mkdir(parents=True,exist_ok=True)
        (metrics/"pipeline_execution_summary.json").write_text(json.dumps({
            "stage":"CURRENT_VILLOC_LOCALIZATION_PIPELINE",
            "status":f"{status}_CURRENT_STACK_PIPELINE_EXECUTION",
            "planned_stage_count":len(commands),
            "completed_stage_count":len(completed),
            "completed_stages":completed,
            "wall_s":float(time.perf_counter()-started),
            "run_root":str(run_root),
            "resolved_config":str(runtime_cfg_path),
            "reference_used":False,
        },indent=2))

    print("\nSTATUS: PASS_CURRENT_STACK_PIPELINE_EXECUTION")
    print("canonical trajectory:",run_root/"trajectories/estimated_trajectory.csv")
    print("run manifest:",run_root/"provenance/run_manifest.json")


if __name__=="__main__":
    main()
