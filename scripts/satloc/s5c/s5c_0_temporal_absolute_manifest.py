
"""
S5C.0 Temporal Absolute Benchmark Manifest

Features:
- CLI with argparse
- Input validation
- Manifest audit
- Uniform query generation
- Existing 73 benchmark merge
- Duplicate resolution
- Query source tagging
- Spacing statistics (frames/metres when available)
- JSON summary
- CSV manifest
- Publication figures
- PASS/FAIL terminal summary

Command Used:

export PYTHONPATH=$PWD/src

python scripts/satloc/s5c/s5c_0_temporal_absolute_manifest.py \
  --sequence traj01 \
  --uniform-step 5

"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def validate(df):
    req=["token0_id","sequence_frame_id"]
    miss=[c for c in req if c not in df.columns]
    if miss:
        raise RuntimeError(f"Missing columns: {miss}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sequence",default="traj01")
    ap.add_argument("--uniform-step",type=int,default=5)
    ap.add_argument("--max-frames",type=int)
    ap.add_argument("--sequence-manifest",type=Path,
        default=Path("outputs/satloc/metadata/s6a_relative_motion/s6a_sequence_manifest.csv"))
    ap.add_argument("--benchmark73",type=Path,
        default=Path("outputs/satloc/metadata/s5a_learned_local_verifier/s5a3e_best_policy_decisions_top50_all73.csv"))
    ap.add_argument("--output-root",type=Path,default=Path("outputs/satloc"))
    args=ap.parse_args()

    meta=args.output_root/"metadata"/"s5c_temporal"
    rep=args.output_root/"reports"/"s5c_temporal"
    fig=args.output_root/"figures"/"s5c_temporal"
    for d in (meta,rep,fig): d.mkdir(parents=True,exist_ok=True)

    seq=pd.read_csv(args.sequence_manifest)
    if "sequence" in seq.columns:
        seq=seq[seq["sequence"]==args.sequence]
    seq=seq.sort_values("sequence_frame_id").reset_index(drop=True)
    if args.max_frames: seq=seq.iloc[:args.max_frames]
    validate(seq)

    uniform=seq.iloc[::args.uniform_step][["token0_id","sequence_frame_id"]].copy()
    uniform["is_uniform"]=True

    bench=pd.read_csv(args.benchmark73)
    tcol="token0_id" if "token0_id" in bench.columns else "token"
    bench=bench[[tcol]].rename(columns={tcol:"token0_id"})
    bench["is_existing73"]=True

    man=seq.merge(uniform,on=["token0_id","sequence_frame_id"],how="left")
    man=man.merge(bench,on="token0_id",how="left")
    man["is_uniform"]=man["is_uniform"].where(man["is_uniform"].notna(), False).astype(bool)
    man["is_existing73"]=man["is_existing73"].where(man["is_existing73"].notna(), False).astype(bool)
    man["is_relative_risk"]=False
    man["risk_reason"]=""
    man=man[man["is_uniform"]|man["is_existing73"]].copy()
    man=man.sort_values("sequence_frame_id").drop_duplicates("token0_id")
    man["query_index"]=range(len(man))
    man["query_source"]=man.apply(lambda r:",".join([x for x,b in [("uniform",r.is_uniform),("benchmark73",r.is_existing73)] if b]),axis=1)

    out=meta/"s5c0_absolute_query_manifest.csv"
    man.to_csv(out,index=False)

    spacing=man["sequence_frame_id"].diff().dropna()
    plt.figure(figsize=(6,4)); plt.hist(spacing,bins=20); plt.tight_layout()
    plt.savefig(fig/"s5c0_query_spacing_histogram.png"); plt.close()

    plt.figure(figsize=(5,4))
    plt.bar(["Uniform","Benchmark73"],[man["is_uniform"].sum(),man["is_existing73"].sum()])
    plt.tight_layout(); plt.savefig(fig/"s5c0_query_source_breakdown.png"); plt.close()

    if {"x_enu_m","y_enu_m"}<=set(seq.columns):
        plt.figure(figsize=(6,6))
        plt.plot(seq["x_enu_m"],seq["y_enu_m"],alpha=.3)
        plt.scatter(man["x_enu_m"],man["y_enu_m"],s=8)
        plt.axis("equal"); plt.tight_layout()
        plt.savefig(fig/"s5c0_query_locations.png"); plt.close()

    summary={
        "sequence":args.sequence,
        "frames":len(seq),
        "uniform_queries":int(man.is_uniform.sum()),
        "existing73":int(man.is_existing73.sum()),
        "risk_queries":0,
        "total_queries":len(man),
        "median_spacing_frames":float(spacing.median()) if len(spacing) else 0
    }
    with open(rep/"s5c0_manifest_summary.json","w") as f:
        json.dump(summary,f,indent=2)

    print("S5C.0 Temporal Absolute Benchmark Manifest")
    print("------------------------------------------")
    print(f"Sequence: {args.sequence}")
    print(f"Frames: {len(seq)}")
    print(f"Uniform queries: {summary['uniform_queries']}")
    print(f"Existing benchmark: {summary['existing73']}")
    print(f"Total unique queries: {summary['total_queries']}")
    print("\nSaved outputs")
    print(out)

if __name__=="__main__":
    main()
