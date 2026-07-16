python - <<'PY'
from pathlib import Path
import json
import pandas as pd

meta_src = Path("outputs/satloc/metadata/s5b_candidate_pool_improvement")
report_src = Path("outputs/satloc/reports/s5b_candidate_pool_improvement")
fig_src = Path("outputs/satloc/figures/s5b_candidate_pool_improvement")

meta_dst = Path("outputs/satloc/metadata/s5c_temporal")
report_dst = Path("outputs/satloc/reports/s5c_temporal")
fig_dst = Path("outputs/satloc/figures/s5c_temporal")

for d in [meta_dst, report_dst, fig_dst]:
    d.mkdir(parents=True, exist_ok=True)

chunks = [f"chunk{i:02d}" for i in range(6)]
run_prefix = "s5c2_temporal_union_top50"

def concat_chunks(kind):
    frames = []
    for ch in chunks:
        p = meta_src / f"s5b2_lightglue_union_{kind}_{run_prefix}_{ch}.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p)
        df["s5c2_chunk"] = ch
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

candidate_scores = concat_chunks("candidate_scores")
query_summary = concat_chunks("query_summary")
policy_chunks = concat_chunks("policy_summary")

candidate_out = meta_dst / "s5c2_lightglue_union_candidate_scores_top50_full263.csv"
query_out = meta_dst / "s5c2_lightglue_union_query_summary_top50_full263.csv"
policy_chunk_out = meta_dst / "s5c2_lightglue_union_policy_summary_chunks_top50_full263.csv"

candidate_scores.to_csv(candidate_out, index=False)
query_summary.to_csv(query_out, index=False)
policy_chunks.to_csv(policy_chunk_out, index=False)

# Aggregate final policy summary from merged query summary.
rows = []
for policy, g in query_summary.groupby("policy"):
    hits = g["hit_le_threshold"].fillna(False).astype(bool).sum()
    tokens = len(g)
    oracle_hits = g["oracle_processed_hit_le_threshold"].fillna(False).astype(bool).sum()

    rows.append({
        "policy": policy,
        "tokens": int(tokens),
        "hits": int(hits),
        "hit_rate": float(hits / tokens) if tokens else 0.0,
        "median_error_m": float(pd.to_numeric(g["chosen_error_m"], errors="coerce").median()),
        "oracle_processed_hits": int(oracle_hits),
        "oracle_processed_hit_rate": float(oracle_hits / tokens) if tokens else 0.0,
        "median_oracle_error_m": float(pd.to_numeric(g["oracle_processed_error_m"], errors="coerce").median()),
        "median_oracle_lg_rank": float(pd.to_numeric(g["oracle_lightglue_rank"], errors="coerce").median()),
    })

policy_summary = pd.DataFrame(rows).sort_values("hit_rate", ascending=False)
policy_out = meta_dst / "s5c2_lightglue_union_policy_summary_top50_full263.csv"
policy_summary.to_csv(policy_out, index=False)

report = {
    "stage": "S5C.2",
    "description": "Temporal LightGlue verification on S5C.1 union Top-50 candidate pools.",
    "tokens": int(query_summary["token"].nunique() if "token" in query_summary.columns else query_summary["token0_id"].nunique()),
    "candidate_rows": int(len(candidate_scores)),
    "chunks": chunks,
    "policy_summary": policy_summary.to_dict(orient="records"),
    "outputs": {
        "candidate_scores": str(candidate_out),
        "query_summary": str(query_out),
        "policy_summary": str(policy_out),
        "policy_chunk_summary": str(policy_chunk_out),
    },
    "locked_rule": "Reference/error columns were used only after ranking for evaluation."
}

report_out = report_dst / "s5c2_lightglue_union_top50_full263_summary.json"
report_out.write_text(json.dumps(report, indent=2))

print("S5C.2 merged outputs")
print("--------------------")
print(candidate_out)
print(query_out)
print(policy_out)
print(report_out)

print("\nFinal S5C.2 policy summary")
print("--------------------------")
print(policy_summary.to_string(index=False))
PY