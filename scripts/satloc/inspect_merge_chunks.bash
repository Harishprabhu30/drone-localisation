python - <<'PY'
import pandas as pd

q = pd.read_csv("outputs/satloc/metadata/s5c_temporal/s5c2_lightglue_union_query_summary_top50_full263.csv")
c = pd.read_csv("outputs/satloc/metadata/s5c_temporal/s5c2_lightglue_union_candidate_scores_top50_full263.csv")

print("QUERY SUMMARY COLUMNS")
print(list(q.columns))

print("\nCANDIDATE SCORE COLUMNS")
print(list(c.columns))

print("\nLightGlue-only sample")
print(q[q["policy"]=="lightglue_only"].head(5).to_string(index=False))
PY
