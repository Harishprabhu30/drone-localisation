python - <<'PY'
import pandas as pd
from pathlib import Path

p = Path("outputs/satloc/metadata/s5c_temporal/s5b1c_union_query_summary_s5c1_temporal_full263.csv")
df = pd.read_csv(p)

print("Columns:")
print(list(df.columns))

rank_col = None
for c in ["first_correct_rank", "first_rank_under_threshold", "first_correct_union_rank", "first_rank_under_40m"]:
    if c in df.columns:
        rank_col = c
        break

if rank_col is None:
    raise SystemExit("Could not find first-correct-rank column. Paste columns above.")

r = pd.to_numeric(df[rank_col], errors="coerce")

print("\nS5C.1 union temporal oracle recall")
print("----------------------------------")
print("Tokens:", len(df))
for k in [10, 20, 50]:
    hits = (r <= k).sum()
    print(f"Oracle@{k}: {hits}/{len(df)} = {hits/len(df):.3f}")

print("\nFirst correct rank stats")
print("------------------------")
print(r.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).to_string())

misses = df[r.isna()]
print("\nNo correct candidate within evaluated union pool:", len(misses))
PY