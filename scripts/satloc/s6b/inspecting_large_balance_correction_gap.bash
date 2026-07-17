#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

python - <<'PY'
import pandas as pd

path = (
    "outputs/satloc/metadata/s6b_relative_absolute/"
    "s6b0_correction_spacing_by_policy.csv"
)

df = pd.read_csv(path)

balanced = (
    df[df["policy"] == "balanced_accept_online"]
    .sort_values("gap_m", ascending=False)
)

print(
    balanced[
        [
            "gap_type",
            "from_token",
            "to_token",
            "start_distance_m",
            "end_distance_m",
            "gap_m",
        ]
    ].head(15).to_string(index=False)
)
PY
