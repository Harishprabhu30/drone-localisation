#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

python - <<'PY'
import pandas as pd

path = (
    "outputs/satloc/metadata/s6b_relative_absolute/"
    "s6b1a_position_only_correction_events.csv"
)

df = pd.read_csv(path)

policies = [
    "oracle_position_reset_eval_only",
    "balanced_position_reset_online",
    "permissive_position_reset_online",
]

print("S6B.1A correction-event impact")
print("--------------------------------")

for policy in policies:
    group = df[df["policy"] == policy].copy()

    true_mask = group["correction_true_eval_only"].fillna(False).astype(bool)
    false_mask = group["correction_false_eval_only"].fillna(False).astype(bool)
    improved = group["error_improvement_m"] > 0
    worsened = group["error_improvement_m"] < 0

    print(f"\n{policy}")
    print(f"  Events:                  {len(group)}")
    print(f"  Events improving error:  {int(improved.sum())}")
    print(f"  Events worsening error:  {int(worsened.sum())}")
    print(
        "  Median improvement:      "
        f"{group['error_improvement_m'].median():.3f} m"
    )

    if true_mask.any():
        print(
            "  True-fix median impact:  "
            f"{group.loc[true_mask, 'error_improvement_m'].median():.3f} m"
        )

    if false_mask.any():
        false_group = group.loc[false_mask]

        print(
            "  False-fix median impact: "
            f"{false_group['error_improvement_m'].median():.3f} m"
        )
        print(
            "  False fixes improving:   "
            f"{int((false_group['error_improvement_m'] > 0).sum())}"
        )
        print(
            "  False fixes worsening:   "
            f"{int((false_group['error_improvement_m'] < 0).sum())}"
        )

print("\nWorst balanced correction impacts")
print("---------------------------------")

balanced = df[
    df["policy"] == "balanced_position_reset_online"
].copy()

columns = [
    "event_index",
    "correction_role",
    "sequence_frame_id",
    "token",
    "reference_cumulative_distance_m",
    "correction_true_eval_only",
    "pre_correction_error_m",
    "post_correction_error_m",
    "error_improvement_m",
    "position_shift_applied_m",
    "gap_since_previous_event_m",
]

print(
    balanced.sort_values("error_improvement_m")
    [columns]
    .head(12)
    .to_string(index=False)
)

print("\nBalanced false corrections")
print("--------------------------")

print(
    balanced[
        balanced["correction_false_eval_only"].fillna(False).astype(bool)
    ]
    .sort_values("error_improvement_m")
    [columns]
    .to_string(index=False)
)
PY
