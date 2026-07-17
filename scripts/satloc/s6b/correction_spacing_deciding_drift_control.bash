#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

python - <<'PY'
import json
from pathlib import Path

path = Path(
    "outputs/satloc/reports/s6b_relative_absolute/"
    "s6b0_absolute_correction_manifest_summary.json"
)

data = json.loads(path.read_text())

print("Relative variant:", data["relative_variant"])
print("Trajectory:", data["trajectory"])

print("\nCorrection policies")
print("-------------------")
for policy, result in data["policy_spacing_and_accuracy"].items():
    print(f"\n{policy}")
    for key, value in result.items():
        print(f"  {key}: {value}")
PY
