# VPR Showcase — Source Audit and Token Selection

The claims are frozen, including repeated structures such as railways, riverbanks, roads, forests, ponds and agricultural fields.

No hero token is frozen yet. The shortlist combines:

- S4C.6 classical 73-token failure groups;
- S7 manually assigned scene labels;
- DINOv2-VLAD center/resize failures and rescues;
- agricultural-open-field failures;
- frames still missed by both DINOv2-VLAD streams.

## Install

```bash
mkdir -p scripts/presentation configs/presentation docs/presentation
cp audit_vpr_showcase_sources.py scripts/presentation/
cp presentation_vpr_showcase.yaml configs/presentation/
cp README_vpr_showcase_source_audit.md docs/presentation/
```

## Run the local audit

```bash
export PYTHONPATH=$PWD/src
python scripts/presentation/audit_vpr_showcase_sources.py \
  --repo-root "$PWD" \
  --output-dir outputs/presentation/source_audit
```

It writes:

```text
outputs/presentation/source_audit/
├── vpr_showcase_source_audit.json
├── vpr_showcase_source_audit.csv
└── vpr_hardest_frame_shortlist.csv
```

## Review the shortlist

```bash
python - <<'PY'
import pandas as pd
p = "outputs/presentation/source_audit/vpr_hardest_frame_shortlist.csv"
df = pd.read_csv(p)
print("\nHero candidates")
print(df[df.recommended_role == "hero_candidate"].head(20).to_string(index=False))
print("\nHardest failures")
print(df[df.recommended_role == "hardest_failure"].head(30).to_string(index=False))
PY
```

The repository code confirms the canonical paths and expected schemas, but the cached experiment outputs live in the local checkout. The local audit must pass before token IDs are written into the YAML.

## Diagram recommendation

Use SVG as the presentation master format.

- Data-driven diagrams: Python/Matplotlib patches or Graphviz.
- Hand-tuned showcase diagrams: Figma or diagrams.net.
- TikZ/Overleaf: reserve for mathematical or publication figures; it is slower for visual iteration and PowerPoint animation.
