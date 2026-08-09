# Minimum-Confident Bootstrap Research

Research-only work on safer absolute initialization for the Villoc blind localization pipeline.

## Stage flow

- R1: read-only interface audit.
- R2: approved multi-candidate architecture.
- R3: blind minimum-confident bootstrap.
- R4.0: freeze exact blind implementation, inputs, outputs, and lock.
- R4.1: post-freeze GT quality evaluation.
- R4.2: post-freeze failure localization.
- R4.3: transform-family persistence diagnostic (next).

## Blind boundary

`minimum_confident_bootstrap.py` and any pre-freeze stage must not read GPS/SRT, reference lat/lon or ENU, oracle identities, GT errors, RMSE, or post-freeze diagnostic outputs.

Files under `diagnostics/` are explicitly reference-allowed only after the R4.0 freeze. They are method-specific research diagnostics kept for reproducibility, not production utilities.

## Layout

```text
scripts/villoc/research/minimum_confident_bootstrap/
├── minimum_confident_bootstrap.py
├── diagnostics/
│   ├── r4_1_postfreeze_evaluation.py
│   └── r4_2_failure_localization.py
└── README.md
```

Generated evidence remains in `outputs/research_runs/minimum_confident_bootstrap/`.

## Current finding

The first R3 implementation delayed the known q19 false commitment to q38 and materially improved trajectory error, but the q38 transform remained incorrect. R4.2 showed that useful candidates already existed within Top-4 and that a much better q38 transform existed among the 32 geometrically admissible hypotheses. The current Pareto pruning discarded it, while the 0.5-tile transform-clustering tolerance merged visibly different transform families. Top-M expansion is therefore not currently justified; R4.3 will test transform-family persistence through time before changing the algorithm.
