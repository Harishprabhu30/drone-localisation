# CV1.1 — fixed scale / rotation reference-free evidence

Status: **IMPLEMENTATION PREPARED / LOCAL EXECUTION REQUIRED**

Parent protocol:
`docs/handoffs/crossview_scale_rotation_domain_v1/CV1_0_PROTOCOL.md`

This stage does not use SRT or reference truth.

## Purpose

Run the frozen ORB geometry on the frozen uniform40 x Top20 screen under:
- baseline;
- four fixed scale perturbations around the ORB 1.2 pyramid;
- three exact quarter-turn rotations.

The generated evidence remains reference-free.  It is frozen before any
containment/error labels are attached.

## Files

- `src/uavloc/geometry/crossview_query_transforms.py`
- `scripts/villoc/geometry/cv1_1_freeze_scale_rotation_evidence.py`
- `tests/test_cv1_1_crossview_query_transforms.py`
- `configs/research/crossview_scale_rotation_domain_cv1.yaml`

## Scientific safeguards

- baseline uses the unchanged frozen `orb_subtile.compute_features` path;
- transformed queries start from the exact frozen processed grayscale query;
- frozen map tiles and QV1.4 Top20 identities/ranks are unchanged;
- no SRT is loaded;
- no reference coordinates/oracle labels are loaded;
- no best scale/rotation selection occurs;
- no candidate winner/reranking is executed;
- baseline evidence must match frozen GD1.0 uniform40 evidence before a freeze can pass.

## Local execution

First run the focused tests:

```bash
python -m unittest tests.test_cv1_1_crossview_query_transforms -v
```

Then freeze the reference-free CV1.1 evidence:

```bash
python scripts/villoc/geometry/cv1_1_freeze_scale_rotation_evidence.py \
  --config configs/research/crossview_scale_rotation_domain_cv1.yaml
```

Expected pair count:

`40 queries x 20 candidates x 8 unique conditions = 6400 rows`

Do not evaluate accuracy until the freeze succeeds and the evidence SHA256 is recorded.
