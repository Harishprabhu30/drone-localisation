# CV1.3 full403 reference-free scale_m2 freeze

Status: **PASS_CV1_3_REFERENCE_FREE_FULL403_SCALE_M2_FREEZE**

Promoted condition:
`scale_m2 = 0.6944444444444444`

## Frozen result

Local execution reported:

- status: `PASS_CV1_3_REFERENCE_FREE_FULL403_SCALE_M2_FREEZE`
- evidence SHA256: `94353f763725f44f278192bcedcfd139ab136ecebf4f1dcc6e07487380964810`
- observed runtime: 349.296 s
- reference loaded: false
- SRT loaded: false
- baseline regenerated: false
- candidate selected/reranked: false
- accuracy claim: false

Freeze report:
`outputs/research_runs/crossview_scale_rotation_domain_v1/cv1_3/cv1_3_freeze_report.json`

Evidence:
`outputs/research_runs/crossview_scale_rotation_domain_v1/cv1_3/cv1_3_full403_scale_m2_reference_free_evidence.csv`

## Interpretation

The promoted uniform40 scale treatment has now been measured over the complete
403-query development candidate pool without opening truth.  The existing
GD1.0 full403 ORB evidence remains the unchanged baseline control.

The next legal step is CV1.4 post-freeze comparison.  No additional scale factor
may be introduced or retuned before that comparison.
