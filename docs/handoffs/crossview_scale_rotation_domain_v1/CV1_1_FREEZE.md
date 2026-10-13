# CV1.1 reference-free transform evidence freeze

Status: **PASS_CV1_1_REFERENCE_FREE_TRANSFORM_EVIDENCE_FREEZE**

Branch:
`research/crossview-scale-rotation-domain-v1`

Implementation commit:
`2284556a3311d0f44c3e58e9dd6bd541d98f754d`

## Frozen result

Local execution reported:

- status: `PASS_CV1_1_REFERENCE_FREE_TRANSFORM_EVIDENCE_FREEZE`
- baseline parity: `PASS_CV1_1_BASELINE_PARITY`
- evidence SHA256: `4abfacb77cf9d1d4b09ea917e048b0f2191659907dc4391f7d8fc3aadf020666`
- observed runtime: 272.930 s
- reference loaded: false
- SRT loaded: false
- candidate selected/reranked: false
- accuracy claim: false

Freeze report path:
`outputs/research_runs/crossview_scale_rotation_domain_v1/cv1_1/cv1_1_freeze_report.json`

Evidence path:
`outputs/research_runs/crossview_scale_rotation_domain_v1/cv1_1/cv1_1_uniform40_reference_free_evidence.csv`

## Interpretation

The baseline condition reproduced the frozen GD1.0 ORB evidence under the CV1.1
runner.  Therefore later differences between the fixed transform conditions and
baseline can be interpreted as transform effects rather than an accidental ORB
implementation change.

No geographic correctness conclusion is made at CV1.1.  Truth may be attached
only in CV1.2 after the evidence bytes are verified against the SHA256 above.
