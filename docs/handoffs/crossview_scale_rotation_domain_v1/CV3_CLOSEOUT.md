# CV3 closeout

Decision: **CLOSE_CV3_AERIAL_MAST3R_COMPACT_SCREEN**.

CV3.3 compared the exact same 40 queries and exact same six candidate identities
for SIFT+LightGlue and aerial-MASt3R. MASt3R was directionally better but only
by +1 containment, +1 center<=80, +2 selected projection<=40, and -1 unsafe.
This did not meet the predeclared promotion gate.

The full 688,638,856-parameter FP32 aerial-MASt3R model was already used.
No full Top20 expansion is promoted.

Promoted practical absolute geometric frontend: **SIFT + LightGlue**.
