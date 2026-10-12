# LF2.0-C closeout — SuperPoint + LightGlue rejected

Status: **PASS / CLOSED** on user-reported local execution.

Status:
`PASS_LF2_0C_POSTFREEZE_LIGHTGLUE_UNIFORM40_VS_ORB_AUDIT`

Decision:
`CLOSE_SUPERPOINT_LIGHTGLUE`

Audit diagnostics SHA256:
`00a9b94c171bb738199d0e65edb915fc46c54d2f066a4c13eb0c8c21a247b71c`

Primary Top1 on the deterministic 40-query screen:

- contains_query: ORB 28/40, LightGlue 18/40;
- center <=40 m: ORB 13/40, LightGlue 13/40;
- center <=80 m: ORB 28/40, LightGlue 18/40.

Selected projection:

- ORB median 5.416 m, <=40 29/40, <=80 29/40, >200 10/40;
- SuperPoint+LightGlue median 106.198 m, <=40 16/40, <=80 19/40, >200 19/40.

No LightGlue-native confidence score passed the secondary formulation gate. Therefore a full 403-query SuperPoint+LightGlue run is not justified.

Next question: isolate the extractor by keeping LightGlue fixed and swapping SuperPoint for ALIKED-N16. If ALIKED also fails, close sparse LightGlue and move to detector-free matching.
