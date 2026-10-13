# Transition handoff — learned frontend ablation -> cross-view scale/rotation/domain study

This file marks the end of `research/learned-geometric-frontend-ablation-v1`.

Next branch:
`research/crossview-scale-rotation-domain-v1`

Read next:
`docs/handoffs/crossview_scale_rotation_domain_v1/README.md`

Reason for transition:
- ORB remains clearly stronger than XFeat, SuperPoint+LightGlue and ALIKED+LightGlue for frozen Top20 candidate discrimination;
- ALIKED partially rescues SuperPoint, indicating representation/domain effects;
- generic sparse learned matching is closed for now;
- next work will test scale/orientation/domain hypotheses and aerial-specific/detector-free upper bounds.

Critical continuity rule:
The project has NOT changed from absolute UAV localization. The new branch is a diagnostic and model-selection sub-branch whose output must return to the same frozen Top20 selection/localization problem before any deployment architecture work is promoted.
