# Simplified factor story interpretation — 512_s256

## Main conclusion

The simplified factor analysis shows that DINO Top-1 score is the clearest simple confidence signal. Yaw and altitude have weak relationships with retrieval error in this dataset, and roll is not useful because it is constant. Sharpness and edge density do not mean the image is easier; they can increase in complex or repetitive scenes, which may still be visually ambiguous.

## Confidence calibration

The lowest score bin has 0.0% success ≤40 m with median error 540.8 m.
The highest score bin has 64.2% success ≤40 m with median error 36.5 m.

## Factor verdict

- `abs_top1_score`: strong relationship (Spearman -0.612); Useful confidence signal.
- `yaw_deg`: weak relationship (Spearman 0.023); Weak telemetry effect.
- `relative_altitude_m`: weak relationship (Spearman -0.067); Weak telemetry effect.
- `quality_sharpness_laplacian`: moderate relationship (Spearman 0.457); Scene complexity / ambiguity, not blur.
- `quality_brightness_mean`: moderate relationship (Spearman -0.417); Moderate appearance effect.
- `quality_contrast_std`: weak relationship (Spearman 0.104); Diagnostic only.
- `quality_edge_density`: moderate relationship (Spearman 0.445); Scene complexity / ambiguity, not blur.
- `distance_m`: weak relationship (Spearman 0.063); Weak route-position trend.
- `roll_deg`: not applicable or unavailable.

## Report wording

This supports using descriptor confidence as one input to absolute-correction gating. However, confidence alone is not enough: high-confidence wrong retrievals still occur, and some failures are caused by scene ambiguity or map-to-UAV appearance differences. Therefore, the final system combines retrieval confidence with verifier evidence and temporal consistency rather than relying on a single factor.
