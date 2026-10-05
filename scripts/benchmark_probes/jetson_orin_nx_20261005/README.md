# Probe archive — 2026-10-05

These are the available conversation-generated diagnostic scripts, archived
unchanged under `archive/`. Their exact SHA256s are in `manifest.json`. The
archive does not include the datasets, weights, original reports, or feature
NPZ inputs. Python syntax compilation is the available local verification;
the historical run evidence is in the handoff README.

| Scripts | Scope |
| --- | --- |
| `gpu_probe_four_frames.py`, `dino_cuda_full_query_probe.py` | Jetson-only fixed-path DINO FP32 CUDA controls; frozen CPU cache prerequisites |
| `orb_fixed_pair_probe.py`, `orb_stage_boundary_probe.py` | Two fixed recorded-flight query/tile pairs; repeatability/preprocessing hashes |
| `orb_feature_swap_probe.py` | Export/swap fixed ORB feature sets across machines |
| `orb_30_query_feature_control.py` | Export Mac query IDs 1–30, rerank with copied query features on Jetson |
| `orb_30_bootstrap_feature_control.py` | Controlled 30-query v2 recomputation with supplied Mac features |
| `fix_orb_30_export.py` | Historical patch utility; archived exporter is already corrected, so do not apply it again |
| `traj01_r5_1_postfreeze_probe.py` | Verify freeze, then reference evaluation; supports `--research-root` |
| `traj01_v2_failure_localization.py`, `traj01_q390_acceptance_audit.py` | Fixed Mac full403 research outputs; post-freeze drift diagnosis |
| `traj01_v2_leader_consistency_replay.py`, `traj01_v2_leader_consistency_replay_q134.py` | Endpoint guard counterfactuals; the two available files are identical |
| `traj01_v2_transform_continuity_replay.py` | Exploratory state-jump/rotation guard replay, not production code |

Run from the repository root, using the intended machine's environment. Dynamic
imports use frozen modules under `scripts/villoc`. Some scripts require recorded
absolute image paths in CSVs and cannot be relocated without an explicit path
mapping. Others assert the frozen source SHA256. Do not disable those checks to
force a run against changed code.

For historical replay, install copies into the original scratch-output location
only when the destination filenames are absent:

```bash
mkdir -p outputs/benchmark_probes
cp -n scripts/benchmark_probes/jetson_orin_nx_20261005/archive/*.py outputs/benchmark_probes/
cp -n scripts/benchmark_probes/jetson_orin_nx_20261005/archive/traj01_r5_1_postfreeze_probe.py outputs/benchmark_probes/traj01_r5_1_postfreeze_probe_full403.py
```

The second command supplies the historical helper filename imported by the
full403 diagnostics. When executing that helper directly, pass
`--research-root outputs/research_runs/minimum_confident_bootstrap/traj01_blind_r5_1_v2_full403_20261005_001`;
its default is the older 38-query equivalence root. That alias is an installation
convenience, not proof that all historical versions were identical.

The GPU probes create outputs at fixed names; inspect them before rerunning to
avoid replacing the prior probe results. They deliberately bypass the frozen
builder's CPU-only loader in isolated code; they do not add a CUDA CLI to it.

`environment_inventory.py` is new and read-only. It collects provenance as JSON
on stdout, imports installed runtimes if present, and can hash explicitly named
assets. It does not run inference, inspect GT, change power settings, or install
dependencies. Example:

```bash
python scripts/benchmark_probes/jetson_orin_nx_20261005/environment_inventory.py \
  --asset data/raw/villoc/traj01_90deg_stable120m/villoc_traj01_90deg_stable120m_V_merged.MP4
```

Redirect stdout to a fresh report file if desired; redirected commands naturally
show no report in the terminal until the file is opened.
