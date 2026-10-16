from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class CurrentLocalizationPipelineContractTests(unittest.TestCase):
    def test_current_config_contract(self):
        cfg_path = ROOT / "configs/localization/current_stack_reference_free.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())

        self.assertTrue(cfg["reference_free"]["enabled"])
        self.assertFalse(cfg["streams"]["V"]["reference_available"])
        self.assertEqual(cfg["map"]["primary_variant"], "768_s256")
        self.assertEqual(cfg["retrieval"]["top_k"], 20)
        self.assertIn("_img518_", cfg["retrieval"]["descriptor_tag"])
        self.assertEqual(
            cfg["absolute_localization"]["verifier"]["method"],
            "sift_lightglue",
        )
        self.assertEqual(cfg["absolute_localization"]["authority_candidate_count"], 4)
        self.assertEqual(cfg["bootstrap"]["backend"], "minimum_confident_v2")
        self.assertEqual(
            cfg["bootstrap"]["candidate_interface"],
            "projected_absolute_candidates_v1",
        )

        arch = ROOT / cfg["bootstrap"]["architecture_contract"]
        final = ROOT / cfg["bootstrap"]["final_policy_contract"]
        self.assertEqual(sha256(arch), cfg["bootstrap"]["architecture_contract_sha256"])
        self.assertEqual(sha256(final), cfg["bootstrap"]["final_policy_contract_sha256"])

    def test_runner_uses_current_frontend_not_historical_orb(self):
        text = (ROOT / "scripts/villoc/run_localization_pipeline.py").read_text()
        self.assertIn("current_sift_lightglue_topk_verifier.py", text)
        self.assertIn("stage10b2_bootstrap_backend.py", text)
        self.assertIn("package_current_localization_run.py", text)
        self.assertNotIn("s8_12e1_top20_verifier_reranker.py", text)

    def test_projected_bootstrap_does_not_recompute_orb(self):
        text = (
            ROOT
            / "scripts/villoc/blind_demo/bootstrap/minimum_confident_projected_candidates.py"
        ).read_text()
        self.assertIn("projected_absolute_candidates_v1", text)
        self.assertNotIn("BFMatcher", text)
        self.assertNotIn("create_detector()", text)
        self.assertNotIn("recompute_projection", text)

    def test_neutral_canonical_artifacts(self):
        text = (
            ROOT / "scripts/villoc/pipeline/package_current_localization_run.py"
        ).read_text()
        for name in (
            "metadata/localization_query_manifest.csv",
            "trajectories/estimated_trajectory.csv",
            "map_state/map_state.json",
            "provenance/output_freeze.json",
            "provenance/run_manifest.json",
        ):
            self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()
