"""RG2.1-B post-freeze availability oracle tests, all synthetic."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.villoc.geometry.rg2_1b_postfreeze_availability_oracle import (
    attach_reference,
    budget_summary,
    evaluate,
    load_and_validate_frozen_sets,
    pair_block,
    verify_config,
    verify_historical_parity,
)


def tile_geometry() -> pd.DataFrame:
    rows = []
    for name, x in (("A", 0.0), ("B", 50.0), ("C", 100.0), ("D", 200.0)):
        rows.append({
            "tile_id": name,
            "left_easting": x - 10,
            "right_easting": x + 10,
            "bottom_northing": -10.0,
            "top_northing": 10.0,
            "center_easting": x,
            "center_northing": 0.0,
        })
    return pd.DataFrame(rows)


def reference() -> pd.DataFrame:
    return pd.DataFrame(
        {"gt_x": [0.0, 50.0, 100.0], "gt_y": [0.0, 0.0, 0.0]},
        index=pd.Index([1, 2, 3], name="query_id"),
    )


def local_rows() -> pd.DataFrame:
    rows = [
        # q1: anchor already correct.
        (1, "A", True, True, True),
        # q2: immediate B rescues A.
        (2, "A", True, True, True),
        (2, "B", False, True, True),
        # q3: only the wider full-overlap set reaches C.
        (3, "A", True, True, True),
        (3, "B", False, True, True),
        (3, "C", False, False, True),
    ]
    return pd.DataFrame([
        {
            "trajectory_id": "dev", "query_id": qid,
            "anchor_tile_id": "A", "tile_id": tile,
            "union_rank": i + 1, "is_anchor": anchor,
            "in_immediate": immediate, "in_full_overlap": overlap,
            "in_frozen_qv_top20": tile in {"A", "B"},
            "frozen_qv_rank": 1 if tile == "A" else 2 if tile == "B" else "",
        }
        for i, (qid, tile, anchor, immediate, overlap) in enumerate(rows)
    ])


def top20_rows() -> pd.DataFrame:
    # Synthetic variable-depth table is fine for evaluate(); strict Top20
    # contract is exercised separately through load_and_validate_frozen_sets().
    rows = []
    per_q = {1: ["A", "D"], 2: ["A", "B"], 3: ["A", "D"]}
    for qid, tiles in per_q.items():
        for rank, tile in enumerate(tiles, 1):
            rows.append({
                "trajectory_id": "dev", "query_id": qid,
                "pool_rank": rank, "tile_id": tile,
                "source_view": "center_square" if rank == 1 else "right_square",
                "source_rank": rank,
            })
    return pd.DataFrame(rows)


class RG21BTests(unittest.TestCase):
    def test_config_forbids_blind_and_same_budget_claim(self):
        cfg = {
            "stage": "RG2.1-B", "map_variant": "768_s256", "map_crs": "EPSG:3346",
            "sets": ["anchor", "immediate", "full_overlap", "frozen_top20"],
            "scientific_boundary": {
                "development_reference_allowed": True,
                "reference_after_freeze_only": True,
                "blind_stress_loaded": False,
                "blind_stress_reference_allowed": False,
                "candidate_sets_mutable": False,
                "qv_top20_reranking_allowed": False,
                "orb_enabled": False, "projection_enabled": False,
                "bootstrap_enabled": False, "state_enabled": False,
                "learned_model_enabled": False,
            },
            "interpretation": {"same_budget_claim_allowed": False},
        }
        verify_config(cfg)
        cfg["scientific_boundary"]["blind_stress_loaded"] = True
        with self.assertRaisesRegex(ValueError, "blind_stress_loaded"):
            verify_config(cfg)

    def test_attach_reference_uses_inclusive_bounds_and_center_distance(self):
        candidates = pd.DataFrame([
            {"query_id": 1, "tile_id": "A"},
            {"query_id": 2, "tile_id": "A"},
        ])
        out = attach_reference(candidates, tile=tile_geometry(), reference=reference())
        self.assertTrue(bool(out.loc[0, "contains_query"]))
        self.assertFalse(bool(out.loc[1, "contains_query"]))
        self.assertAlmostEqual(float(out.loc[1, "center_error_m"]), 50.0)
        self.assertFalse(bool(out.loc[1, "le40"]))
        self.assertTrue(bool(out.loc[1, "le80"]))

    def test_evaluate_nested_rescues(self):
        _, _, query, analysis = evaluate(
            local_rows(), top20_rows(), tile_geometry(), reference()
        )
        self.assertEqual(int(query["anchor_contain"].sum()), 1)
        self.assertEqual(int(query["immediate_contain"].sum()), 2)
        self.assertEqual(int(query["full_overlap_contain"].sum()), 3)
        self.assertEqual(int(query["frozen_top20_contain"].sum()), 2)
        paired = analysis["paired_outcomes"]["full_overlap_vs_frozen_top20"]["contain"]
        self.assertEqual(paired, {
            "both_hit": 2, "a_only": 1, "b_only": 0, "neither": 0
        })

    def test_full_overlap_budget_is_reported_not_normalized_to_top20(self):
        _, _, _, analysis = evaluate(
            local_rows(), top20_rows(), tile_geometry(), reference()
        )
        full = analysis["sets"]["full_overlap"]["candidate_budget"]
        top = analysis["sets"]["frozen_top20"]["candidate_budget"]
        self.assertEqual(full["histogram"], {"1": 1, "2": 1, "3": 1})
        self.assertEqual(top["histogram"], {"2": 3})

    def test_pair_block_directionality(self):
        a = pd.Series([True, True, False, False])
        b = pd.Series([True, False, True, False])
        self.assertEqual(pair_block(a, b), {
            "both_hit": 1, "a_only": 1, "b_only": 1, "neither": 1
        })

    def test_budget_summary(self):
        summary = budget_summary(pd.Series([1, 2, 2, 4]))
        self.assertEqual(summary["min"], 1)
        self.assertEqual(summary["max"], 4)
        self.assertEqual(summary["histogram"], {"1": 1, "2": 2, "4": 1})

    def test_historical_parity_pass_and_fail(self):
        _, _, query, _ = evaluate(
            local_rows(), top20_rows(), tile_geometry(), reference()
        )
        cfg = {
            "historical_parity": {
                "frozen_top20": {
                    "query_count": 3, "contain_hits": 2,
                    "le40_hits": 2, "le80_hits": 2,
                },
                "anchor": {
                    "contain_hits": 1, "le40_hits": 1, "le80_hits": 2,
                },
            }
        }
        verify_historical_parity(cfg, query)
        cfg["historical_parity"]["frozen_top20"]["contain_hits"] = 3
        with self.assertRaisesRegex(RuntimeError, "Top20 historical parity"):
            verify_historical_parity(cfg, query)

    def test_unknown_tile_rejected(self):
        candidates = pd.DataFrame([{"query_id": 1, "tile_id": "missing"}])
        with self.assertRaisesRegex(RuntimeError, "unknown tile"):
            attach_reference(candidates, tile=tile_geometry(), reference=reference())

    def test_missing_reference_rejected(self):
        candidates = pd.DataFrame([{"query_id": 4, "tile_id": "A"}])
        with self.assertRaisesRegex(RuntimeError, "Reference missing"):
            attach_reference(candidates, tile=tile_geometry(), reference=reference())

    def test_strict_frozen_set_validator_requires_20_and_anchor_top1(self):
        local = pd.DataFrame([{
            "trajectory_id": "dev", "query_id": 1, "anchor_tile_id": "A",
            "tile_id": "A", "union_rank": 1, "is_anchor": True,
            "in_immediate": True, "in_full_overlap": True,
            "in_frozen_qv_top20": True, "frozen_qv_rank": 1,
        }])
        summary = pd.DataFrame([{
            "trajectory_id": "dev", "query_id": 1, "anchor_tile_id": "A",
            "anchor_count": 1, "immediate_count": 1,
            "full_overlap_count": 1, "union_count": 1, "top20_count": 20,
        }])
        top = pd.DataFrame([
            {
                "trajectory_id": "dev", "query_id": 1, "pool_rank": rank,
                "tile_id": f"T{rank}" if rank > 1 else "A",
                "source_view": "center_square" if rank == 1 else "right_square",
                "source_rank": 1 if rank == 1 else rank,
            }
            for rank in range(1, 21)
        ])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lp, sp, tp = root/"local.csv", root/"summary.csv", root/"top.csv"
            local.to_csv(lp, index=False)
            summary.to_csv(sp, index=False)
            top.to_csv(tp, index=False)
            a, b, c = load_and_validate_frozen_sets(
                lp, sp, tp, expected_queries=1
            )
            self.assertEqual((len(a), len(b), len(c)), (1, 1, 20))
            top.loc[top["pool_rank"] == 1, "tile_id"] = "WRONG"
            top.to_csv(tp, index=False)
            with self.assertRaisesRegex(RuntimeError, "Anchor/Top20 Top1 mismatch"):
                load_and_validate_frozen_sets(lp, sp, tp, expected_queries=1)


if __name__ == "__main__":
    unittest.main()
