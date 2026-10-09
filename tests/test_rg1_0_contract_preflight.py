from __future__ import annotations

import unittest

import pandas as pd

from scripts.villoc.geometry.rg1_0_contract_preflight import (
    validate_pool_contract,
)


class RG10ContractPreflightTests(unittest.TestCase):
    def test_valid_pool_contract(self):
        manifest = pd.DataFrame(
            {
                "trajectory_id": ["t", "t"],
                "query_id": [1, 2],
                "image_path": ["a.jpg", "b.jpg"],
                "image_width": [3840, 3840],
                "image_height": [2160, 2160],
                "reference_available": [False, False],
            }
        )

        rows = []
        for qid in (1, 2):
            for rank in (1, 2):
                rows.append(
                    {
                        "query_id": qid,
                        "pool_rank": rank,
                        "tile_id": f"sat_{qid}_{rank}",
                        "source_view": (
                            "center_square"
                            if rank == 1
                            else "right_square"
                        ),
                        "source_rank": rank,
                        "selection_reason": "test",
                    }
                )

        pool = pd.DataFrame(
            rows
        )

        summary = validate_pool_contract(
            pool=pool,
            manifest=manifest,
            expected_queries=2,
            expected_budget=2,
            allowed_sources={
                "center_square",
                "right_square",
            },
            require_center_top1=True,
        )

        self.assertEqual(
            summary[
                "candidate_rows"
            ],
            4,
        )
        self.assertTrue(
            summary[
                "top1_center_preserved"
            ]
        )

    def test_center_top1_violation_fails(self):
        manifest = pd.DataFrame(
            {
                "trajectory_id": ["t"],
                "query_id": [1],
                "image_path": ["a.jpg"],
                "image_width": [3840],
                "image_height": [2160],
                "reference_available": [False],
            }
        )

        pool = pd.DataFrame(
            [
                {
                    "query_id": 1,
                    "pool_rank": 1,
                    "tile_id": "sat_1",
                    "source_view": "right_square",
                    "source_rank": 1,
                    "selection_reason": "bad",
                }
            ]
        )

        with self.assertRaises(
            RuntimeError
        ):
            validate_pool_contract(
                pool=pool,
                manifest=manifest,
                expected_queries=1,
                expected_budget=1,
                allowed_sources={
                    "center_square",
                    "right_square",
                },
                require_center_top1=True,
            )


if __name__ == "__main__":
    unittest.main()
