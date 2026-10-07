from __future__ import annotations

import unittest

from scripts.villoc.retrieval.r1_0_map_pyramid_preflight import (
    axis_starts,
    crs_equivalent,
)


class R1MapPyramidPreflightTests(unittest.TestCase):
    def test_lks94_wkt_is_equivalent_to_epsg_3346(self):
        wkt = (
            'PROJCS["LKS94 / Lithuania TM",'
            'GEOGCS["unnamed",DATUM["unnamed",'
            'SPHEROID["GRS 1980",6378137,298.257222101]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],'
            'PROJECTION["Transverse_Mercator"],'
            'PARAMETER["latitude_of_origin",0],'
            'PARAMETER["central_meridian",24],'
            'PARAMETER["scale_factor",0.9998],'
            'PARAMETER["false_easting",500000],'
            'PARAMETER["false_northing",0],'
            'UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
            'AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
        )

        self.assertTrue(
            crs_equivalent(
                wkt,
                "EPSG:3346",
            )
        )

    def test_axis_starts_anchors_last_window(self):
        self.assertEqual(
            axis_starts(1000, 384, 256),
            [0, 256, 512, 616],
        )

    def test_axis_starts_exact_grid(self):
        self.assertEqual(
            axis_starts(1024, 512, 256),
            [0, 256, 512],
        )

    def test_stride_larger_than_tile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            axis_starts(1024, 256, 512)


if __name__ == "__main__":
    unittest.main()
