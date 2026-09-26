import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vlnce_src.collision_events import evaluate_encoded_depth_collision


def _load_metric_module():
    spec = importlib.util.spec_from_file_location(
        "dual_collision_metric",
        PROJECT_ROOT / "utils" / "metric.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EncodedDepthCollisionTests(unittest.TestCase):
    def test_any_view_above_ten_percent_triggers(self):
        names = ["front", "left", "right", "rear", "down"]
        images = [np.full((10, 10), 255, dtype=np.uint8) for _ in names]
        images[2].flat[:11] = 1

        result = evaluate_encoded_depth_collision(images, names)

        self.assertTrue(result["triggered"])
        self.assertEqual(result["triggered_views"], ["right"])
        self.assertAlmostEqual(result["view_fractions"]["right"], 0.11)

    def test_exactly_ten_percent_does_not_trigger(self):
        names = ["front", "left", "right", "rear", "down"]
        images = [np.full((10, 10), 255, dtype=np.uint8) for _ in names]
        images[0].flat[:10] = 0

        result = evaluate_encoded_depth_collision(images, names)

        self.assertFalse(result["triggered"])

    def test_encoded_values_zero_and_one_are_counted(self):
        names = ["front", "left", "right", "rear", "down"]
        images = [np.full((10, 10), 255, dtype=np.uint8) for _ in names]
        images[-1].flat[:6] = 0
        images[-1].flat[6:12] = 1

        result = evaluate_encoded_depth_collision(images, names)

        self.assertTrue(result["triggered"])
        self.assertAlmostEqual(result["max_view_fraction"], 0.12)


class CollisionCutoffMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metric = _load_metric_module()

    def test_each_condition_uses_its_own_first_trigger(self):
        record = {
            "name": "sample",
            "frames": [0, 5, 10],
            "positions": [
                np.asarray([50.0, 0.0, 0.0]),
                np.asarray([30.0, 0.0, 0.0]),
                np.asarray([10.0, 0.0, 0.0]),
            ],
            "gt_positions": [
                np.asarray([50.0, 0.0, 0.0]),
                np.asarray([0.0, 0.0, 0.0]),
            ],
            "target_position": np.asarray([0.0, 0.0, 0.0]),
            "collision_events": {
                "airsim_has_collided": {"triggered": True, "first_frame_index": 5},
                "depth_le1_gt10pct": {"triggered": True, "first_frame_index": 10},
            },
            "original_success": True,
            "original_oracle": True,
        }

        airsim = self.metric._evaluate_record(record, "airsim_has_collided")
        depth = self.metric._evaluate_record(record, "depth_le1_gt10pct")

        self.assertEqual(airsim["cutoff_frame"], 5)
        self.assertFalse(airsim["success"])
        self.assertFalse(airsim["oracle_success"])
        self.assertAlmostEqual(airsim["ne_m"], 30.0)
        self.assertEqual(depth["cutoff_frame"], 10)
        self.assertFalse(depth["success"])
        self.assertTrue(depth["oracle_success"])
        self.assertAlmostEqual(depth["ne_m"], 10.0)


if __name__ == "__main__":
    unittest.main()


