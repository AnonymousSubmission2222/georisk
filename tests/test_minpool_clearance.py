import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from georisk.llm.qwen_uav_model import GeoRiskForNavigation  


class MinPoolClearanceTests(unittest.TestCase):
    def test_local_min_returns_value_and_actual_argmin_coordinates(self):
        raw = torch.full((16, 16), 255.0)
        samples = ((3, 5, 0.0), (2, 12, 10.0), (14, 1, 20.0), (15, 15, 30.0))
        for v_coord, u_coord, value in samples:
            raw[v_coord, u_coord] = value

        depth, v_coords, u_coords = GeoRiskForNavigation._local_min_depth_samples(
            raw,
            pool_size=8,
        )

        self.assertTrue(torch.equal(depth, torch.tensor([[0.0, 10.0], [20.0, 30.0]])))
        self.assertTrue(torch.equal(v_coords, torch.tensor([[3, 2], [14, 15]])))
        self.assertTrue(torch.equal(u_coords, torch.tensor([[5, 12], [1, 15]])))

    def test_zero_depth_is_retained_during_unprojection(self):
        raw = torch.full((16, 16), 255, dtype=torch.uint8)
        raw[3, 5] = 0
        stub = SimpleNamespace(
            clearance_depth_pool_size=8,
            clearance_depth_max_m=100.0,
            clearance_local_range_m=30.0,
            _local_min_depth_samples=GeoRiskForNavigation._local_min_depth_samples,
        )

        points = GeoRiskForNavigation._unproject_depth_view_to_current(
            stub,
            raw_depth=raw,
            valid=torch.tensor(1.0),
            view_to_current_pose=torch.eye(4),
            camera_slot=1,
        )

        self.assertEqual(tuple(points.shape), (1, 3))
        self.assertTrue(torch.equal(points, torch.zeros((1, 3))))

    def test_invalid_view_produces_no_points(self):
        raw = torch.zeros((16, 16), dtype=torch.uint8)
        stub = SimpleNamespace(clearance_depth_pool_size=8)
        points = GeoRiskForNavigation._unproject_depth_view_to_current(
            stub,
            raw_depth=raw,
            valid=torch.tensor(0.0),
            view_to_current_pose=torch.eye(4),
            camera_slot=0,
        )
        self.assertEqual(tuple(points.shape), (0, 3))

    def test_clearance_loss_backpropagates_to_waypoints(self):
        model = GeoRiskForNavigation.__new__(GeoRiskForNavigation)
        torch.nn.Module.__init__(model)
        model.clearance_depth_pool_size = 8
        model.clearance_depth_max_m = 100.0
        model.clearance_local_range_m = 30.0
        model.clearance_voxel_size_m = 100.0 / 255.0
        model.clearance_max_voxels = 2048
        model.clearance_path_samples_per_segment = 4
        model.clearance_safe_margin_m = 2.0
        model.clearance_temperature = 0.25
        model.traj_execute_points = 5

        depth = torch.full((1, 1, 256, 256), 255, dtype=torch.uint8)
        depth[0, 0, 128, 128] = 3
        waypoints = torch.zeros((1, 10, 3), dtype=torch.float32, requires_grad=True)
        loss = model._compute_clearance_loss(
            predicted_trajectory=waypoints,
            clearance_depth_gt=depth,
            clearance_depth_valid_mask=torch.ones((1, 1)),
            clearance_geom_valid_mask=torch.ones((1,)),
            clearance_view_to_current_poses=torch.eye(4).view(1, 1, 4, 4),
            clearance_target_to_current_rot=torch.eye(3).view(1, 3, 3),
            traj_valid_mask=torch.ones((1, 10)),
            phase2_valid_mask=torch.ones((1,)),
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(waypoints.grad)
        self.assertTrue(torch.isfinite(waypoints.grad).all())
        self.assertGreater(float(waypoints.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
