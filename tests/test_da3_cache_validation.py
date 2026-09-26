import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_da3_teacher_cache",
    PROJECT_ROOT / "tools" / "validate_da3_teacher_cache.py",
)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class DA3CacheValidationTests(unittest.TestCase):
    def test_resolves_merged_data_all_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory_root = root / "Map" / "trajectory"
            trajectory_root.mkdir(parents=True)
            expected = trajectory_root / "merged_data_all.json"
            expected.write_text("{}", encoding="utf-8")

            self.assertEqual(
                VALIDATOR.resolve_merged_data_path(root, "Map", "trajectory"),
                expected,
            )

    def test_valid_da3_large_frame_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame_dir = Path(tmp) / "frames" / "000007"
            frame_dir.mkdir(parents=True)
            np.save(frame_dir / "da3_sliced.npy", np.ones((2, 64, 1024), dtype=np.float16))
            (frame_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "raw_frame_id": 7,
                        "cameras": ["frontcamera", "downcamera"],
                        "expected_feature_dim": 1024,
                    }
                ),
                encoding="utf-8",
            )
            error = VALIDATOR.validate_cache_file(
                frame_dir / "da3_sliced.npy",
                raw_frame_id=7,
                expected_views=2,
                expected_dim=1024,
                expected_token_count=64,
                require_meta=True,
                full_finite_check=True,
            )
            self.assertIsNone(error)

    def test_rejects_wrong_teacher_dimension(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame_dir = Path(tmp)
            np.save(frame_dir / "da3_sliced.npy", np.ones((2, 64, 768), dtype=np.float16))
            error = VALIDATOR.validate_cache_file(
                frame_dir / "da3_sliced.npy",
                raw_frame_id=0,
                expected_views=2,
                expected_dim=1024,
                expected_token_count=64,
                require_meta=False,
                full_finite_check=False,
            )
            self.assertIn("feature dim mismatch", error)

    def test_rejects_unpooled_token_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame_dir = Path(tmp)
            np.save(frame_dir / "da3_sliced.npy", np.ones((2, 324, 1024), dtype=np.float16))
            error = VALIDATOR.validate_cache_file(
                frame_dir / "da3_sliced.npy",
                raw_frame_id=0,
                expected_views=2,
                expected_dim=1024,
                expected_token_count=64,
                require_meta=False,
                full_finite_check=False,
            )
            self.assertIn("token count mismatch", error)


if __name__ == "__main__":
    unittest.main()
