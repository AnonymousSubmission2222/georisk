from typing import Dict, Sequence

import numpy as np


AIRSIM_COLLISION_EVENT = "airsim_has_collided"
DEPTH_COLLISION_EVENT = "depth_le1_gt10pct"
DUAL_COLLISION_REASON = "dual_collision_confirmed"
DEPTH_ENCODED_THRESHOLD = 1
DEPTH_FRACTION_THRESHOLD = 0.10


def evaluate_encoded_depth_collision(
    depth_images: Sequence[np.ndarray],
    camera_names: Sequence[str],
    encoded_threshold: int = DEPTH_ENCODED_THRESHOLD,
    fraction_threshold: float = DEPTH_FRACTION_THRESHOLD,
) -> Dict[str, object]:
    
    if len(depth_images) != len(camera_names):
        raise ValueError(
            f"Expected {len(camera_names)} depth views, got {len(depth_images)}."
        )

    view_fractions = {}
    triggered_views = []
    for camera_name, depth_image in zip(camera_names, depth_images):
        depth = np.asarray(depth_image)
        if depth.size == 0:
            raise ValueError(f"Empty encoded depth image for {camera_name}.")
        fraction = float(np.count_nonzero(depth <= encoded_threshold) / depth.size)
        view_fractions[str(camera_name)] = fraction
        if fraction > fraction_threshold:
            triggered_views.append(str(camera_name))

    return {
        "triggered": bool(triggered_views),
        "triggered_views": triggered_views,
        "view_fractions": view_fractions,
        "max_view_fraction": max(view_fractions.values()),
        "encoded_depth_threshold": int(encoded_threshold),
        "fraction_threshold": float(fraction_threshold),
        "comparison": "encoded_depth <= threshold and fraction > threshold",
    }
