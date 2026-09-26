import numpy as np


def airsim_bgr_to_model_rgb(image: np.ndarray) -> np.ndarray:
    
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("Raw AirSim Scene input must be a uint8 numpy array.")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an HWC BGR Scene image, got {image.shape}.")
    
    return image[..., ::-1].copy(order="C")


