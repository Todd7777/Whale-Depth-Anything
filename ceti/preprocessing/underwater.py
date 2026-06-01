"""
Underwater image preprocessing for Depth Anything inference and training.

Addresses color cast, low contrast, and wavelength-dependent attenuation
common in ROV/AUV camera feeds. Methods based on established underwater
vision literature (Sea-thru, UIEB, CLAHE-based enhancement).
"""

from __future__ import annotations

import cv2
import numpy as np


def gray_world_white_balance(image: np.ndarray) -> np.ndarray:
    """Simple gray-world white balance for underwater color cast."""
    result = image.astype(np.float32)
    for c in range(3):
        channel_mean = result[:, :, c].mean()
        if channel_mean > 1e-6:
            result[:, :, c] *= 128.0 / channel_mean
    return np.clip(result, 0, 255).astype(np.uint8)


def clahe_enhance(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization on L channel in LAB space."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def red_channel_compensation(image: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    """
    Boost red channel attenuated by water absorption.
    alpha controls compensation strength (0=none, 1=full).
    """
    result = image.astype(np.float32)
    green_mean = result[:, :, 1].mean()
    red_mean = result[:, :, 0].mean()
    if red_mean > 1e-6 and green_mean > red_mean:
        scale = 1.0 + alpha * (green_mean / red_mean - 1.0)
        result[:, :, 0] = np.clip(result[:, :, 0] * scale, 0, 255)
    return result.astype(np.uint8)


def combined_underwater_preprocess(image: np.ndarray) -> np.ndarray:
    """Recommended pipeline: white balance → red compensation → CLAHE."""
    img = gray_world_white_balance(image)
    img = red_channel_compensation(img)
    img = clahe_enhance(img)
    return img


def preprocess_underwater(
    image: np.ndarray,
    method: str = "combined",
) -> np.ndarray:
    """
    Apply underwater preprocessing.

    Args:
        image: BGR uint8 image (OpenCV format).
        method: One of 'none', 'white_balance', 'clahe', 'combined'.

    Returns:
        Preprocessed BGR image.
    """
    if method == "none":
        return image
    if method == "white_balance":
        return gray_world_white_balance(image)
    if method == "clahe":
        return clahe_enhance(image)
    if method == "combined":
        return combined_underwater_preprocess(image)
    raise ValueError(f"Unknown underwater preprocess method: {method}")


class UnderwaterAugmentation:
    """Training-time augmentations simulating underwater conditions."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() > self.p:
            return image

        img = image.astype(np.float32)

        # Blue-green color cast
        img[:, :, 0] *= np.random.uniform(0.6, 0.9)   # red attenuation
        img[:, :, 1] *= np.random.uniform(0.85, 1.0)  # green
        img[:, :, 2] *= np.random.uniform(0.9, 1.1)   # blue

        # Scattering haze
        haze = np.random.uniform(0.05, 0.25)
        img = img * (1 - haze) + 128 * haze

        # Reduced contrast
        contrast = np.random.uniform(0.7, 1.0)
        img = (img - 128) * contrast + 128

        return np.clip(img, 0, 255).astype(np.uint8)
