import numpy as np
import cv2


def lucas_kanade_flow(img1, img2):
    """
    Compute optical flow between two grayscale images using Lucas-Kanade.
    Returns flow in polar coordinates: (magnitude, angle).
    """
    img1_u8 = (img1 * 255).astype(np.uint8)
    img2_u8 = (img2 * 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(
        img1_u8, img2_u8, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return magnitude, angle


def compute_velocity_channels(X):
    """
    For a sample X of shape (128, 128, 4), compute 3 velocity channels
    using matrix differences between consecutive frames.
    Returns extended array of shape (128, 128, 7).
    """
    v1 = X[:, :, 1] - X[:, :, 0]
    v2 = X[:, :, 2] - X[:, :, 1]
    v3 = X[:, :, 3] - X[:, :, 2]
    return np.concatenate([X, v1[:, :, np.newaxis], v2[:, :, np.newaxis], v3[:, :, np.newaxis]], axis=-1)


def compute_optical_flow_channels(X):
    """
    For a sample X of shape (128, 128, 4), compute optical flow channels.
    Returns extended array of shape (128, 128, 7).
    """
    flows = []
    for i in range(3):
        mag, _ = lucas_kanade_flow(X[:, :, i], X[:, :, i + 1])
        flows.append(mag[:, :, np.newaxis])
    return np.concatenate([X] + flows, axis=-1)
