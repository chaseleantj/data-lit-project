import cv2
import numpy as np


def calculate_image_features(image_bgr: np.ndarray) -> dict:
    """
    Calculates the average hue, saturation, and lightness (HSL) of an image.
    Additionally calculates the contrast and sharpness based on the grayscale version.
    """

    # Convert BGR to HLS
    image_hls = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HLS).astype(np.float32)

    # Split into H, L, S channels
    H = image_hls[:, :, 0]  # Hue channel (0-179 in OpenCV)
    L = image_hls[:, :, 1]  # Lightness channel (0-255)
    S = image_hls[:, :, 2]  # Saturation channel (0-255)

    # Normalize Hue to [0, 360) degrees
    H_degrees = (H * 2) % 360  # OpenCV Hue ranges from 0-179, scaled to 0-358

    # Normalize Saturation and Lightness to [0, 1]
    S_normalized = S / 255.0
    L_normalized = L / 255.0

    # Flatten the arrays for processing
    H_rad = np.deg2rad(H_degrees.flatten())
    S_flat = S_normalized.flatten()
    L_flat = L_normalized.flatten()

    # Compute mean of sine and cosine of Hue
    sin_sum = np.mean(np.sin(H_rad))
    cos_sum = np.mean(np.cos(H_rad))

    # Calculate average Hue
    avg_h_rad = np.arctan2(sin_sum, cos_sum)
    if avg_h_rad < 0:
        avg_h_rad += 2 * np.pi
    avg_hue = np.degrees(avg_h_rad)

    # Calculate average Saturation and Lightness
    avg_saturation = np.mean(S_flat)
    avg_lightness = np.mean(L_flat)

    # Calculate image RMS contrast based on grayscale
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rms_contrast = np.std(image_gray / 255.0)

    # Calculate image sharpness using the variance of the Laplacian
    laplacian_var = cv2.Laplacian(image_gray, cv2.CV_32F).var()

    output = {
        "hue": round(avg_hue, 2),
        "saturation": round(avg_saturation, 4),
        "lightness": round(avg_lightness, 4),
        "contrast": round(rms_contrast, 4),
        "sharpness": round(laplacian_var, 4)
    }

    return output
