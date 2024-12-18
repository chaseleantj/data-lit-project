import cv2
import numpy as np
from typing import List


def load_single_image(image_path: str) -> np.ndarray:
    """
    Takes a single image path and returns the image as a Numpy array.
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")
    return image_bgr


def load_images(image_paths: List[str]) -> List[np.ndarray]:
    """
    Takes a list of image paths and returns the images as a list of Numpy arrays.
    """
    return [load_single_image(image_path) for image_path in image_paths]
