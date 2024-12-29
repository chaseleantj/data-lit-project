from typing import List
import numpy as np
import easyocr
from utils.time_utils import time_it


class OCRModel:
    def __init__(self):
        self.reader = easyocr.Reader(['en', 'de'])

    @time_it
    def extract_text_from_image(self, image_path: str | np.ndarray) -> List[str]:
        """
        Extract text from an image.
        """
        result = self.reader.readtext(image_path, detail=0, threshold=0.5)
        return result

