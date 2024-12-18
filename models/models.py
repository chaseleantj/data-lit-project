from pydantic import BaseModel
import numpy as np


class Thumbnail(BaseModel):
    url: str
    pixels: np.ndarray = None
    hue: int = None
    saturation: float = None
    lightness: float = None
    face_present: bool = False
    words_present: bool = False