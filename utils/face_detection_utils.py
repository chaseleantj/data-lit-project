from typing import List, Dict, Any
from deepface import DeepFace

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
from utils.time_utils import time_it


# @time_it
def detect_faces(image_path: str | np.ndarray, detector_backend: str = "ssd") -> List[Dict[str, Any]]:
    """
    Detect faces in an image and return the bounding boxes of the detected faces.
    """
    resp = DeepFace.extract_faces(img_path=image_path, detector_backend=detector_backend, enforce_detection=False, align=False)
    return resp

def count_faces(faces: List[Dict[str, Any]]) -> int:
    count = int(np.sum([1 for face in faces if face['confidence'] > 0]))
    return count

def plot_detected_faces(image_path: str | np.ndarray, detector_backend: str = "ssd") -> None:
    """
    Detect faces in an image and plot the image with bounding boxes around detected faces.
    
    Args:
        image_path (str): Path to the image file
        detector_backend (str): Face detector backend to use. 
            Options: 'opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface'
            (default is 'opencv')
    """
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = detect_faces(image_path, detector_backend)
    
    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    # Plot each detected face
    for face in faces:
        facial_area = face['facial_area']
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        
        # Create a rectangle patch
        rect = patches.Rectangle(
            (x, y), w, h, 
            linewidth=2, 
            edgecolor='r', 
            facecolor='none'
        )
        
        # Add the rectangle to the plot
        ax.add_patch(rect)
    
    plt.axis('off')
    plt.show()

def plot_image_with_faces(img, faces, ax=None, show=False):
    """
    Plot a single image with detected faces and face count.
    
    Args:
        img: Image array in RGB format
        faces: List of detected faces
        ax: Matplotlib axis to plot on
        show: Whether to show the plot immediately
    """
    if ax is None:
        fig, ax = plt.subplots(1)
    
    ax.imshow(img)
    
    # Plot each detected face
    for face in faces:
        facial_area = face['facial_area']
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        rect = patches.Rectangle(
            (x, y), w, h, 
            linewidth=1, 
            edgecolor='r', 
            facecolor='none'
        )
        ax.add_patch(rect)
    
    # Add face count text below the image
    ax.text(0.5, -0.1, f'Faces: {count_faces(faces)}', 
            horizontalalignment='center',
            transform=ax.transAxes)
    
    ax.axis('off')
    
    if show:
        plt.show()
