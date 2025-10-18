"""
Image processing utilities for medical image analysis.
This module provides functions for preprocessing medical images.
"""

import numpy as np
from PIL import Image
import cv2


def load_image(image_path):
    """
    Load an image from the specified path.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Loaded image as numpy array
    """
    try:
        img = Image.open(image_path)
        return np.array(img)
    except Exception as e:
        raise Exception(f"Error loading image: {str(e)}")


def resize_image(image, target_size=(224, 224)):
    """
    Resize image to target dimensions.
    
    Args:
        image (numpy.ndarray): Input image
        target_size (tuple): Target dimensions (width, height)
        
    Returns:
        numpy.ndarray: Resized image
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def normalize_image(image):
    """
    Normalize image pixel values to [0, 1] range.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Normalized image
    """
    return image.astype(np.float32) / 255.0


def apply_clahe(image):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Useful for enhancing medical images.
    
    Args:
        image (numpy.ndarray): Input grayscale image
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    if len(image.shape) == 3:
        # Convert to grayscale if color image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
        
    return clahe.apply(gray)


def denoise_image(image, method='bilateral'):
    """
    Apply denoising to the image.
    
    Args:
        image (numpy.ndarray): Input image
        method (str): Denoising method ('bilateral', 'gaussian', 'median')
        
    Returns:
        numpy.ndarray: Denoised image
    """
    if method == 'bilateral':
        return cv2.bilateralFilter(image, 9, 75, 75)
    elif method == 'gaussian':
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == 'median':
        return cv2.medianBlur(image, 5)
    else:
        raise ValueError(f"Unknown denoising method: {method}")


def augment_image(image, rotation=0, flip_horizontal=False, flip_vertical=False):
    """
    Apply data augmentation transformations to the image.
    
    Args:
        image (numpy.ndarray): Input image
        rotation (int): Rotation angle in degrees
        flip_horizontal (bool): Whether to flip horizontally
        flip_vertical (bool): Whether to flip vertically
        
    Returns:
        numpy.ndarray: Augmented image
    """
    augmented = image.copy()
    
    # Rotation
    if rotation != 0:
        rows, cols = augmented.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
        augmented = cv2.warpAffine(augmented, M, (cols, rows))
    
    # Horizontal flip
    if flip_horizontal:
        augmented = cv2.flip(augmented, 1)
    
    # Vertical flip
    if flip_vertical:
        augmented = cv2.flip(augmented, 0)
    
    return augmented


def preprocess_pipeline(image_path, target_size=(224, 224), enhance=True, denoise=True):
    """
    Complete preprocessing pipeline for medical images.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target dimensions for resizing
        enhance (bool): Whether to apply CLAHE enhancement
        denoise (bool): Whether to apply denoising
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Load image
    image = load_image(image_path)
    
    # Resize
    image = resize_image(image, target_size)
    
    # Denoise if requested
    if denoise:
        image = denoise_image(image, method='bilateral')
    
    # Enhance if requested
    if enhance and len(image.shape) == 2:  # Only for grayscale
        image = apply_clahe(image)
    
    # Normalize
    image = normalize_image(image)
    
    return image
