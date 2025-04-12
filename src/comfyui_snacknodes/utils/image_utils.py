"""Utility functions for image processing."""

import math
from typing import Tuple
from PIL import Image
from ..config.constants import INTERPOLATION_METHODS

def calculate_dimensions(width: int, height: int, max_pixels: int, 
                        scale_factor: int) -> Tuple[int, int]:
    """Calculate new dimensions while maintaining aspect ratio.
    
    Args:
        width: Original image width
        height: Original image height
        max_pixels: Maximum number of pixels in output
        scale_factor: Factor to align dimensions to
        
    Returns:
        Tuple of (new_width, new_height)
    """
    aspect_ratio = width / height
    new_width = min(width, math.isqrt(int(max_pixels * aspect_ratio)))
    new_height = min(height, math.isqrt(int(max_pixels / aspect_ratio)))

    new_width = (new_width // scale_factor) * scale_factor
    new_height = (new_height // scale_factor) * scale_factor
    return new_width, new_height

def crop_image(img: Image.Image, target_width: int, target_height: int, 
               reference_position: str) -> Image.Image:
    """Crop image to target dimensions based on reference position.
    
    Args:
        img: Input PIL Image
        target_width: Desired output width
        target_height: Desired output height
        reference_position: Position to use as reference for cropping
        
    Returns:
        Cropped PIL Image
    """
    width, height = img.size
    left = 0
    top = 0
    right = width
    bottom = height

    if width > target_width:
        if "Left" in reference_position:
            right = target_width
        elif "Right" in reference_position:
            left = width - target_width
        else:  # Center
            left = (width - target_width) // 2
            right = left + target_width

    if height > target_height:
        if "Top" in reference_position:
            bottom = target_height
        elif "Bottom" in reference_position:
            top = height - target_height
        else:  # Center
            top = (height - target_height) // 2
            bottom = top + target_height

    return img.crop((left, top, right, bottom))

def scale_with_padding(img: Image.Image, target_width: int, target_height: int, 
                      reference_position: str, supersampling_method: str, 
                      padding: str) -> Image.Image:
    """Scale image with padding while maintaining aspect ratio.
    
    Args:
        img: Input PIL Image
        target_width: Desired output width
        target_height: Desired output height
        reference_position: Position to use as reference for padding
        supersampling_method: Interpolation method to use
        padding: Color to use for padding
        
    Returns:
        Scaled and padded PIL Image
    """
    width, height = img.size
    aspect_ratio = width / height

    if width * target_height / height <= target_width:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    else:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)

    img = img.resize((new_width, new_height), 
                    getattr(Image, INTERPOLATION_METHODS[supersampling_method]))

    new_img = Image.new("RGBA" if padding == "Transparent" else "RGB", 
                       (target_width, target_height), 
                       padding)

    paste_x = 0
    paste_y = 0

    if "Left" in reference_position:
        paste_x = 0
    elif "Right" in reference_position:
        paste_x = target_width - new_width
    else:  # Center
        paste_x = (target_width - new_width) // 2

    if "Top" in reference_position:
        paste_y = 0
    elif "Bottom" in reference_position:
        paste_y = target_height - new_height
    else:  # Center
        paste_y = (target_height - new_height) // 2

    new_img.paste(img, (paste_x, paste_y))
    return new_img 