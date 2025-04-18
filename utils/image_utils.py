"""Image utility functions for comfyui_snacknodes."""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Any, Union

def _align_to_factor(value: int, factor: int, min_size: int) -> int:
    """Align value to nearest multiple of factor, ensuring minimum size.
    
    Args:
        value: Value to align
        factor: Alignment factor
        min_size: Minimum allowed value
        
    Returns:
        Aligned value
    """
    if value < min_size:
        return ((min_size + factor - 1) // factor) * factor
    return ((value + factor - 1) // factor) * factor

def calculate_dimensions(
    width: int,
    height: int,
    max_pixels: int,
    scaling_factor: int,
    keep_proportion: bool = True,
    min_size: int = 1
) -> Tuple[int, int]:
    """Calculate new image dimensions based on specified logic.
    
    Args:
        width: Original width
        height: Original height
        max_pixels: Maximum pixel count (area)
        scaling_factor: Size alignment factor
        keep_proportion: Whether to maintain image proportion
        min_size: Minimum size limit
        
    Returns:
        Tuple of (new_width, new_height)
    """
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive")
    if max_pixels <= 0:
        raise ValueError("Maximum pixels must be positive")
    if scaling_factor <= 0:
        raise ValueError("Scaling factor must be positive")
    if min_size <= 0:
        raise ValueError("Minimum size must be positive")
        
    # Calculate input image total pixels and ratio
    input_pixels = width * height
    original_ratio = width / height if height > 0 else 1.0
    
    if keep_proportion:
        # Calculate scale factor based on pixel budget
        if input_pixels > max_pixels:
            scale = np.sqrt(max_pixels / input_pixels)
        else:
            scale = np.sqrt(max_pixels / input_pixels) * 0.99
            
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
    else:
        # Square output
        new_width = new_height = int(np.sqrt(max_pixels))
    
    # Align dimensions to scaling factor
    new_width = _align_to_factor(new_width, scaling_factor, min_size)
    new_height = _align_to_factor(new_height, scaling_factor, min_size)
    
    # Ensure within pixel budget
    while new_width * new_height > max_pixels:
        if new_width >= new_height and new_width > scaling_factor:
            new_width -= scaling_factor
        elif new_height > scaling_factor:
            new_height -= scaling_factor
        else:
            break
    
    return new_width, new_height

def _calculate_position(
    position: str,
    source_size: int,
    target_size: int
) -> int:
    """Calculate position offset for image alignment.
    
    Args:
        position: Position string (left, center, right or top, center, bottom)
        source_size: Size of source dimension
        target_size: Size of target dimension
        
    Returns:
        Position offset
    """
    if position in ["left", "top"]:
        return 0
    elif position in ["right", "bottom"]:
        return target_size - source_size
    else:  # center
        return (target_size - source_size) // 2

def crop_image(
    image: Image.Image,
    target_width: int,
    target_height: int,
    position: str = "center",
    min_size: int = 1
) -> Image.Image:
    """Crop image to target dimensions.
    
    Args:
        image: Input image
        target_width: Target width
        target_height: Target height
        position: Image position
        min_size: Minimum size limit
        
    Returns:
        Cropped image
    """
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Target dimensions must be positive")
    if min_size <= 0:
        raise ValueError("Minimum size must be positive")
        
    # Get image dimensions
    width, height = image.size
    
    # Calculate crop dimensions
    crop_width = min(width, target_width)
    crop_height = min(height, target_height)
    
    # Calculate crop position
    left = _calculate_position(position, crop_width, width)
    top = _calculate_position(position, crop_height, height)
    
    # Perform crop
    return image.crop((left, top, left + crop_width, top + crop_height))

def scale_with_padding(
    image: Image.Image,
    target_width: int,
    target_height: int,
    position: str = "center",
    interpolation: Any = Image.NEAREST,
    padding_color: Tuple[int, int, int, int] = (0, 0, 0, 0),
    min_size: int = 1
) -> Image.Image:
    """Scale image with padding to target dimensions.
    
    Args:
        image: Input image
        target_width: Target width
        target_height: Target height
        position: Image position
        interpolation: Interpolation method
        padding_color: RGBA color for padding
        min_size: Minimum size limit
        
    Returns:
        Scaled and padded image
    """
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Target dimensions must be positive")
    if min_size <= 0:
        raise ValueError("Minimum size must be positive")
        
    # Get image dimensions
    width, height = image.size
    
    # Calculate scale factors
    width_scale = target_width / width
    height_scale = target_height / height
    scale = min(width_scale, height_scale)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = image.resize((new_width, new_height), interpolation)
    
    # Create new image with padding
    result = Image.new("RGBA" if image.mode == "RGBA" else "RGB", 
                      (target_width, target_height), 
                      padding_color)
    
    # Calculate paste position
    left = _calculate_position(position, new_width, target_width)
    top = _calculate_position(position, new_height, target_height)
    
    # Paste resized image
    result.paste(resized, (left, top))
    
    return result 