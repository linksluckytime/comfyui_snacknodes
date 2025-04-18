"""Image utility functions for comfyui_snacknodes."""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Any, Union

def _ensure_min_size(value: int, min_size: int) -> int:
    """Ensure value is not smaller than minimum size.
    
    Args:
        value: Value to check
        min_size: Minimum allowed value
        
    Returns:
        Value adjusted to minimum size if necessary
    """
    return max(min_size, value)

def _align_to_factor(value: int, factor: int, min_size: int) -> int:
    """Align value to nearest multiple of factor.
    
    Args:
        value: Value to align
        factor: Alignment factor
        min_size: Minimum allowed value
        
    Returns:
        Aligned value
    """
    aligned = (value // factor) * factor
    if aligned < min_size:
        aligned = ((value + factor - 1) // factor) * factor
    return aligned

def calculate_dimensions(
    width: int,
    height: int,
    max_pixels: int,
    scaling_factor: int,
    keep_proportion: bool = True,
    min_size: int = 1
) -> Tuple[int, int]:
    """Calculate new image dimensions based on specified logic.
    
    Logic flow:
    1. Calculate input image ratio
    2. Compare input image total pixels with target pixel limit
    3. Calculate new dimensions based on different cases
    
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
    # Calculate input image total pixels and ratio
    input_pixels = width * height
    original_ratio = width / height if height > 0 else 1.0
    
    # Calculate base target size (square side length)
    target_size = int(np.sqrt(max_pixels))
    
    # Initialize new dimensions
    new_width, new_height = width, height
    
    # Case 1: Input image pixels > target pixels
    if input_pixels > max_pixels:
        if keep_proportion:
            # Scale down while maintaining proportion
            scale = np.sqrt(max_pixels / input_pixels)
            new_width = int(width * scale)
            new_height = int(height * scale)
        else:
            # Stretch to target square without maintaining proportion
            new_width = target_size
            new_height = target_size
    
    # Case 2: Input image pixels < target pixels
    elif input_pixels < max_pixels:
        if keep_proportion:
            # Scale up while maintaining proportion, but not exceeding target area
            max_scale = np.sqrt(max_pixels / input_pixels)
            scale = max_scale * 0.99  # Use 99% to prevent rounding errors
            new_width = int(width * scale)
            new_height = int(height * scale)
        else:
            # Stretch to target square without maintaining proportion
            new_width = target_size
            new_height = target_size
    
    # Case 3: Input image pixels = target pixels
    else:  # input_pixels == max_pixels
        # Check if both sides are divisible by scaling factor
        if width % scaling_factor == 0 and height % scaling_factor == 0:
            return width, height
    
    # Ensure dimensions are not smaller than minimum
    new_width = _ensure_min_size(new_width, min_size)
    new_height = _ensure_min_size(new_height, min_size)
    
    # Align to scaling factor
    if keep_proportion:
        new_width = _align_to_factor(new_width, scaling_factor, min_size)
        new_height = _align_to_factor(new_height, scaling_factor, min_size)
    else:
        # Ensure target size is divisible by scaling factor
        new_width = _align_to_factor(target_size, scaling_factor, min_size)
        new_height = _align_to_factor(target_size, scaling_factor, min_size)
    
    return new_width, new_height

def _calculate_position(
    position: str,
    source_size: int,
    target_size: int
) -> int:
    """Calculate position offset based on alignment.
    
    Args:
        position: Position string (e.g., "left", "center", "right")
        source_size: Size of source element
        target_size: Size of target container
        
    Returns:
        Position offset
    """
    position = position.lower()
    if "left" in position or "top" in position:
        return 0
    elif "right" in position or "bottom" in position:
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
    """Crop image to specified dimensions.
    
    Args:
        image: Input PIL image
        target_width: Target width
        target_height: Target height
        position: Crop position (e.g., "center", "top left")
        min_size: Minimum width and height
        
    Returns:
        Cropped PIL image
        
    Raises:
        ValueError: If image is None or invalid
    """
    if image is None:
        raise ValueError("Input image cannot be None")
        
    # Ensure target dimensions not smaller than minimum
    target_width = _ensure_min_size(target_width, min_size)
    target_height = _ensure_min_size(target_height, min_size)
    
    # Get original dimensions
    width, height = image.size
    
    # Return if dimensions already match
    if height == target_height and width == target_width:
        return image
    
    # Scale up if target larger than source
    if width < target_width or height < target_height:
        scale_ratio = max(target_width / width, target_height / height)
        new_width = int(width * scale_ratio)
        new_height = int(height * scale_ratio)
        image = image.resize((new_width, new_height), Image.LANCZOS)
        width, height = new_width, new_height
    
    # Calculate crop position
    x_start = _calculate_position(position, target_width, width)
    y_start = _calculate_position(position, target_height, height)
    
    # Ensure crop area within image bounds
    x_start = max(0, min(x_start, width - target_width))
    y_start = max(0, min(y_start, height - target_height))
    
    return image.crop((x_start, y_start, x_start + target_width, y_start + target_height))

def scale_with_padding(
    image: Image.Image,
    target_width: int,
    target_height: int,
    position: str = "center",
    interpolation: Any = Image.NEAREST,
    padding_color: Tuple[int, int, int, int] = (0, 0, 0, 0),
    min_size: int = 1
) -> Image.Image:
    """Scale image proportionally and add padding to maintain canvas size.
    
    Args:
        image: Input PIL image
        target_width: Target width
        target_height: Target height
        position: Image position on canvas
        interpolation: Interpolation method (PIL constant)
        padding_color: RGBA color for padding area
        min_size: Minimum width and height
        
    Returns:
        Scaled and padded PIL image
        
    Raises:
        ValueError: If image is None or invalid
    """
    if image is None:
        raise ValueError("Input image cannot be None")
        
    # Ensure target dimensions not smaller than minimum
    target_width = _ensure_min_size(target_width, min_size)
    target_height = _ensure_min_size(target_height, min_size)
    
    # Get original dimensions
    width, height = image.size
    
    # Return if dimensions already match
    if height == target_height and width == target_width:
        return image
    
    # Calculate proportional scale ratio
    ratio = min(target_width / width, target_height / height)
    
    # Calculate scaled dimensions
    new_width = _ensure_min_size(int(width * ratio), min_size)
    new_height = _ensure_min_size(int(height * ratio), min_size)
    
    # Scale image
    resized_image = image.resize((new_width, new_height), interpolation)
    
    # Create new canvas
    new_image = Image.new("RGBA", (target_width, target_height), padding_color)
    
    # Calculate paste position
    x = _calculate_position(position, new_width, target_width)
    y = _calculate_position(position, new_height, target_height)
    
    # Paste scaled image
    if resized_image.mode == "RGBA":
        new_image.paste(resized_image, (x, y), resized_image)
    else:
        new_image.paste(resized_image, (x, y))
    
    return new_image 