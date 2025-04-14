"""Image utility functions for comfyui_snacknodes."""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Any

def calculate_dimensions(
    width: int,
    height: int,
    max_pixels: int,
    scaling_factor: int,
    keep_proportion: bool = True
) -> Tuple[int, int]:
    """Calculate new dimensions for image scaling.
    
    Args:
        width: Original width
        height: Original height
        max_pixels: Maximum number of pixels
        scaling_factor: Factor to align dimensions to
        keep_proportion: Whether to maintain aspect ratio
        
    Returns:
        Tuple of (new_width, new_height)
    """
    if keep_proportion:
        ratio = width / height
        if width * height > max_pixels:
            new_height = int(np.sqrt(max_pixels / ratio))
            new_width = int(new_height * ratio)
        else:
            new_width = width
            new_height = height
    else:
        new_width = min(width, max_pixels)
        new_height = min(height, max_pixels)
    
    # Align to scaling factor
    new_width = (new_width // scaling_factor) * scaling_factor
    new_height = (new_height // scaling_factor) * scaling_factor
    
    return new_width, new_height

def crop_image(
    image: Image.Image,
    target_width: int,
    target_height: int,
    position: str = "center"
) -> Image.Image:
    """Crop an image to specified dimensions.
    
    Args:
        image: Input PIL Image
        target_width: Target width
        target_height: Target height
        position: Position to crop from (e.g., "center", "top left", etc.)
        
    Returns:
        Cropped PIL Image
    """
    width, height = image.size
    if height == target_height and width == target_width:
        return image
    
    y_start = 0
    x_start = 0
    
    # Parse position
    position = position.lower()
    if "top" in position:
        y_start = 0
    elif "bottom" in position:
        y_start = height - target_height
    else:  # center
        y_start = (height - target_height) // 2
        
    if "left" in position:
        x_start = 0
    elif "right" in position:
        x_start = width - target_width
    else:  # center
        x_start = (width - target_width) // 2
    
    # Ensure valid crop dimensions
    if x_start < 0: x_start = 0
    if y_start < 0: y_start = 0
    if x_start + target_width > width: x_start = width - target_width
    if y_start + target_height > height: y_start = height - target_height
    
    return image.crop((x_start, y_start, x_start + target_width, y_start + target_height))

def scale_with_padding(
    image: Image.Image,
    target_width: int,
    target_height: int,
    position: str = "center",
    interpolation: str = "none",
    padding_color: Tuple[int, int, int, int] = (0, 0, 0, 0)
) -> Image.Image:
    """Scale an image with padding to maintain aspect ratio.
    
    Args:
        image: Input PIL Image
        target_width: Target width
        target_height: Target height
        position: Position of the image on the canvas
        interpolation: Interpolation method name or PIL constant
        padding_color: RGBA color for padding
        
    Returns:
        Scaled and padded PIL Image
    """
    width, height = image.size
    if height == target_height and width == target_width:
        return image
    
    # Calculate scaling ratio
    ratio = min(target_width / width, target_height / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    # Resize image
    if isinstance(interpolation, str):
        from PIL import Image as PILImage
        interp_method = getattr(PILImage, interpolation.upper(), PILImage.NEAREST)
    else:
        interp_method = interpolation
    
    resized_image = image.resize((new_width, new_height), interp_method)
    
    # Create new image with padding
    new_image = Image.new("RGBA", (target_width, target_height), padding_color)
    
    # Calculate position to paste
    position = position.lower()
    
    if "left" in position:
        x = 0
    elif "right" in position:
        x = target_width - new_width
    else:  # center
        x = (target_width - new_width) // 2
        
    if "top" in position:
        y = 0
    elif "bottom" in position:
        y = target_height - new_height
    else:  # center
        y = (target_height - new_height) // 2
    
    # Paste original image
    new_image.paste(resized_image, (x, y))
    
    return new_image 