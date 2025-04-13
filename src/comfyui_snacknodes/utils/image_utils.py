"""Image utility functions for comfyui_snacknodes."""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional

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
    image: torch.Tensor,
    width: int,
    height: int,
    position: str = "Center"
) -> torch.Tensor:
    """Crop an image to specified dimensions.
    
    Args:
        image: Input image tensor (B,H,W,C)
        width: Target width
        height: Target height
        position: Position to crop from
        
    Returns:
        Cropped image tensor
    """
    b, h, w, c = image.shape
    if h == height and w == width:
        return image
    
    y_start = 0
    x_start = 0
    
    if position == "Center":
        y_start = (h - height) // 2
        x_start = (w - width) // 2
    elif position == "Top-Left":
        y_start = 0
        x_start = 0
    elif position == "Top-Right":
        y_start = 0
        x_start = w - width
    elif position == "Bottom-Left":
        y_start = h - height
        x_start = 0
    elif position == "Bottom-Right":
        y_start = h - height
        x_start = w - width
    
    return image[:, y_start:y_start+height, x_start:x_start+width, :]

def scale_with_padding(
    image: torch.Tensor,
    width: int,
    height: int,
    padding_color: Tuple[int, int, int, int] = (0, 0, 0, 0)
) -> torch.Tensor:
    """Scale an image with padding to maintain aspect ratio.
    
    Args:
        image: Input image tensor (B,H,W,C)
        width: Target width
        height: Target height
        padding_color: RGBA color for padding
        
    Returns:
        Scaled and padded image tensor
    """
    b, h, w, c = image.shape
    if h == height and w == width:
        return image
    
    # Convert to PIL Image for processing
    pil_image = Image.fromarray((image[0].numpy() * 255).astype(np.uint8))
    
    # Create new image with padding
    new_image = Image.new("RGBA", (width, height), padding_color)
    
    # Calculate position to paste
    x = (width - w) // 2
    y = (height - h) // 2
    
    # Paste original image
    new_image.paste(pil_image, (x, y))
    
    # Convert back to tensor
    result = torch.from_numpy(np.array(new_image)).float() / 255.0
    return result.unsqueeze(0) 