"""ImageScaler node for ComfyUI that provides image scaling and transformation capabilities."""

import torch
import numpy as np
import math
from PIL import Image
from typing import Dict, Tuple, Optional, Any, List, Union
from ..common.base_node import BaseNode
from ...config.constants import (
    PADDING_COLORS,
    INTERPOLATION_METHODS,
    RELATIVE_POSITIONS,
    PIXEL_BUDGETS,
    SCALING_FACTORS
)
from ...utils.image_utils import (
    calculate_dimensions,
    crop_image,
    scale_with_padding
)

def _generate_candidate_sizes(
    width: int,
    height: int,
    scaling_factor: int,
    min_size: int
) -> List[Tuple[int, int]]:
    """Generate candidate sizes by rounding up and down to scaling factor.
    
    Args:
        width: Original width
        height: Original height
        scaling_factor: Size alignment factor
        min_size: Minimum allowed size
        
    Returns:
        List of candidate (width, height) tuples
        
    Raises:
        ValueError: If input parameters are invalid
    """
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive")
    if scaling_factor <= 0:
        raise ValueError("Scaling factor must be positive")
    if min_size <= 0:
        raise ValueError("Minimum size must be positive")
        
    candidates = []
    
    # Round up sizes
    w1 = ((width + scaling_factor - 1) // scaling_factor) * scaling_factor
    h1 = ((height + scaling_factor - 1) // scaling_factor) * scaling_factor
    
    # Round down sizes
    w2 = (width // scaling_factor) * scaling_factor
    h2 = (height // scaling_factor) * scaling_factor
    
    # Generate all possible combinations
    for w in [w1, w2]:
        if w < min_size:
            continue
        for h in [h1, h2]:
            if h < min_size:
                continue
            candidates.append((w, h))
            
    return candidates

def _select_best_size(
    candidates: List[Tuple[int, int, float, int]],
    aspect_ratio: float,
    pixel_budget: int
) -> Optional[Tuple[int, int]]:
    """Select the best size from candidates based on ratio deviation and pixel count.
    
    Args:
        candidates: List of (width, height, ratio_diff, pixels) tuples
        aspect_ratio: Original aspect ratio
        pixel_budget: Maximum allowed pixels
        
    Returns:
        Tuple of (width, height) or None if no valid candidates
        
    Raises:
        ValueError: If input parameters are invalid
    """
    if not candidates:
        return None
    if aspect_ratio <= 0:
        raise ValueError("Aspect ratio must be positive")
    if pixel_budget <= 0:
        raise ValueError("Pixel budget must be positive")
        
    # Sort by ratio deviation first, then pixel count
    candidates.sort(key=lambda x: (x[2], -x[3]))
    return candidates[0][:2]

def _process_image(
    image: torch.Tensor,
    width: int,
    height: int,
    interpolation: Any,
    padding: bool,
    padding_color: Tuple[int, int, int, int],
    position: str,
    min_size: int
) -> Image.Image:
    """Process image with specified parameters.
    
    Args:
        image: Input image tensor
        width: Target width
        height: Target height
        interpolation: Interpolation method
        padding: Whether to use padding
        padding_color: RGBA color for padding
        position: Image position
        min_size: Minimum size
        
    Returns:
        Processed PIL image
        
    Raises:
        ValueError: If input parameters are invalid
    """
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive")
    if min_size <= 0:
        raise ValueError("Minimum size must be positive")
        
    # Convert tensor to PIL image
    img_data = (image[0].cpu().numpy() * 255).astype(np.uint8)
    channels = image.shape[-1]
    img = Image.fromarray(img_data, 'RGBA' if channels == 4 else 'RGB')
    
    # Process image
    if padding:
        return scale_with_padding(
            image=img,
            target_width=width,
            target_height=height,
            position=position,
            interpolation=interpolation,
            padding_color=padding_color,
            min_size=min_size
        )
    else:
        return img.resize((width, height), interpolation)

class ImageScaler(BaseNode):
    """A node for scaling and transforming images with various options."""
    
    # Class attributes
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "scale_image"
    OUTPUT_NODE = True
    CATEGORY = "SnackNodes"
    
    # Constants
    MIN_REASONABLE_SIZE = 32
    ENABLE_DEBUG = True
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """Define the input types for this node.
        
        Returns:
            Dictionary containing input type definitions
        """
        return {
            "required": {
                "image": ("IMAGE", {"description": "Input image (B,H,W,C format)"}),
                "keep_proportion": ("BOOLEAN", {
                    "default": True,
                    "description": "Maintain aspect ratio"
                }),
                "pixels": (list(PIXEL_BUDGETS.keys()), {
                    "default": "1024px²",
                    "description": "Target pixel count"
                }),
                "scaling_factor": (SCALING_FACTORS, {
                    "default": 64,
                    "description": "Size alignment factor"
                }),
                "relative_position": (RELATIVE_POSITIONS, {
                    "default": "center",
                    "description": "Image position"
                }),
                "supersampling": (list(INTERPOLATION_METHODS.keys()), {
                    "default": "none",
                    "description": "Interpolation method"
                }),
                "padding": ("BOOLEAN", {
                    "default": False,
                    "description": "Use padding"
                }),
                "padding_method": (list(PADDING_COLORS.keys()), {
                    "default": "transparent",
                    "description": "Padding method"
                }),
            },
        }
    
    def debug_log(self, *args, **kwargs) -> None:
        """Output debug information.
        
        Args:
            *args: Arguments to print
            **kwargs: Keyword arguments to print
        """
        if self.ENABLE_DEBUG:
            print("[ImageScaler]", *args, **kwargs)
    
    def scale_image(
        self,
        image: torch.Tensor,
        keep_proportion: bool,
        pixels: str,
        scaling_factor: int,
        relative_position: str,
        supersampling: str,
        padding: bool,
        padding_method: str
    ) -> Tuple[torch.Tensor, int, int]:
        """Scale and transform image according to specified logic.
        
        Args:
            image: Input image tensor
            keep_proportion: Whether to maintain aspect ratio
            pixels: Target pixel count
            scaling_factor: Size alignment factor
            relative_position: Image position
            supersampling: Interpolation method
            padding: Whether to use padding
            padding_method: Padding method
            
        Returns:
            Tuple of (output image, width, height)
            
        Raises:
            ValueError: If input parameters are invalid
        """
        try:
            # Validate input
            if len(image.shape) != 4:
                raise ValueError("Input tensor must be 4-dimensional (B,H,W,C)")
                
            # Get pixel budget
            pixel_budget = PIXEL_BUDGETS.get(pixels)
            if not pixel_budget:
                raise ValueError(f"Invalid pixel budget: {pixels}")
            
            # Extract image info
            batch_size, height, width, channels = image.shape
            input_pixels = width * height
            aspect_ratio = width / height
            
            # Log initial info
            self.debug_log(f"Input: {width}x{height}, Pixels: {input_pixels}")
            self.debug_log(f"Ratio: {aspect_ratio:.3f}")
            self.debug_log(f"Target: {int(math.sqrt(pixel_budget))}x{int(math.sqrt(pixel_budget))}")
            
            # Calculate target dimensions
            if keep_proportion:
                # Calculate base dimensions
                if input_pixels > pixel_budget:
                    new_width = min(width, int(math.sqrt(pixel_budget * aspect_ratio)))
                    new_height = min(height, int(math.sqrt(pixel_budget / aspect_ratio)))
                else:
                    scale = math.sqrt(pixel_budget / input_pixels) * 0.99
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                
                # Generate and select best size
                candidates = []
                for w, h in _generate_candidate_sizes(new_width, new_height, scaling_factor, self.MIN_REASONABLE_SIZE):
                    pixels = w * h
                    if pixels > pixel_budget:
                        continue
                    ratio = w / h
                    ratio_diff = abs(ratio - aspect_ratio) / aspect_ratio
                    candidates.append((w, h, ratio_diff, pixels))
                
                if candidates:
                    new_width, new_height = _select_best_size(candidates, aspect_ratio, pixel_budget)
                else:
                    new_width = max(self.MIN_REASONABLE_SIZE, (new_width // scaling_factor) * scaling_factor)
                    new_height = max(self.MIN_REASONABLE_SIZE, (new_height // scaling_factor) * scaling_factor)
            else:
                # Square output
                square_size = int(math.sqrt(pixel_budget))
                square_size = ((square_size + scaling_factor - 1) // scaling_factor) * scaling_factor
                new_width = new_height = square_size
            
            # Ensure within pixel budget
            while new_width * new_height > pixel_budget:
                if new_width >= new_height and new_width > scaling_factor:
                    new_width -= scaling_factor
                elif new_height > scaling_factor:
                    new_height -= scaling_factor
                else:
                    break
            
            # Final size validation
            new_width = max(self.MIN_REASONABLE_SIZE, new_width)
            new_height = max(self.MIN_REASONABLE_SIZE, new_height)
            
            # Log results
            self.debug_log(f"Target: {new_width}x{new_height}")
            self.debug_log(f"Pixels: {new_width * new_height}")
            if keep_proportion:
                final_ratio = new_width / new_height
                ratio_diff = abs(final_ratio - aspect_ratio) / aspect_ratio
                self.debug_log(f"Final ratio: {final_ratio:.3f} (Deviation: {ratio_diff:.1%})")
            
            # Process image
            interp_method = INTERPOLATION_METHODS.get(supersampling, Image.NEAREST)
            pad_color = PADDING_COLORS.get(padding_method, (0, 0, 0, 0))
            
            img_resized = _process_image(
                image=image,
                width=new_width,
                height=new_height,
                interpolation=interp_method,
                padding=padding,
                padding_color=pad_color,
                position=relative_position,
                min_size=self.MIN_REASONABLE_SIZE
            )
            
            # Convert back to tensor
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            output_channels = img_array.shape[-1]
            
            # Handle channel count
            if output_channels != channels:
                if padding_method == "transparent":
                    if output_channels == 4 and channels == 3:
                        channels = 4
                    elif output_channels == 3 and channels == 4:
                        alpha = np.ones((*img_array.shape[:-1], 1), dtype=np.float32)
                        img_array = np.concatenate([img_array, alpha], axis=-1)
            
            # Create output tensor
            output = torch.from_numpy(img_array).unsqueeze(0)
            
            return (output, new_width, new_height)
            
        except Exception as e:
            self.debug_log(f"Error: {str(e)}")
            raise 