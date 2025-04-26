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
            self.debug_log(f"Target pixel budget: {pixel_budget}")
            
            # Calculate dimensions using utility function
            new_width, new_height = calculate_dimensions(
                width=width,
                height=height,
                max_pixels=pixel_budget,
                scaling_factor=scaling_factor,
                keep_proportion=keep_proportion,
                min_size=self.MIN_REASONABLE_SIZE
            )
            
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