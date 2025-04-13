"""ImageScaler node for ComfyUI that provides image scaling and transformation capabilities."""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Tuple
from .base_node import BaseNode
from ..config.constants import (
    PADDING_COLORS,
    INTERPOLATION_METHODS,
    RELATIVE_POSITIONS,
    PIXEL_BUDGETS,
    SCALING_FACTORS
)
from ..utils.image_utils import (
    calculate_dimensions,
    crop_image,
    scale_with_padding
)

class ImageScaler(BaseNode):
    """A node for scaling and transforming images with various options."""
    
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "scale_image"

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        """Define the input types for this node.
        
        Returns:
            Dictionary containing input type definitions
        """
        return {
            "required": {
                "image": ("IMAGE", {"description": "Input image tensor (B,H,W,C format)"}),
                "keep_proportion": ("BOOLEAN", {
                    "default": True,
                    "description": "Maintain the original aspect ratio of the image"
                }),
                "pixels": (list(PIXEL_BUDGETS.keys()), {
                    "default": "1024px²",
                    "description": "Maximum number of pixels in the output image"
                }),
                "scaling_factor": (SCALING_FACTORS, {
                    "default": 64,
                    "description": "Factor to align dimensions to (e.g., 64 for SDXL compatibility)"
                }),
                "relative_position": (RELATIVE_POSITIONS, {
                    "default": "center",
                    "description": "Position of the image relative to the output canvas (9-grid layout)"
                }),
                "supersampling": (list(INTERPOLATION_METHODS.keys()), {
                    "default": "none",
                    "description": "Interpolation method for scaling (none = no interpolation)"
                }),
                "padding": ("BOOLEAN", {
                    "default": False,
                    "description": "Add padding to maintain aspect ratio"
                }),
                "padding_color": (list(PADDING_COLORS.keys()), {
                    "default": "transparent",
                    "description": "Color of the padding area"
                }),
            },
        }

    def scale_image(self, image: torch.Tensor, keep_proportion: bool, pixels: str, 
                   scaling_factor: int, relative_position: str, supersampling: str, 
                   padding: bool, padding_color: str) -> Tuple[torch.Tensor, int, int]:
        """Scale and transform the input image according to the specified parameters.
        
        Args:
            image: Input image tensor
            keep_proportion: Whether to maintain aspect ratio
            pixels: Maximum number of pixels
            scaling_factor: Factor to align dimensions to
            relative_position: Position of the image
            supersampling: Interpolation method
            padding: Whether to add padding
            padding_color: Color of the padding
            
        Returns:
            Tuple containing scaled image and dimensions
        """
        # Validate inputs
        if len(image.shape) != 4:
            raise ValueError("Input tensor must be 4-dimensional (B,H,W,C)")
            
        max_pixels = PIXEL_BUDGETS.get(pixels)
        if not max_pixels:
            raise ValueError(f"Invalid pixel budget: {pixels}")

        try:
            # Extract image dimensions and convert to PIL Image
            batch_size, height, width, channels = image.shape
            img = Image.fromarray((image[0].numpy() * 255).astype("uint8"))

            # Ensure image is in RGBA mode for transparent padding
            if padding_color == "Transparent":
                img = img.convert("RGBA")

            # Calculate target dimensions
            if keep_proportion:
                new_width, new_height = calculate_dimensions(
                    width, height, max_pixels, scaling_factor
                )
                img = img.resize(
                    (new_width, new_height), 
                    getattr(Image, INTERPOLATION_METHODS[supersampling])
                )
            else:
                target_size = math.isqrt(max_pixels)
                target_size = (target_size // scaling_factor) * scaling_factor

                can_fit = (width <= target_size and height <= target_size) or \
                         (width * target_size / height <= target_size and 
                          height * target_size / width <= target_size)

                if not can_fit:
                    img = crop_image(img, target_size, target_size, relative_position)
                else:
                    if padding:
                        img = scale_with_padding(
                            img, target_size, target_size, relative_position, 
                            supersampling, PADDING_COLORS[padding_color]
                        )
                    else:
                        img = img.resize(
                            (target_size, target_size), 
                            getattr(Image, INTERPOLATION_METHODS[supersampling])
                        )

            # Convert back to tensor
            output_image = torch.from_numpy(np.array(img).astype(np.float32)) / 255.0
            output_image = output_image.unsqueeze(0)

            return output_image, img.width, img.height

        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}") 