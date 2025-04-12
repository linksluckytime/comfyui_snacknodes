"""ImageScaler node for ComfyUI that provides image scaling and transformation capabilities."""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Tuple
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

class ImageScaler:
    """A node for scaling and transforming images with various options."""
    
    CATEGORY = "SnackNodes"

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        """Define the input types for this node.
        
        Returns:
            Dictionary containing input type definitions
        """
        return {
            "required": {
                "image": ("IMAGE", {"description": "Input image tensor (B,H,W,C format)"}),
                "KeepProportion": ("BOOLEAN", {
                    "default": True,
                    "description": "Maintain image aspect ratio during scaling"
                }),
                "Pixels": (list(PIXEL_BUDGETS.keys()), {
                    "default": "1024px²",
                    "description": "Maximum number of pixels in the output image"
                }),
                "ScalingFactor": (SCALING_FACTORS, {
                    "default": 64,
                    "description": "Factor to align dimensions to"
                }),
                "RelativePosition": (RELATIVE_POSITIONS, {
                    "default": "Center",
                    "description": "Position of the image relative to the output canvas"
                }),
                "Supersampling": (list(INTERPOLATION_METHODS.keys()), {
                    "default": "None",
                    "description": "Interpolation method for scaling"
                }),
                "Padding": ("BOOLEAN", {
                    "default": False,
                    "description": "Add padding to maintain aspect ratio"
                }),
                "PaddingElements": (list(PADDING_COLORS.keys()), {
                    "default": "Transparent",
                    "description": "Color of the padding"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "scale_image"

    def scale_image(self, image: torch.Tensor, KeepProportion: bool, Pixels: str, 
                   ScalingFactor: int, RelativePosition: str, Supersampling: str, 
                   Padding: bool, PaddingElements: str) -> Tuple[torch.Tensor, int, int]:
        """Scale and transform an input image according to specified parameters.
        
        Args:
            image: Input image tensor
            KeepProportion: Whether to maintain aspect ratio
            Pixels: Maximum pixel budget
            ScalingFactor: Factor to align dimensions to
            RelativePosition: Position of image in output
            Supersampling: Interpolation method
            Padding: Whether to add padding
            PaddingElements: Color of padding
            
        Returns:
            Tuple containing scaled image tensor and its dimensions
            
        Raises:
            ValueError: If input tensor is invalid or parameters are invalid
        """
        # Validate inputs
        if len(image.shape) != 4:
            raise ValueError("Input tensor must be 4-dimensional (B,H,W,C)")
            
        max_pixels = PIXEL_BUDGETS.get(Pixels)
        if not max_pixels:
            raise ValueError(f"Invalid pixel budget: {Pixels}")

        try:
            # Extract image dimensions and convert to PIL Image
            batch_size, height, width, channels = image.shape
            img = Image.fromarray((image[0].numpy() * 255).astype("uint8"))

            # Ensure image is in RGBA mode for transparent padding
            if PaddingElements == "Transparent":
                img = img.convert("RGBA")

            # Calculate target dimensions
            if KeepProportion:
                new_width, new_height = calculate_dimensions(
                    width, height, max_pixels, ScalingFactor
                )
                img = img.resize(
                    (new_width, new_height), 
                    getattr(Image, INTERPOLATION_METHODS[Supersampling])
                )
            else:
                target_size = math.isqrt(max_pixels)
                target_size = (target_size // ScalingFactor) * ScalingFactor

                can_fit = (width <= target_size and height <= target_size) or \
                         (width * target_size / height <= target_size and 
                          height * target_size / width <= target_size)

                if not can_fit:
                    img = crop_image(img, target_size, target_size, RelativePosition)
                else:
                    if Padding:
                        img = scale_with_padding(
                            img, target_size, target_size, RelativePosition, 
                            Supersampling, PADDING_COLORS[PaddingElements]
                        )
                    else:
                        img = img.resize(
                            (target_size, target_size), 
                            getattr(Image, INTERPOLATION_METHODS[Supersampling])
                        )

            # Convert back to tensor
            output_image = torch.from_numpy(np.array(img).astype(np.float32)) / 255.0
            output_image = output_image.unsqueeze(0)

            return output_image, img.width, img.height

        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}") 