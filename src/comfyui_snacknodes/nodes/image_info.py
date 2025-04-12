"""ImageInfo node for ComfyUI that provides information about input images."""

import torch
from typing import Dict, Tuple

class ImageInfo:
    """A node that provides information about an input image tensor."""
    
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
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("batch_size", "height", "width", "channels")
    FUNCTION = "get_image_info"

    def get_image_info(self, image: torch.Tensor) -> Tuple[int, int, int, int]:
        """Extract and return image dimensions.
        
        Args:
            image: Input image tensor in B,H,W,C format
            
        Returns:
            Tuple containing batch_size, height, width, and channels
            
        Raises:
            ValueError: If input tensor is not 4-dimensional
        """
        if len(image.shape) != 4:
            raise ValueError("Input tensor must be 4-dimensional (B,H,W,C)")
        return image.shape 