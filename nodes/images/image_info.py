"""Image info node for ComfyUI.

This node provides detailed information about input images, including dimensions,
channels, batch size, and other relevant properties. It's useful for debugging
and understanding the characteristics of images in your workflow.
"""

import torch
from typing import Dict, Tuple
from ..common.base_node import BaseNode

class ImageInfo(BaseNode):
    """A node for retrieving image information."""
    
    CATEGORY = "SnackNodes"
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        """Define the input types for this node."""
        return {
            "required": {
                "image": ("IMAGE", {"description": "Input image tensor"}),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "batch_size", "channels")
    FUNCTION = "get_image_info"
    
    def get_image_info(self, image: torch.Tensor) -> Tuple[int, int, int, int]:
        """Get image information including dimensions, batch size and channels.
        
        Args:
            image: Input image tensor in (B,H,W,C) format
            
        Returns:
            Tuple containing width, height, batch size and channels
            
        Raises:
            ValueError: If input tensor is not 4-dimensional
        """
        if len(image.shape) != 4:
            raise ValueError("Input tensor must be 4-dimensional (B,H,W,C)")
            
        batch_size, height, width, channels = image.shape
        return (width, height, batch_size, channels) 