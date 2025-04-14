"""ImageInfo node for ComfyUI that provides information about input images."""

import torch
from typing import Dict, Tuple
from .base_node import BaseNode

class ImageInfo(BaseNode):
    """A node that provides information about an input image tensor."""
    
    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "batch_size", "channels")
    FUNCTION = "get_image_info"

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

    def get_image_info(self, image: torch.Tensor) -> Tuple[int, int, int, int]:
        """Extract and return image dimensions.
        
        Args:
            image: Input image tensor in B,H,W,C format
            
        Returns:
            Tuple containing width, height, batch_size, and channels
            
        Raises:
            ValueError: If input tensor is not 4-dimensional
        """
        if len(image.shape) != 4:
            raise ValueError("Input tensor must be 4-dimensional (B,H,W,C)")
        batch_size, height, width, channels = image.shape
        return width, height, batch_size, channels
        
    def _process(self, **kwargs) -> Tuple[int, int, int, int]:
        """Process the node's inputs and return outputs.
        
        Args:
            **kwargs: Input parameters for the node
            
        Returns:
            Tuple containing width, height, batch_size, and channels
        """
        return self.get_image_info(kwargs["image"]) 