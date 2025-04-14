"""Image info node for ComfyUI.

This node provides detailed information about input images, including dimensions,
channels, batch size, and other relevant properties. It's useful for debugging
and understanding the characteristics of images in your workflow.
"""

import logging
import torch
from .base_node import BaseNode

class ImageInfo(BaseNode):
    """Node for displaying image information."""
    
    CATEGORY = "SnackNodes"
    
    # 添加UI可显示的值
    @classmethod
    def IS_CHANGED(cls, image):
        """通知ComfyUI节点需要重新渲染，用于显示动态值。"""
        return float("NaN")
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define the input types for the node."""
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "batch_size", "channels")
    FUNCTION = "get_image_info"
    
    def get_image_info(self, image: torch.Tensor):
        """Get image information."""
        # Get image dimensions
        batch_size, height, width, channels = image.shape
        
        return (width, height, batch_size, channels) 