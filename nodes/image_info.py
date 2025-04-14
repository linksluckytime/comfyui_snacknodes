"""Image info node for ComfyUI.

This node provides detailed information about input images, including dimensions,
channels, batch size, and other relevant properties. It's useful for debugging
and understanding the characteristics of images in your workflow.
"""

import torch
from .base_node import BaseNode

class ImageInfo(BaseNode):
    """节点用于获取图像信息。"""
    
    CATEGORY = "SnackNodes"
    
    @classmethod
    def INPUT_TYPES(cls):
        """定义节点的输入类型。"""
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "batch_size", "channels")
    FUNCTION = "get_image_info"
    
    def get_image_info(self, image: torch.Tensor):
        """获取图像信息。"""
        # 获取图像维度
        batch_size, height, width, channels = image.shape
        
        return (width, height, batch_size, channels) 