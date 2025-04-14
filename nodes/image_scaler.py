"""ImageScaler node for ComfyUI that provides image scaling and transformation capabilities."""

import torch
import numpy as np
import math
from PIL import Image
from typing import Dict, Tuple, Optional, Any
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
            # Extract image dimensions
            batch_size, height, width, channels = image.shape
            
            # 优化：只处理第一批次的图像，保持批次维度不变
            # 只转换要处理的那一张图片，而不是整个批次
            first_image = image[0].cpu().numpy()
            
            # 优化：避免不必要的数据类型转换
            img_data = (first_image * 255).astype(np.uint8)
            img = Image.fromarray(img_data)

            # 优化：只在需要时才转换图像模式
            need_alpha = padding_color.lower() == "transparent" and padding
            if need_alpha and img.mode != "RGBA":
                img = img.convert("RGBA")

            # Calculate target dimensions
            if keep_proportion:
                new_width, new_height = calculate_dimensions(
                    width, height, max_pixels, scaling_factor
                )
                
                # 优化：只在尺寸实际变化时才调整大小
                if new_width != width or new_height != height:
                    interp_method = INTERPOLATION_METHODS.get(supersampling, Image.NEAREST)
                    img = img.resize((new_width, new_height), interp_method)
            else:
                target_size = math.isqrt(max_pixels)
                target_size = (target_size // scaling_factor) * scaling_factor

                can_fit = (width <= target_size and height <= target_size) or \
                         (width * target_size / height <= target_size and 
                          height * target_size / width <= target_size)

                if not can_fit:
                    img = crop_image(img, target_size, target_size, relative_position)
                elif padding:
                    img = scale_with_padding(
                        img, target_size, target_size, relative_position, 
                        INTERPOLATION_METHODS.get(supersampling, Image.NEAREST), 
                        PADDING_COLORS[padding_color]
                    )
                elif width != target_size or height != target_size:
                    interp_method = INTERPOLATION_METHODS.get(supersampling, Image.NEAREST)
                    img = img.resize((target_size, target_size), interp_method)

            # 优化：只在需要时进行数据类型转换，减少内存使用
            # 获取图像尺寸
            img_width, img_height = img.size
            
            # 转回tensor，保持原始批次大小
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # 处理通道数不匹配的情况(例如RGBA变成了RGB)
            if img_array.shape[-1] != channels:
                if img_array.shape[-1] == 4 and channels == 3:
                    # 从RGBA转换到RGB
                    img_array = img_array[..., :3]
                elif img_array.shape[-1] == 3 and channels == 4:
                    # 从RGB转换到RGBA
                    alpha = np.ones((*img_array.shape[:-1], 1), dtype=np.float32)
                    img_array = np.concatenate([img_array, alpha], axis=-1)
            
            # 创建新的批次张量
            output_image = torch.zeros((batch_size, img_height, img_width, img_array.shape[-1]), 
                                      dtype=torch.float32, device=image.device)
            
            # 将处理后的图像放入第一个批次位置
            output_image[0] = torch.from_numpy(img_array)
            
            # 如果有多个批次，复制到其他位置
            for i in range(1, batch_size):
                output_image[i] = output_image[0]

            return output_image, img_width, img_height

        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")
            
    def _process(self, **kwargs) -> Tuple[torch.Tensor, int, int]:
        """Process the node inputs and return outputs.
        
        Args:
            **kwargs: Input parameters for the node
            
        Returns:
            Tuple containing the scaled image, width, and height
        """
        return self.scale_image(
            kwargs["image"],
            kwargs["keep_proportion"],
            kwargs["pixels"],
            kwargs["scaling_factor"],
            kwargs["relative_position"],
            kwargs["supersampling"],
            kwargs["padding"],
            kwargs["padding_color"]
        ) 