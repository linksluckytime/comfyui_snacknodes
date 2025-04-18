"""ImageScaler node for ComfyUI that provides image scaling and transformation capabilities."""

import torch
import numpy as np
import math
from PIL import Image
from typing import Dict, Tuple, Optional, Any
from ..base_node import BaseNode
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

class ImageScaler(BaseNode):
    """A node for scaling and transforming images with various options."""
    
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "scale_image"
    OUTPUT_NODE = True  # 添加此属性以在节点上显示输出信息
    CATEGORY = "SnackNodes"  # 修改分类到根目录

    # 最小合理尺寸，避免创建过小的图像
    MIN_REASONABLE_SIZE = 32  # 设置最小尺寸为32x32像素
    # 是否启用调试输出
    ENABLE_DEBUG = True

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        """Define the input types for this node.
        
        Returns:
            Dictionary containing input type definitions
        """
        return {
            "required": {
                "image": ("IMAGE", {"description": "输入图像 (B,H,W,C 格式)"}),
                "keep_proportion": ("BOOLEAN", {
                    "default": True,
                    "description": "等比缩放：保持原始图像比例"
                }),
                "pixels": (list(PIXEL_BUDGETS.keys()), {
                    "default": "1024px²",
                    "description": "像素总量：以1:1矩形为标准的目标像素数"
                }),
                "scaling_factor": (SCALING_FACTORS, {
                    "default": 64,
                    "description": "缩放因子：确保输出尺寸能被此数整除"
                }),
                "relative_position": (RELATIVE_POSITIONS, {
                    "default": "center",
                    "description": "原图参考位置：图像在输出画布中的位置(九宫格)"
                }),
                "supersampling": (list(INTERPOLATION_METHODS.keys()), {
                    "default": "none",
                    "description": "超级采样：图像插值方法"
                }),
                "padding": ("BOOLEAN", {
                    "default": False,
                    "description": "填充：是否在保持比例时添加填充"
                }),
                "padding_method": (list(PADDING_COLORS.keys()), {
                    "default": "transparent",
                    "description": "填充方式：填充区域的处理方式"
                }),
            },
        }

    def debug_log(self, *args, **kwargs):
        """Output debug information"""
        if self.ENABLE_DEBUG:
            print("[ImageScaler]", *args, **kwargs)

    def scale_image(self, image: torch.Tensor, keep_proportion: bool, pixels: str, 
                   scaling_factor: int, relative_position: str, supersampling: str, 
                   padding: bool, padding_method: str) -> Tuple[torch.Tensor, int, int]:
        """Scale and transform image according to specified logic."""
        # Validate input
        if len(image.shape) != 4:
            raise ValueError("Input tensor must be 4-dimensional (B,H,W,C)")
            
        # Get pixel budget value
        pixel_budget = PIXEL_BUDGETS.get(pixels)
        if not pixel_budget:
            raise ValueError(f"Invalid pixel budget: {pixels}")

        try:
            # Extract image information
            batch_size, height, width, channels = image.shape
            input_pixels = width * height
            aspect_ratio = width / height
            
            # Get device information
            device = image.device
            
            # Log initial information
            self.debug_log(f"Input image size: {width}x{height}, Total pixels: {input_pixels}")
            self.debug_log(f"Image ratio: {aspect_ratio:.3f}")
            self.debug_log(f"Target pixel budget: {int(math.sqrt(pixel_budget))}x{int(math.sqrt(pixel_budget))} ({pixel_budget} pixels)")

            # Calculate target dimensions
            if keep_proportion:
                # Maintain original aspect ratio
                if input_pixels > pixel_budget:
                    # Need to downscale
                    new_width = min(width, int(math.sqrt(pixel_budget * aspect_ratio)))
                    new_height = min(height, int(math.sqrt(pixel_budget / aspect_ratio)))
                else:
                    # Need to upscale
                    scale = math.sqrt(pixel_budget / input_pixels) * 0.99  # Slightly conservative
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                
                # Ensure dimensions are not smaller than minimum
                new_width = max(self.MIN_REASONABLE_SIZE, new_width)
                new_height = max(self.MIN_REASONABLE_SIZE, new_height)
                
                # Generate multiple candidate sizes (round up and down to scaling factor)
                candidates = []
                
                # Round up sizes
                w1 = ((new_width + scaling_factor - 1) // scaling_factor) * scaling_factor
                h1 = ((new_height + scaling_factor - 1) // scaling_factor) * scaling_factor
                
                # Round down sizes
                w2 = (new_width // scaling_factor) * scaling_factor
                h2 = (new_height // scaling_factor) * scaling_factor
                
                # Generate all possible combinations
                for w in [w1, w2]:
                    if w < self.MIN_REASONABLE_SIZE:
                        continue
                    for h in [h1, h2]:
                        if h < self.MIN_REASONABLE_SIZE:
                            continue
                        # Calculate pixel count and ratio deviation
                        pixels = w * h
                        if pixels > pixel_budget:
                            continue
                        ratio = w / h
                        ratio_diff = abs(ratio - aspect_ratio) / aspect_ratio
                        candidates.append((w, h, ratio_diff, pixels))
                
                if candidates:
                    # Sort by ratio deviation first
                    candidates.sort(key=lambda x: (x[2], -x[3]))  # Prioritize smaller ratio deviation, then larger pixel count
                    new_width, new_height, ratio_diff, pixels = candidates[0]
                    self.debug_log(f"Selected best size option: {new_width}x{new_height} (Ratio deviation: {ratio_diff:.1%}, Pixels: {pixels})")
                else:
                    # If no suitable candidates, use conservative round down
                    new_width = max(self.MIN_REASONABLE_SIZE, (new_width // scaling_factor) * scaling_factor)
                    new_height = max(self.MIN_REASONABLE_SIZE, (new_height // scaling_factor) * scaling_factor)
            else:
                # Don't maintain aspect ratio, use square
                square_size = int(math.sqrt(pixel_budget))
                square_size = ((square_size + scaling_factor - 1) // scaling_factor) * scaling_factor
                new_width = new_height = square_size

            # Ensure not exceeding pixel budget
            while new_width * new_height > pixel_budget:
                if new_width >= new_height and new_width > scaling_factor:
                    new_width -= scaling_factor
                elif new_height > scaling_factor:
                    new_height -= scaling_factor
                else:
                    break

            # Finally ensure dimensions are not smaller than minimum
            new_width = max(self.MIN_REASONABLE_SIZE, new_width)
            new_height = max(self.MIN_REASONABLE_SIZE, new_height)

            # Log calculation results
            self.debug_log(f"Calculated target size: {new_width}x{new_height}")
            self.debug_log(f"Target pixel count: {new_width * new_height}")
            if keep_proportion:
                final_ratio = new_width / new_height
                ratio_diff = abs(final_ratio - aspect_ratio) / aspect_ratio
                self.debug_log(f"Final ratio: {final_ratio:.3f} (Deviation: {ratio_diff:.1%})")

            # Prepare image processing
            first_image = image[0].cpu().numpy()
            img_data = (first_image * 255).astype(np.uint8)
            
            # Ensure correct number of channels
            if channels == 3:
                img = Image.fromarray(img_data, 'RGB')
            else:  # channels == 4
                img = Image.fromarray(img_data, 'RGBA')
            
            # Get interpolation method
            interp_method = INTERPOLATION_METHODS.get(supersampling, Image.NEAREST)

            # Perform scaling
            if padding and keep_proportion:
                # Use padding
                pad_color = PADDING_COLORS.get(padding_method, (0, 0, 0, 0))
                
                # If transparent padding, ensure output is RGBA mode
                if padding_method == "transparent":
                    # If original image has no alpha channel, add it
                    if img.mode != 'RGBA':
                        rgb_img = np.array(img)
                        alpha = np.ones((*rgb_img.shape[:-1], 1), dtype=np.uint8) * 255
                        img_rgba = np.concatenate([rgb_img, alpha], axis=-1)
                        img = Image.fromarray(img_rgba, 'RGBA')
                
                # Use padding scaling function
                img_resized = scale_with_padding(
                    image=img,
                    target_width=new_width,
                    target_height=new_height,
                    position=relative_position,
                    interpolation=interp_method,
                    padding_color=pad_color,
                    min_size=self.MIN_REASONABLE_SIZE
                )
            else:
                # Direct resize
                img_resized = img.resize((new_width, new_height), interp_method)

            # Get final dimensions
            output_width, output_height = img_resized.size
            self.debug_log(f"Final output size: {output_width}x{output_height}")

            # Convert back to tensor
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            
            # Handle channel count
            output_channels = img_array.shape[-1]
            self.debug_log(f"Output image channels: {output_channels}")
            
            # Ensure output channels match input or handle transparency properly
            if output_channels != channels:
                if padding_method == "transparent":
                    # If using transparent padding, our output should be 4 channels
                    if output_channels == 4 and channels == 3:
                        # If original image had no alpha channel, update channel count
                        channels = 4
                    elif output_channels == 3 and channels == 4:
                        # Should not happen: transparent padding but output only has 3 channels
                        alpha = np.ones((*img_array.shape[:-1], 1), dtype=np.float32)
                        img_array = np.concatenate([img_array, alpha], axis=-1)
                else:
                    # Non-transparent padding, try to match input channel count
                    if output_channels == 4 and channels == 3:
                        img_array = img_array[..., :3]
                    elif output_channels == 3 and channels == 4:
                        alpha = np.ones((*img_array.shape[:-1], 1), dtype=np.float32)
                        img_array = np.concatenate([img_array, alpha], axis=-1)

            # Create output tensor
            channels = img_array.shape[-1]  # Use final channel count
            output_image = torch.zeros((batch_size, output_height, output_width, channels), 
                                    dtype=torch.float32, device=device)
            output_image[0] = torch.from_numpy(img_array).to(device)
            
            # Copy to other batches
            if batch_size > 1:
                for i in range(1, batch_size):
                    output_image[i] = output_image[0]

            return output_image, output_width, output_height

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f"Error processing image: {str(e)}") 