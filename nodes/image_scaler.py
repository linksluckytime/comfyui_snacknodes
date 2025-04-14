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
        """输出调试信息"""
        if self.ENABLE_DEBUG:
            print("[ImageScaler]", *args, **kwargs)

    def scale_image(self, image: torch.Tensor, keep_proportion: bool, pixels: str, 
                   scaling_factor: int, relative_position: str, supersampling: str, 
                   padding: bool, padding_method: str) -> Tuple[torch.Tensor, int, int]:
        """按照指定逻辑对图像进行缩放处理。"""
        # 验证输入
        if len(image.shape) != 4:
            raise ValueError("输入张量必须是4维的 (B,H,W,C)")
            
        # 获取像素预算值
        pixel_budget = PIXEL_BUDGETS.get(pixels)
        if not pixel_budget:
            raise ValueError(f"无效的像素预算: {pixels}")

        try:
            # 提取图像信息
            batch_size, height, width, channels = image.shape
            input_pixels = width * height
            aspect_ratio = width / height
            
            # 获取设备信息
            device = image.device
            
            # 记录初始信息
            self.debug_log(f"输入图像尺寸: {width}x{height}, 像素总量: {input_pixels}")
            self.debug_log(f"图像比例: {aspect_ratio:.3f}")
            self.debug_log(f"目标像素预算: {int(math.sqrt(pixel_budget))}x{int(math.sqrt(pixel_budget))} ({pixel_budget}像素)")

            # 计算目标尺寸
            if keep_proportion:
                # 保持原始比例
                if input_pixels > pixel_budget:
                    # 需要缩小
                    new_width = min(width, int(math.sqrt(pixel_budget * aspect_ratio)))
                    new_height = min(height, int(math.sqrt(pixel_budget / aspect_ratio)))
                else:
                    # 需要放大
                    scale = math.sqrt(pixel_budget / input_pixels) * 0.99  # 稍微保守一点
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                
                # 确保尺寸不小于最小值
                new_width = max(self.MIN_REASONABLE_SIZE, new_width)
                new_height = max(self.MIN_REASONABLE_SIZE, new_height)
                
                # 生成多个候选尺寸（向上和向下取整到缩放因子的倍数）
                candidates = []
                
                # 向上取整的尺寸
                w1 = ((new_width + scaling_factor - 1) // scaling_factor) * scaling_factor
                h1 = ((new_height + scaling_factor - 1) // scaling_factor) * scaling_factor
                
                # 向下取整的尺寸
                w2 = (new_width // scaling_factor) * scaling_factor
                h2 = (new_height // scaling_factor) * scaling_factor
                
                # 生成所有可能的组合
                for w in [w1, w2]:
                    if w < self.MIN_REASONABLE_SIZE:
                        continue
                    for h in [h1, h2]:
                        if h < self.MIN_REASONABLE_SIZE:
                            continue
                        # 计算像素数和比例偏差
                        pixels = w * h
                        if pixels > pixel_budget:
                            continue
                        ratio = w / h
                        ratio_diff = abs(ratio - aspect_ratio) / aspect_ratio
                        candidates.append((w, h, ratio_diff, pixels))
                
                if candidates:
                    # 首先按比例偏差排序
                    candidates.sort(key=lambda x: (x[2], -x[3]))  # 优先考虑比例偏差小的，其次是像素数大的
                    new_width, new_height, ratio_diff, pixels = candidates[0]
                    self.debug_log(f"选择最佳尺寸方案: {new_width}x{new_height} (比例偏差: {ratio_diff:.1%}, 像素数: {pixels})")
                else:
                    # 如果没有合适的候选项，使用保守的向下取整
                    new_width = max(self.MIN_REASONABLE_SIZE, (new_width // scaling_factor) * scaling_factor)
                    new_height = max(self.MIN_REASONABLE_SIZE, (new_height // scaling_factor) * scaling_factor)
            else:
                # 不保持比例，使用正方形
                square_size = int(math.sqrt(pixel_budget))
                square_size = ((square_size + scaling_factor - 1) // scaling_factor) * scaling_factor
                new_width = new_height = square_size

            # 确保不超过像素预算
            while new_width * new_height > pixel_budget:
                if new_width >= new_height and new_width > scaling_factor:
                    new_width -= scaling_factor
                elif new_height > scaling_factor:
                    new_height -= scaling_factor
                else:
                    break

            # 最后确保尺寸不小于最小值
            new_width = max(self.MIN_REASONABLE_SIZE, new_width)
            new_height = max(self.MIN_REASONABLE_SIZE, new_height)

            # 记录计算结果
            self.debug_log(f"计算的目标尺寸: {new_width}x{new_height}")
            self.debug_log(f"目标像素数: {new_width * new_height}")
            if keep_proportion:
                final_ratio = new_width / new_height
                ratio_diff = abs(final_ratio - aspect_ratio) / aspect_ratio
                self.debug_log(f"最终比例: {final_ratio:.3f} (偏差: {ratio_diff:.1%})")

            # 准备图像处理
            first_image = image[0].cpu().numpy()
            img_data = (first_image * 255).astype(np.uint8)
            
            # 确保图像有正确的通道数
            if channels == 3:
                img = Image.fromarray(img_data, 'RGB')
            else:  # channels == 4
                img = Image.fromarray(img_data, 'RGBA')
            
            # 获取插值方法
            interp_method = INTERPOLATION_METHODS.get(supersampling, Image.NEAREST)

            # 执行缩放
            if padding and keep_proportion:
                # 使用填充
                pad_color = PADDING_COLORS.get(padding_method, (0, 0, 0, 0))
                
                # 如果是透明填充，确保输出图像是RGBA模式
                if padding_method == "transparent":
                    # 如果原图没有透明通道，添加透明通道
                    if img.mode != 'RGBA':
                        rgb_img = np.array(img)
                        alpha = np.ones((*rgb_img.shape[:-1], 1), dtype=np.uint8) * 255
                        img_rgba = np.concatenate([rgb_img, alpha], axis=-1)
                        img = Image.fromarray(img_rgba, 'RGBA')
                
                # 使用填充缩放函数
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
                # 直接调整大小
                img_resized = img.resize((new_width, new_height), interp_method)

            # 获取最终尺寸
            output_width, output_height = img_resized.size
            self.debug_log(f"最终输出尺寸: {output_width}x{output_height}")

            # 转换回张量
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            
            # 处理通道数
            output_channels = img_array.shape[-1]
            self.debug_log(f"输出图像通道数: {output_channels}")
            
            # 确保输出通道数与输入匹配或正确处理透明通道
            if output_channels != channels:
                if padding_method == "transparent":
                    # 如果使用透明填充，我们的输出应该是4通道
                    if output_channels == 4 and channels == 3:
                        # 如果原始图像没有透明通道，但现在有了，需要更新通道数
                        channels = 4
                    elif output_channels == 3 and channels == 4:
                        # 应该不会发生：透明填充但输出只有3通道
                        alpha = np.ones((*img_array.shape[:-1], 1), dtype=np.float32)
                        img_array = np.concatenate([img_array, alpha], axis=-1)
                else:
                    # 非透明填充，尝试匹配输入通道数
                    if output_channels == 4 and channels == 3:
                        img_array = img_array[..., :3]
                    elif output_channels == 3 and channels == 4:
                        alpha = np.ones((*img_array.shape[:-1], 1), dtype=np.float32)
                        img_array = np.concatenate([img_array, alpha], axis=-1)

            # 创建输出张量
            channels = img_array.shape[-1]  # 使用最终的通道数
            output_image = torch.zeros((batch_size, output_height, output_width, channels), 
                                    dtype=torch.float32, device=device)
            output_image[0] = torch.from_numpy(img_array).to(device)
            
            # 复制到其他批次
            if batch_size > 1:
                for i in range(1, batch_size):
                    output_image[i] = output_image[0]

            return output_image, output_width, output_height

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f"处理图像时出错: {str(e)}") 