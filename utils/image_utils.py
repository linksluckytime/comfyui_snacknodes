"""Image utility functions for comfyui_snacknodes."""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Any

def calculate_dimensions(
    width: int,
    height: int,
    max_pixels: int,
    scaling_factor: int,
    keep_proportion: bool = True,
    min_size: int = 1
) -> Tuple[int, int]:
    """根据指定逻辑计算图像的新尺寸。
    
    逻辑流程:
    1. 计算输入图像比例
    2. 比较输入图像总像素与目标像素限制
    3. 根据不同情况计算新尺寸
    
    Args:
        width: 原始宽度
        height: 原始高度
        max_pixels: 最大像素数(面积)
        scaling_factor: 尺寸对齐因子
        keep_proportion: 是否保持图像比例
        min_size: 最小尺寸限制
        
    Returns:
        (new_width, new_height)元组
    """
    # 计算输入图像像素总数和比例
    input_pixels = width * height
    original_ratio = width / height if height > 0 else 1.0
    
    # 计算目标尺寸的基础值（正方形边长）
    target_size = int(np.sqrt(max_pixels))
    
    # 初始化新尺寸
    new_width, new_height = width, height
    
    # 情况1: 输入图像像素大于目标像素
    if input_pixels > max_pixels:
        if keep_proportion:
            # 保持比例缩小
            scale = np.sqrt(max_pixels / input_pixels)
            new_width = int(width * scale)
            new_height = int(height * scale)
        else:
            # 不保持比例，拉伸到目标正方形
            new_width = target_size
            new_height = target_size
    
    # 情况2: 输入图像像素小于目标像素
    elif input_pixels < max_pixels:
        if keep_proportion:
            # 保持比例放大，但不超过目标面积
            # 计算最大可能的放大比例
            max_scale = np.sqrt(max_pixels / input_pixels)
            # 保守放大，确保不会超过目标面积
            scale = max_scale * 0.99  # 使用99%防止舍入误差导致超出
            new_width = int(width * scale)
            new_height = int(height * scale)
        else:
            # 不保持比例，拉伸到目标正方形
            new_width = target_size
            new_height = target_size
    
    # 情况3: 输入图像像素等于目标像素
    else:  # input_pixels == max_pixels
        # 检查是否两边都能被缩放因子整除
        if width % scaling_factor == 0 and height % scaling_factor == 0:
            # 都能整除，直接返回原尺寸
            return width, height
        # 否则继续处理，会在下面的代码中进行调整
    
    # 确保尺寸不小于最小值
    new_width = max(new_width, min_size)
    new_height = max(new_height, min_size)
    
    # 调整到能被缩放因子整除
    if keep_proportion:
        # 保持比例的情况下，向下取整到缩放因子的倍数
        new_width_aligned = (new_width // scaling_factor) * scaling_factor
        new_height_aligned = (new_height // scaling_factor) * scaling_factor
        
        # 如果调整后尺寸太小，向上取整
        if new_width_aligned < min_size:
            new_width_aligned = ((new_width + scaling_factor - 1) // scaling_factor) * scaling_factor
        
        if new_height_aligned < min_size:
            new_height_aligned = ((new_height + scaling_factor - 1) // scaling_factor) * scaling_factor
        
        # 采用调整后的尺寸
        new_width, new_height = new_width_aligned, new_height_aligned
    else:
        # 不保持比例的情况，确保目标尺寸能被缩放因子整除
        new_width = (target_size // scaling_factor) * scaling_factor
        new_height = (target_size // scaling_factor) * scaling_factor
        
        # 如果调整后尺寸太小，使用最小缩放因子的倍数
        if new_width < min_size or new_height < min_size:
            new_width = max(min_size, scaling_factor)
            new_height = max(min_size, scaling_factor)
    
    return new_width, new_height

def crop_image(
    image: Image.Image,
    target_width: int,
    target_height: int,
    position: str = "center",
    min_size: int = 1
) -> Image.Image:
    """将图像裁剪到指定尺寸。
    
    Args:
        image: 输入的PIL图像
        target_width: 目标宽度
        target_height: 目标高度
        position: 裁剪位置（如"center"、"top left"等）
        min_size: 宽度和高度的最小尺寸
        
    Returns:
        裁剪后的PIL图像
    """
    # 确保目标尺寸不小于最小尺寸
    target_width = max(min_size, target_width)
    target_height = max(min_size, target_height)
    
    # 获取原图尺寸
    width, height = image.size
    
    # 如果原图尺寸与目标尺寸相同，直接返回
    if height == target_height and width == target_width:
        return image
    
    # 确保目标尺寸不大于原图尺寸，否则需要缩放
    if width < target_width or height < target_height:
        # 原图太小，需要先放大到至少能满足裁剪需求
        scale_ratio = max(target_width / width, target_height / height)
        new_width = int(width * scale_ratio)
        new_height = int(height * scale_ratio)
        image = image.resize((new_width, new_height), Image.LANCZOS)
        width, height = new_width, new_height
    
    # 确定裁剪的起始位置
    x_start = 0
    y_start = 0
    
    # 根据position确定裁剪位置
    position = position.lower()
    
    # 计算水平位置
    if "left" in position:
        x_start = 0
    elif "right" in position:
        x_start = width - target_width
    else:  # center
        x_start = (width - target_width) // 2
    
    # 计算垂直位置
    if "top" in position:
        y_start = 0
    elif "bottom" in position:
        y_start = height - target_height
    else:  # center
        y_start = (height - target_height) // 2
    
    # 确保裁剪区域不超出图像范围
    x_start = max(0, min(x_start, width - target_width))
    y_start = max(0, min(y_start, height - target_height))
    
    # 执行裁剪
    return image.crop((x_start, y_start, x_start + target_width, y_start + target_height))

def scale_with_padding(
    image: Image.Image,
    target_width: int,
    target_height: int,
    position: str = "center",
    interpolation: Any = Image.NEAREST,
    padding_color: Tuple[int, int, int, int] = (0, 0, 0, 0),
    min_size: int = 1
) -> Image.Image:
    """将图像等比例缩放并添加填充以保持画布尺寸。
    
    Args:
        image: 输入的PIL图像
        target_width: 目标宽度
        target_height: 目标高度
        position: 图像在画布上的位置
        interpolation: 插值方法（PIL常量）
        padding_color: 填充区域的RGBA颜色
        min_size: 宽度和高度的最小尺寸
        
    Returns:
        缩放和填充后的PIL图像
    """
    # 确保目标尺寸不小于最小尺寸
    target_width = max(min_size, target_width)
    target_height = max(min_size, target_height)
    
    # 获取原图尺寸
    width, height = image.size
    
    # 如果原图尺寸与目标尺寸相同，直接返回
    if height == target_height and width == target_width:
        return image
    
    # 计算等比例缩放比例
    # 选择最小的比例，以确保图像完全适应目标矩形
    ratio = min(target_width / width, target_height / height)
    
    # 计算缩放后的尺寸
    new_width = max(min_size, int(width * ratio))
    new_height = max(min_size, int(height * ratio))
    
    # 缩放图像
    resized_image = image.resize((new_width, new_height), interpolation)
    
    # 创建新画布，使用指定的填充颜色
    new_image = Image.new("RGBA", (target_width, target_height), padding_color)
    
    # 根据position确定粘贴位置
    position = position.lower()
    
    # 计算水平位置
    if "left" in position:
        x = 0
    elif "right" in position:
        x = target_width - new_width
    else:  # center
        x = (target_width - new_width) // 2
    
    # 计算垂直位置    
    if "top" in position:
        y = 0
    elif "bottom" in position:
        y = target_height - new_height
    else:  # center
        y = (target_height - new_height) // 2
    
    # 粘贴调整后的图像
    if resized_image.mode == "RGBA":
        # 使用透明通道正确粘贴RGBA图像
        new_image.paste(resized_image, (x, y), resized_image)
    else:
        # 对于RGB图像直接粘贴
        new_image.paste(resized_image, (x, y))
    
    return new_image 