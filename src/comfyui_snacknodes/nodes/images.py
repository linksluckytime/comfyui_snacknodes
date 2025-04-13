"""Image processing nodes for ComfyUI."""

import numpy as np
import torch
import dlib
import cv2
from PIL import Image, ImageDraw
from typing import Tuple, List, Dict, Any, Optional
import os

from ..base_node import BaseNode

class ImageInfo(BaseNode):
    """Node that provides information about an image."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Return input types."""
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "batch_size", "channels")
    FUNCTION = "get_image_info"
    CATEGORY = "image"

    def get_image_info(self, image: torch.Tensor) -> Tuple[int, int, int, int]:
        """Get image information."""
        batch_size, height, width, channels = image.shape
        return (width, height, batch_size, channels)

class ImageScaler(BaseNode):
    """Node that scales an image."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Return input types."""
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 8}),
                "method": (["nearest", "bilinear", "bicubic", "lanczos"], {"default": "lanczos"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "scale_image"
    CATEGORY = "image"

    def scale_image(
        self, image: torch.Tensor, width: int, height: int, method: str
    ) -> torch.Tensor:
        """Scale image to specified dimensions."""
        try:
            # 获取批处理大小
            batch_size = image.shape[0]
            resized_images = []
            
            # 处理每张图片
            for i in range(batch_size):
                # 转换图像格式
                image_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
                
                # 处理 alpha 通道
                if image_np.shape[-1] == 4:
                    # 分离 alpha 通道
                    rgb = image_np[..., :3]
                    alpha = image_np[..., 3]
                    
                    # 分别缩放 RGB 和 alpha
                    rgb_pil = Image.fromarray(rgb)
                    alpha_pil = Image.fromarray(alpha)
                    
                    resized_rgb = rgb_pil.resize((width, height), {
                        "nearest": Image.NEAREST,
                        "bilinear": Image.BILINEAR,
                        "bicubic": Image.BICUBIC,
                        "lanczos": Image.LANCZOS
                    }[method])
                    
                    resized_alpha = alpha_pil.resize((width, height), Image.NEAREST)
                    
                    # 合并通道
                    resized_np = np.dstack((np.array(resized_rgb), np.array(resized_alpha)))
                else:
                    # 没有 alpha 通道的情况
                    image_pil = Image.fromarray(image_np)
                    resized_image = image_pil.resize((width, height), {
                        "nearest": Image.NEAREST,
                        "bilinear": Image.BILINEAR,
                        "bicubic": Image.BICUBIC,
                        "lanczos": Image.LANCZOS
                    }[method])
                    resized_np = np.array(resized_image)
                
                # 转换回 torch 张量
                resized_tensor = torch.from_numpy(resized_np).float() / 255.0
                resized_images.append(resized_tensor)
            
            # 合并所有处理后的图片
            return torch.stack(resized_images)
        except Exception as e:
            raise RuntimeError(f"Error scaling image: {str(e)}")

class FaceDetector(BaseNode):
    """Face detection node that detects faces and facial landmarks."""

    def __init__(self):
        super().__init__()
        self.detector = dlib.get_frontal_face_detector()
        
        # 检查模型文件是否存在
        comfyui_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        model_path = os.path.join(comfyui_path, "models", "dlib", "shape_predictor_68_face_landmarks.dat")
        
        if not os.path.exists(model_path):
            # 创建模型目录
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            raise FileNotFoundError(
                "Face detection model not found. Please download it from: "
                "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
                "Extract the file and place it in: ComfyUI/models/dlib/"
            )
        
        self.predictor = dlib.shape_predictor(model_path)
        
        # 面部特征点索引定义
        self.landmark_indices = {
            "left_eyebrow": list(range(17, 22)),
            "right_eyebrow": list(range(22, 27)),
            "left_eye": list(range(36, 42)),
            "right_eye": list(range(42, 48)),
            "nose": list(range(27, 36)),
            "mouth": list(range(48, 68)),
            "jaw": list(range(0, 17)),
            "left_cheek": [1, 2, 3, 4, 5, 48],
            "right_cheek": [13, 14, 15, 16, 17, 54]
        }

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Return input types."""
        return {
            "required": {
                "image": ("IMAGE",),
                "min_face_size": ("INT", {
                    "default": 64,
                    "min": 64,
                    "max": 1024,
                    "step": 8
                }),
                "upscale_method": (["nearest", "bilinear", "bicubic", "lanczos"], {
                    "default": "lanczos"
                }),
            },
            "optional": {
                "features": (["all", "left_eyebrow", "right_eyebrow", "left_eye", 
                            "right_eye", "nose", "mouth", "jaw", "left_cheek", 
                            "right_cheek"], {
                    "default": "all"
                }),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("mask", "image", "controlnet_points")
    FUNCTION = "detect_faces"
    CATEGORY = "image/face"

    def detect_faces(
        self, 
        image: torch.Tensor,
        min_face_size: int,
        upscale_method: str,
        features: str = "all"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Detect faces and return masks and control points."""
        try:
            # 转换图像格式
            image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
            # 转换为 BGR 格式（OpenCV 使用）
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # 检测人脸，使用 min_face_size 参数
            faces = self.detector(image_bgr, 1)
            
            # 创建蒙版和控制点图像
            mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
            control_points = np.zeros_like(image_np)
            
            for face in faces:
                # 检查人脸大小是否满足最小要求
                face_width = face.right() - face.left()
                face_height = face.bottom() - face.top()
                if face_width < min_face_size or face_height < min_face_size:
                    continue
                    
                # 获取面部特征点
                landmarks = self.predictor(image_bgr, face)
                
                # 根据选择的特征更新蒙版
                if features == "all":
                    indices = list(range(68))
                else:
                    indices = self.landmark_indices[features]
                
                # 绘制蒙版
                points = [(landmarks.part(i).x, landmarks.part(i).y) for i in indices]
                cv2.fillPoly(mask, [np.array(points)], 255)
                
                # 绘制控制点
                for i in indices:
                    point = (landmarks.part(i).x, landmarks.part(i).y)
                    cv2.circle(control_points, point, 2, (255, 255, 255), -1)
            
            # 转换回 torch 张量
            mask_tensor = torch.from_numpy(mask).float() / 255.0
            control_points_tensor = torch.from_numpy(control_points).float() / 255.0
            
            # 保持原始图像尺寸
            return (mask_tensor.unsqueeze(0), image, control_points_tensor.unsqueeze(0))
        except Exception as e:
            raise RuntimeError(f"Error detecting faces: {str(e)}") 