"""ComfyUI SnackNodes package."""

__version__ = "0.0.5"

# 导入节点
from .nodes.image_info import ImageInfo
from .nodes.image_scaler import ImageScaler

NODE_CLASS_MAPPINGS = {
    "ImageInfo": ImageInfo,
    "ImageScaler": ImageScaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageInfo": "Image Info 🍿",
    "ImageScaler": "Image Scaler 🍿",
}

WEB_DIRECTORY = "web"

# 导出节点映射
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"] 