"""Node definitions for comfyui_snacknodes."""

from .images.image_scaler import ImageScaler
from .images.image_info import ImageInfo

NODE_CLASS_MAPPINGS = {
    "ImageScaler": ImageScaler,
    "ImageInfo": ImageInfo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageScaler": "Image Scaler 🍿",
    "ImageInfo": "Image Info 🍿"
}

__all__ = ["ImageInfo", "ImageScaler"] 