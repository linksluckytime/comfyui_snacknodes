"""ComfyUI SnackNodes package."""

from .nodes.image_info import ImageInfo
from .nodes.image_scaler import ImageScaler

NODE_CLASS_MAPPINGS = {
    "ImageInfo": ImageInfo,
    "ImageScaler": ImageScaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageInfo": "Image Info",
    "ImageScaler": "Image Scaler",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
