"""ComfyUI SnackNodes package."""

__version__ = "0.0.3"

from .nodes.images import ImageInfo, ImageScaler, FaceDetector

NODE_CLASS_MAPPINGS = {
    "ImageInfo": ImageInfo,
    "ImageScaler": ImageScaler,
    "FaceDetector": FaceDetector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageInfo": "🍿 Image Info",
    "ImageScaler": "🍿 Image Scaler",
    "FaceDetector": "🍿 Face Detector",
}

WEB_DIRECTORY = "web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"] 