"""ComfyUI SnackNodes nodes package."""

from .images import ImageInfo, ImageScaler, FaceDetector

NODE_CLASS_MAPPINGS = {
    "ImageInfo": ImageInfo,
    "ImageScaler": ImageScaler,
    "FaceDetector": FaceDetector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageInfo": "Image Info 🍿",
    "ImageScaler": "Image Scaler 🍿",
    "FaceDetector": "Face Detector 🍿",
} 