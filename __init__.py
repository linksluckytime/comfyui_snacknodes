"""ComfyUI SnackNodes package."""

# Import nodes
from .nodes.images.image_info import ImageInfo
from .nodes.images.image_scaler import ImageScaler
from .nodes.feature.text_box import TextBox
from .nodes.feature.text_processor import TextProcessor

NODE_CLASS_MAPPINGS = {
    "ImageInfo": ImageInfo,
    "ImageScaler": ImageScaler,
    "TextBox": TextBox,
    "TextProcessor": TextProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageInfo": "Image Info 🍿",
    "ImageScaler": "Image Scaler 🍿",
    "TextBox": "Text Box 🍿",
    "TextProcessor": "Text Processor 🍿",
}

WEB_DIRECTORY = "web"

# Export node mappings
__all__ = ["ImageInfo", "ImageScaler", "TextBox", "TextProcessor"]