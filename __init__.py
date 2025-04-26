"""ComfyUI SnackNodes package."""

# Import nodes
from .src.nodes.images.image_info import ImageInfo
from .src.nodes.images.image_scaler import ImageScaler
from .src.nodes.feature.text_box import TextBox
from .src.nodes.feature.text_processor import TextProcessor

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

WEB_DIRECTORY = "src/web"

# Export node mappings
__all__ = ["ImageInfo", "ImageScaler", "TextBox", "TextProcessor"]