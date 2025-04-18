"""Feature nodes for ComfyUI."""

from .text_box import TextBox
from .text_processor import TextProcessor

NODE_CLASS_MAPPINGS = {
    "TextBox": TextBox,
    "TextProcessor": TextProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextBox": "Text Box",
    "TextProcessor": "Text Processor",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 