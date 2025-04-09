"""Top-level package for comfyui_snacknodes."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """linksluckytime"""
__email__ = "linksluckytime@gmail.com"
__version__ = "0.0.1"

from .src.comfyui_snacknodes.nodes import NODE_CLASS_MAPPINGS
from .src.comfyui_snacknodes.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
