"""Text Box node for ComfyUI."""

from typing import Dict, Tuple
from ..base_node import BaseNode

class TextBox(BaseNode):
    """A node for handling multiline text input."""
    
    CATEGORY = "SnackNodes"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "get_text"
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        """Define the input types for this node."""
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your text here..."
                }),
            }
        }
    
    def get_text(self, text: str) -> Tuple[str]:
        """Get the input text.
        
        Args:
            text: The input text
            
        Returns:
            Tuple containing the input text
        """
        return (text,) 