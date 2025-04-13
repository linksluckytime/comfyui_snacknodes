"""Base node class for comfyui_snacknodes."""

from typing import Dict, Any

class BaseNode:
    """Base class for all nodes in comfyui_snacknodes."""
    
    CATEGORY = "SnackNodes"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = ""
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define the input types for this node.
        
        Returns:
            Dictionary containing input type definitions
        """
        return {} 