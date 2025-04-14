"""Base node class for comfyui_snacknodes."""

from typing import Dict, Any, Tuple

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
    
    def _process(self, **kwargs) -> Tuple:
        """Process the node's inputs and return outputs.
        
        This method should be overridden by subclasses to implement
        the node's functionality.
        
        Args:
            **kwargs: Input parameters for the node
            
        Returns:
            A tuple containing the node's outputs
        """
        func_name = getattr(self, "FUNCTION", "")
        if func_name and hasattr(self, func_name):
            return getattr(self, func_name)(**kwargs)
        return () 