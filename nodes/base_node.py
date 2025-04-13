"""Base node class for ComfyUI SnackNodes."""

from typing import Dict, Any, Tuple
import torch

class BaseNode:
    """Base class for all nodes in the package."""
    
    CATEGORY = "SnackNodes"
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define the input types for the node.
        
        Returns:
            Dictionary containing input type definitions
        """
        return {}
    
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = ""
    
    def __init__(self):
        """Initialize the node."""
        pass
    
    def process(self, **kwargs) -> Tuple:
        """Process the node's inputs and return outputs.
        
        Args:
            **kwargs: Input parameters for the node
            
        Returns:
            Tuple containing the output values
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If processing fails
        """
        try:
            self._validate_inputs(**kwargs)
            return self._process(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Error processing node {self.__class__.__name__}: {str(e)}")
    
    def _validate_inputs(self, **kwargs) -> None:
        """Validate the inputs for this node.
        
        Args:
            **kwargs: Input parameters to validate
            
        Raises:
            ValueError: If validation fails
        """
        is_valid, error_message = self.VALIDATE_INPUTS(**kwargs)
        if not is_valid:
            raise ValueError(error_message)
    
    def _process(self, **kwargs) -> Tuple:
        """Internal processing method to be implemented by subclasses.
        
        Args:
            **kwargs: Input parameters for the node
            
        Returns:
            Tuple containing the output values
        """
        raise NotImplementedError("Subclasses must implement _process()")
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs) -> Tuple[bool, str]:
        """Validate the inputs for this node.
        
        Args:
            **kwargs: Input parameters to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        return True, ""
    
    @classmethod
    def IS_CHANGED(cls, **kwargs) -> Any:
        """Determine if the node's output would change given the inputs.
        
        Args:
            **kwargs: Input parameters
            
        Returns:
            Any value that can be used to determine if the output would change
        """
        return float("NaN") 