"""Base node class for ComfyUI nodes."""

import torch
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
from PIL import Image
import os

class BaseNode:
    """Base node class for ComfyUI nodes."""
    
    # Class attributes
    RETURN_TYPES: Tuple = ()
    RETURN_NAMES: Tuple = ()
    FUNCTION: str = "process"
    CATEGORY: str = "SnackNodes"
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """Define node input types.
        
        Returns:
            Dictionary containing required and optional input types
        """
        return {
            "required": {},
            "optional": {}
        }
    
    def process(self, **kwargs) -> Tuple:
        """Process function to be implemented by subclasses.
        
        Args:
            **kwargs: Input parameters
            
        Returns:
            Tuple of output values
            
        Raises:
            NotImplementedError: If subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement process method")
    
    @staticmethod
    def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL image.
        
        Args:
            tensor: Input tensor (B,H,W,C) or (H,W,C)
            
        Returns:
            PIL Image object
            
        Raises:
            ValueError: If tensor shape is invalid
        """
        if tensor.dim() not in [3, 4]:
            raise ValueError("Tensor must be 3D (H,W,C) or 4D (B,H,W,C)")
            
        if tensor.dim() == 4:
            tensor = tensor[0]
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
            
        tensor = tensor.cpu().numpy()
        tensor = (tensor * 255).astype(np.uint8)
        return Image.fromarray(tensor)
    
    @staticmethod
    def image_to_tensor(image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tensor in (B,C,H,W) format
        """
        tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
        if tensor.dim() == 3:
            tensor = tensor.permute(2, 0, 1)
        return tensor.unsqueeze(0)
    
    @staticmethod
    def get_model_path(model_name: str, model_dir: str) -> str:
        """Get model file path.
        
        Args:
            model_name: Name of the model file
            model_dir: Directory containing model files
            
        Returns:
            Full path to model file
            
        Raises:
            FileNotFoundError: If model file does not exist
        """
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return model_path 