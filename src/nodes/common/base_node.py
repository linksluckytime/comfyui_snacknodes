"""Base node class for ComfyUI nodes."""

import torch
from typing import Dict, Any, List, Tuple, Optional, Union, TypeVar
import numpy as np
from PIL import Image
import os

T = TypeVar('T', bound='BaseNode')

class BaseNode:
    """Base node class for ComfyUI nodes.
    
    This class provides common functionality for all ComfyUI nodes, including:
    - Input/output type definitions
    - Image/tensor conversion utilities
    - Model path resolution
    """
    
    # Class attributes
    RETURN_TYPES: Tuple = ()
    RETURN_NAMES: Tuple = ()
    FUNCTION: str = "process"
    CATEGORY: str = "SnackNodes"
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """Define node input types.
        
        Returns:
            Dictionary containing required and optional input types:
            {
                "required": {param_name: {"type": type, ...}},
                "optional": {param_name: {"type": type, ...}}
            }
        """
        return {
            "required": {},
            "optional": {}
        }
    
    def process(self, **kwargs) -> Tuple:
        """Process function to be implemented by subclasses.
        
        Args:
            **kwargs: Input parameters as defined in INPUT_TYPES
            
        Returns:
            Tuple of output values matching RETURN_TYPES
            
        Raises:
            NotImplementedError: If subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement process method")
    
    @staticmethod
    def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL image.
        
        Args:
            tensor: Input tensor in (B,H,W,C) or (H,W,C) format,
                   with values in range [0, 1]
            
        Returns:
            PIL Image object in RGB or RGBA format
            
        Raises:
            ValueError: If tensor shape or values are invalid
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
            
        if tensor.dim() not in [3, 4]:
            raise ValueError("Tensor must be 3D (H,W,C) or 4D (B,H,W,C)")
            
        if tensor.dim() == 4:
            if tensor.size(0) != 1:
                raise ValueError("Batch size must be 1 for 4D tensors")
            tensor = tensor[0]
            
        if tensor.dim() == 3:
            if tensor.size(2) not in [3, 4]:
                raise ValueError("Channel dimension must be 3 (RGB) or 4 (RGBA)")
            tensor = tensor.permute(1, 2, 0)
            
        if tensor.min() < 0 or tensor.max() > 1:
            raise ValueError("Tensor values must be in range [0, 1]")
            
        tensor = tensor.cpu().numpy()
        tensor = (tensor * 255).astype(np.uint8)
        return Image.fromarray(tensor)
    
    @staticmethod
    def image_to_tensor(image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor.
        
        Args:
            image: PIL Image object in RGB or RGBA format
            
        Returns:
            Tensor in (B,C,H,W) format with values in range [0, 1]
            
        Raises:
            TypeError: If input is not a PIL Image
            ValueError: If image mode is not supported
        """
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL Image")
            
        if image.mode not in ["RGB", "RGBA"]:
            raise ValueError("Image must be in RGB or RGBA format")
            
        tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
        if tensor.dim() == 3:
            if tensor.size(2) not in [3, 4]:
                raise ValueError("Channel dimension must be 3 (RGB) or 4 (RGBA)")
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
            ValueError: If input parameters are invalid
            FileNotFoundError: If model file or directory does not exist
        """
        if not model_name or not isinstance(model_name, str):
            raise ValueError("Model name must be a non-empty string")
            
        if not model_dir or not isinstance(model_dir, str):
            raise ValueError("Model directory must be a non-empty string")
            
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
        model_path = os.path.join(model_dir, model_name)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        return model_path 