import os
import torch
import math
import numpy as np
from PIL import Image, ImageOps
from typing import Tuple, Dict, Optional


class ImageInfo:
    """A node that provides information about an input image tensor."""
    CATEGORY = "SnackNodes"

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "required": {
                "image": ("IMAGE", {"description": "Input image tensor (B,H,W,C format)"}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("batch_size", "height", "width", "channels")
    FUNCTION = "get_image_info"

    def get_image_info(self, image: torch.Tensor) -> Tuple[int, int, int, int]:
        """Extract and return image dimensions.
        
        Args:
            image: Input image tensor in B,H,W,C format
            
        Returns:
            Tuple containing batch_size, height, width, and channels
            
        Raises:
            ValueError: If input tensor is not 4-dimensional
        """
        if len(image.shape) != 4:
            raise ValueError("Input tensor must be 4-dimensional (B,H,W,C)")
        return image.shape


class ImageScaler:
    """A node for scaling and transforming images with various options."""
    CATEGORY = "SnackNodes"

    # Cache for padding colors and interpolation methods
    _PADDING_COLORS = {
        "Transparent": (0, 0, 0, 0),
        "Gray": (128, 128, 128),
        "Black": (0, 0, 0),
        "White": (255, 255, 255),
    }

    _INTERPOLATION_METHODS = {
        "Nearest": Image.NEAREST,
        "Box": Image.BOX,
        "Bilinear": Image.BILINEAR,
        "Hamming": Image.HAMMING,
        "Bicubic": Image.BICUBIC,
        "Lanczos": Image.LANCZOS,
        "None": Image.NEAREST,
    }

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "required": {
                "image": ("IMAGE", {"description": "Input image tensor (B,H,W,C format)"}),
                "KeepProportion": ("BOOLEAN", {
                    "default": True,
                    "description": "Maintain image aspect ratio during scaling"
                }),
                "Pixels": (["512px²", "1024px²"], {
                    "default": "1024px²",
                    "description": "Maximum number of pixels in the output image"
                }),
                "ScalingFactor": ([2, 8, 32, 64], {
                    "default": 64,
                    "description": "Factor to align dimensions to"
                }),
                "RelativePosition": ([
                    "Center", "Top-Left", "Top-Center", "Top-Right",
                    "Middle-Left", "Middle-Center", "Middle-Right",
                    "Bottom-Left", "Bottom-Center", "Bottom-Right"
                ], {
                    "default": "Center",
                    "description": "Position of the image relative to the output canvas"
                }),
                "Supersampling": ([
                    "None", "Lanczos", "Nearest", "Bilinear", "Bicubic", "Box", "Hamming"
                ], {
                    "default": "None",
                    "description": "Interpolation method for scaling"
                }),
                "Padding": ("BOOLEAN", {
                    "default": False,
                    "description": "Add padding to maintain aspect ratio"
                }),
                "PaddingElements": (["Transparent", "Gray", "Black", "White"], {
                    "default": "Transparent",
                    "description": "Color of the padding"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "scale_image"

    def scale_image(self, image: torch.Tensor, KeepProportion: bool, Pixels: str, 
                   ScalingFactor: int, RelativePosition: str, Supersampling: str, 
                   Padding: bool, PaddingElements: str) -> Tuple[torch.Tensor, int, int]:
        """Scale and transform an input image according to specified parameters.
        
        Args:
            image: Input image tensor
            KeepProportion: Whether to maintain aspect ratio
            Pixels: Maximum pixel budget
            ScalingFactor: Factor to align dimensions to
            RelativePosition: Position of image in output
            Supersampling: Interpolation method
            Padding: Whether to add padding
            PaddingElements: Color of padding
            
        Returns:
            Tuple containing scaled image tensor and its dimensions
            
        Raises:
            ValueError: If input tensor is invalid or parameters are invalid
        """
        # Validate inputs
        if len(image.shape) != 4:
            raise ValueError("Input tensor must be 4-dimensional (B,H,W,C)")
            
        max_pixels = {"512px²": 512 ** 2, "1024px²": 1024 ** 2}.get(Pixels)
        if not max_pixels:
            raise ValueError(f"Invalid pixel budget: {Pixels}")

        try:
            # Extract image dimensions and convert to PIL Image
            batch_size, height, width, channels = image.shape
            img = Image.fromarray((image[0].numpy() * 255).astype("uint8"))

            # Ensure image is in RGBA mode for transparent padding
            if PaddingElements == "Transparent":
                img = img.convert("RGBA")

            # Calculate target dimensions
            if KeepProportion:
                new_width, new_height = self.calculate_dimensions(width, height, max_pixels, ScalingFactor)
                img = img.resize((new_width, new_height), self._INTERPOLATION_METHODS[Supersampling])
            else:
                target_size = math.isqrt(max_pixels)
                target_size = (target_size // ScalingFactor) * ScalingFactor

                can_fit = (width <= target_size and height <= target_size) or \
                         (width * target_size / height <= target_size and height * target_size / width <= target_size)

                if not can_fit:
                    img = self.crop_image(img, target_size, target_size, RelativePosition)
                else:
                    if Padding:
                        img = self.scale_with_padding(img, target_size, target_size, RelativePosition, 
                                                    Supersampling, PaddingElements)
                    else:
                        img = img.resize((target_size, target_size), 
                                       self._INTERPOLATION_METHODS[Supersampling])

            # Convert back to tensor
            output_image = torch.from_numpy(np.array(img).astype(np.float32)) / 255.0
            output_image = output_image.unsqueeze(0)

            self.send_notification(f"Image scaled to {img.width}x{img.height}")
            return output_image, img.width, img.height

        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")

    @staticmethod
    def calculate_dimensions(width: int, height: int, max_pixels: int, 
                           scale_factor: int) -> Tuple[int, int]:
        """Calculate new dimensions while maintaining aspect ratio."""
        aspect_ratio = width / height
        new_width = min(width, math.isqrt(int(max_pixels * aspect_ratio)))
        new_height = min(height, math.isqrt(int(max_pixels / aspect_ratio)))

        new_width = (new_width // scale_factor) * scale_factor
        new_height = (new_height // scale_factor) * scale_factor
        return new_width, new_height

    @staticmethod
    def crop_image(img: Image.Image, target_width: int, target_height: int, 
                  reference_position: str) -> Image.Image:
        """Crop image to target dimensions based on reference position."""
        width, height = img.size
        left = 0
        top = 0
        right = width
        bottom = height

        if width > target_width:
            if "Left" in reference_position:
                right = target_width
            elif "Right" in reference_position:
                left = width - target_width
            else:  # Center
                left = (width - target_width) // 2
                right = left + target_width

        if height > target_height:
            if "Top" in reference_position:
                bottom = target_height
            elif "Bottom" in reference_position:
                top = height - target_height
            else:  # Center
                top = (height - target_height) // 2
                bottom = top + target_height

        return img.crop((left, top, right, bottom))

    @staticmethod
    def scale_with_padding(img: Image.Image, target_width: int, target_height: int, 
                          reference_position: str, supersampling_method: str, 
                          padding: str) -> Image.Image:
        """Scale image with padding while maintaining aspect ratio."""
        width, height = img.size
        aspect_ratio = width / height

        if width * target_height / height <= target_width:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        else:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)

        img = img.resize((new_width, new_height), 
                        ImageScaler._INTERPOLATION_METHODS[supersampling_method])

        new_img = Image.new("RGBA" if padding == "Transparent" else "RGB", 
                          (target_width, target_height), 
                          ImageScaler._PADDING_COLORS[padding])

        paste_x = 0
        paste_y = 0

        if "Left" in reference_position:
            paste_x = 0
        elif "Right" in reference_position:
            paste_x = target_width - new_width
        else:  # Center
            paste_x = (target_width - new_width) // 2

        if "Top" in reference_position:
            paste_y = 0
        elif "Bottom" in reference_position:
            paste_y = target_height - new_height
        else:  # Center
            paste_y = (target_height - new_height) // 2

        new_img.paste(img, (paste_x, paste_y), img if padding == "Transparent" else None)
        return new_img

    @staticmethod
    def send_notification(message: str) -> None:
        """Send a notification message."""
        print(f"[ImageScaler Notification]: {message}")


# Node registration
NODE_CLASS_MAPPINGS = {
    "ImageInfo": ImageInfo,
    "ImageScaler": ImageScaler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageInfo": "ImageInfo 🍿",
    "ImageScaler": "ImageScaler 🍿"
} 