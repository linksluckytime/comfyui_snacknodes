import os
import torch
import math
import numpy as np
from PIL import Image, ImageOps


class ImageInfo:
    CATEGORY = "SnackNodes"  # Node group name

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"description": "Input image tensor (B,H,W,C format)"}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("batch_size", "height", "width", "channels")
    FUNCTION = "get_image_info"

    def get_image_info(self, image):
        batch_size, height, width, channels = image.shape
        return batch_size, height, width, channels


class ImageScaler:
    CATEGORY = "SnackNodes"  # Node group name

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"description": "Input image tensor (B,H,W,C format)"}),
                "KeepProportion": ("BOOLEAN", {
                    "default": True,
                    "description": "Keep proportion"
                }),
                "Pixels": (["512px²", "1024px²"], {
                    "default": "1024px²",
                    "description": "Reference pixels"
                }),
                "ScalingFactor": ([2, 8, 32, 64], {
                    "default": 64,
                    "description": "Scaling factor"
                }),
                "RelativePosition": ([
                    "Center", "Top-Left", "Top-Center", "Top-Right",
                    "Middle-Left", "Middle-Center", "Middle-Right",
                    "Bottom-Left", "Bottom-Center", "Bottom-Right"
                ], {
                    "default": "Center",
                    "description": "Relative position"
                }),
                "Supersampling": ([
                    "None", "Lanczos", "Nearest", "Bilinear", "Bicubic", "Box", "Hamming"
                ], {
                    "default": "None",
                    "description": "Supersampling"
                }),
                "Padding": ("BOOLEAN", {
                    "default": False,
                    "description": "Padding"
                }),
                "PaddingElements": (["Transparent", "Gray", "Black", "White"], {
                    "default": "Transparent",
                    "description": "Padding elements"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "scale_image"

    def scale_image(self, image, KeepProportion, Pixels, ScalingFactor, RelativePosition, Supersampling, Padding, PaddingElements):
        # Validate inputs
        max_pixels = {"512px²": 512 ** 2, "1024px²": 1024 ** 2}.get(Pixels)
        assert max_pixels, f"Invalid pixel budget: {Pixels}"

        # Extract image dimensions
        batch_size, height, width, channels = image.shape
        img = Image.fromarray((image[0].numpy() * 255).astype("uint8"))

        # Ensure image is in RGBA mode for transparent padding
        if PaddingElements == "Transparent":
            img = img.convert("RGBA")

        # Calculate target dimensions
        if KeepProportion:
            # Case 1: Maintain aspect ratio
            new_width, new_height = self.calculate_dimensions(width, height, max_pixels, ScalingFactor)
            img = img.resize((new_width, new_height), self.get_interpolation_method(Supersampling))
        else:
            # Case 2: Don't maintain aspect ratio
            target_size = math.isqrt(max_pixels)
            target_size = (target_size // ScalingFactor) * ScalingFactor

            # Check if image can fit within target size while maintaining aspect ratio
            can_fit = (width <= target_size and height <= target_size) or \
                     (width * target_size / height <= target_size and height * target_size / width <= target_size)

            if not can_fit:
                # Case 2.1: Image cannot fit - crop
                img = self.crop_image(img, target_size, target_size, RelativePosition)
            else:
                if Padding:
                    # Case 2.2.1: Image can fit and padding is enabled
                    img = self.scale_with_padding(img, target_size, target_size, RelativePosition, 
                                                Supersampling, PaddingElements)
                else:
                    # Case 2.2.2: Image can fit but padding is disabled - stretch
                    img = img.resize((target_size, target_size), self.get_interpolation_method(Supersampling))

        # Convert back to tensor
        output_image = torch.from_numpy(np.array(img).astype(np.float32)) / 255.0
        output_image = output_image.unsqueeze(0)

        # Send success notification
        self.send_notification(f"Image scaled to {img.width}x{img.height}")
        return output_image, img.width, img.height

    @staticmethod
    def calculate_dimensions(width, height, max_pixels, scale_factor):
        aspect_ratio = width / height
        new_width = min(width, math.isqrt(int(max_pixels * aspect_ratio)))
        new_height = min(height, math.isqrt(int(max_pixels / aspect_ratio)))

        # Align dimensions to scale factor
        new_width = (new_width // scale_factor) * scale_factor
        new_height = (new_height // scale_factor) * scale_factor
        return new_width, new_height

    @staticmethod
    def crop_image(img, target_width, target_height, reference_position):
        # Calculate crop box based on reference position
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
    def scale_with_padding(img, target_width, target_height, reference_position, supersampling_method, padding):
        # First scale the image to fit within target dimensions while maintaining aspect ratio
        width, height = img.size
        aspect_ratio = width / height

        if width * target_height / height <= target_width:
            # Fit to height
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        else:
            # Fit to width
            new_width = target_width
            new_height = int(target_width / aspect_ratio)

        # Scale the image
        img = img.resize((new_width, new_height), ImageScaler.get_interpolation_method(supersampling_method))

        # Create a new image with padding
        new_img = Image.new("RGBA" if padding == "Transparent" else "RGB", 
                          (target_width, target_height), 
                          ImageScaler.get_padding_color(padding))

        # Calculate paste position based on reference position
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

        # Paste the scaled image
        new_img.paste(img, (paste_x, paste_y), img if padding == "Transparent" else None)
        return new_img

    @staticmethod
    def get_padding_color(padding):
        return {
            "Transparent": (0, 0, 0, 0),
            "Gray": (128, 128, 128),
            "Black": (0, 0, 0),
            "White": (255, 255, 255),
        }[padding]

    @staticmethod
    def get_interpolation_method(interpolation):
        interpolation_map = {
            "Nearest": Image.NEAREST,
            "Box": Image.BOX,
            "Bilinear": Image.BILINEAR,
            "Hamming": Image.HAMMING,
            "Bicubic": Image.BICUBIC,
            "Lanczos": Image.LANCZOS,
            "None": Image.NEAREST,
        }
        return interpolation_map[interpolation]

    @staticmethod
    def send_notification(message):
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