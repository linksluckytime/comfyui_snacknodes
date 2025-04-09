import torch
import math
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import comfy.utils


class SnackImageScaler:
    CATEGORY = "SnackNodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"description": "Input image tensor (B,H,W,C format)"}),
                "pixel_budget": (["512px²", "1024px²"], {
                    "default": "1024px²",
                    "description": "Maximum allowed pixel area"
                }),
                "scale_factor": ([2, 8, 16, 32, 64], {
                    "default": 64,
                    "description": "Force output dimensions to be multiples of this value"
                }),
                "interpolation": (["nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"], {
                    "default": "nearest",
                    "description": "Resampling algorithm"
                }),
                "aspect_ratio": ("BOOLEAN", {
                    "default": True,
                    "description": "Maintain original width/height ratio"
                }),
                "padding": (["transparent", "black", "white", "edge", "blurred_layer"], {
                    "default": "transparent",
                    "description": "Color used for padding"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "scale_image"

    def scale_image(self, image, pixel_budget, scale_factor, interpolation, aspect_ratio, padding):
        # Validate inputs
        max_pixels = {"512px²": 512 ** 2, "1024px²": 1024 ** 2}.get(pixel_budget)
        assert max_pixels, f"Invalid pixel budget: {pixel_budget}"
        assert interpolation in ["nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"], f"Invalid interpolation: {interpolation}"
        assert padding in ["transparent", "black", "white", "edge", "blurred_layer"], f"Invalid padding: {padding}"

        # Extract image dimensions
        batch_size, height, width, channels = image.shape
        img = Image.fromarray((image[0].numpy() * 255).astype("uint8"))

        # Calculate target dimensions
        new_width, new_height = self.calculate_dimensions(width, height, max_pixels, aspect_ratio, scale_factor)

        # Resize image
        img = img.resize((new_width, new_height), self.get_interpolation_method(interpolation))

        # Apply padding if needed
        if aspect_ratio:
            img, pad_width, pad_height = self.apply_padding(img, new_width, new_height, scale_factor, padding)
        else:
            pad_width, pad_height = 0, 0

        # Convert back to tensor
        output_image = torch.from_numpy(np.array(img).astype(np.float32)) / 255.0
        output_image = output_image.unsqueeze(0)

        # Send success notification
        self.send_notification(f"Image scaled to {new_width + pad_width}x{new_height + pad_height}")
        return output_image, new_width + pad_width, new_height + pad_height

    @staticmethod
    def calculate_dimensions(width, height, max_pixels, aspect_ratio, scale_factor):
        if aspect_ratio:
            aspect_ratio_value = width / height
            new_width = min(width, math.isqrt(int(max_pixels * aspect_ratio_value)))
            new_height = min(height, math.isqrt(int(max_pixels / aspect_ratio_value)))
        else:
            new_width = new_height = math.isqrt(max_pixels)

        # Align dimensions to scale factor
        new_width = (new_width // scale_factor) * scale_factor
        new_height = (new_height // scale_factor) * scale_factor
        return new_width, new_height

    @staticmethod
    def get_interpolation_method(interpolation):
        interpolation_map = {
            "nearest": Image.NEAREST,
            "box": Image.BOX,
            "bilinear": Image.BILINEAR,
            "hamming": Image.HAMMING,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }
        return interpolation_map[interpolation]

    @staticmethod
    def apply_padding(img, width, height, scale_factor, padding):
        pad_width = (scale_factor - (width % scale_factor)) % scale_factor
        pad_height = (scale_factor - (height % scale_factor)) % scale_factor

        if pad_width > 0 or pad_height > 0:
            if padding == "edge":
                img = ImageOps.expand(img, (0, 0, pad_width, pad_height), fill=None)
                img = img.crop((0, 0, width + pad_width, height + pad_height))
            elif padding == "blurred_layer":
                blurred_img = img.filter(ImageFilter.GaussianBlur(10))
                img = ImageOps.expand(blurred_img, (0, 0, pad_width, pad_height), fill=None)
                img.paste(img, (0, 0))
            else:
                padding_values = (0, 0, pad_width, pad_height)
                fill_color = {
                    "black": (0, 0, 0),
                    "white": (255, 255, 255),
                    "transparent": (0, 0, 0, 0)
                }[padding]
                img = ImageOps.expand(img, padding_values, fill=fill_color)
        return img, pad_width, pad_height

    @staticmethod
    def send_notification(message):
        # 替换 emit_socket_event 为简单的打印日志
        print(f"[SnackNodes Notification]: {message}")


# Node registration
NODE_CLASS_MAPPINGS = {
    "SnackImageScaler": SnackImageScaler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SnackImageScaler": "Image Scaler 🍿"
}