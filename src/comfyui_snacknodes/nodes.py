import os
import torch
import math
import numpy as np
from PIL import Image, ImageOps
import locale


class ImageScaler:
    CATEGORY = "SnackNodes"  # Node group name

    @classmethod
    def get_language(cls):
        try:
            lang = locale.getdefaultlocale()[0]
            return 'zh' if lang and lang.startswith('zh') else 'en'
        except:
            return 'en'

    @classmethod
    def INPUT_TYPES(cls):
        lang = cls.get_language()
        
        # Define options
        if lang == 'zh':
            pixel_budget_options = ["512像素平方", "1024像素平方"]
            scale_factor_options = [2, 8, 32, 64]
            reference_position_options = [
                "居中", "左上", "中上", "右上",
                "左中", "中中", "右中",
                "左下", "中下", "右下"
            ]
            supersampling_methods = [
                "无", "兰索斯", "最近邻", "双线性", "双三次", "盒式", "汉明"
            ]
            padding_options = ["透明", "灰色", "黑色", "白色"]
        else:
            pixel_budget_options = ["512px²", "1024px²"]
            scale_factor_options = [2, 8, 32, 64]
            reference_position_options = [
                "Center", "Top-Left", "Top-Center", "Top-Right",
                "Middle-Left", "Middle-Center", "Middle-Right",
                "Bottom-Left", "Bottom-Center", "Bottom-Right"
            ]
            supersampling_methods = [
                "None", "Lanczos", "Nearest", "Bilinear", "Bicubic", "Box", "Hamming"
            ]
            padding_options = ["Transparent", "Gray", "Black", "White"]

        if lang == 'zh':
            return {
                "required": {
                    "image": ("IMAGE", {"description": "输入图像张量 (B,H,W,C 格式)"}),
                    "preserve_aspect_ratio": ("BOOLEAN", {
                        "default": True,
                        "description": "保持宽高比"
                    }),
                    "pixel_budget": (pixel_budget_options, {
                        "default": "1024像素平方",
                        "description": "最大像素面积"
                    }),
                    "scale_factor": (scale_factor_options, {
                        "default": 64,
                        "description": "尺寸倍数"
                    }),
                    "reference_position": (reference_position_options, {
                        "default": "居中",
                        "description": "位置参考点"
                    }),
                    "supersampling_method": (supersampling_methods, {
                        "default": "无",
                        "description": "重采样方法"
                    }),
                    "enable_padding": ("BOOLEAN", {
                        "default": False,
                        "description": "启用填充"
                    }),
                    "padding": (padding_options, {
                        "default": "透明",
                        "description": "填充颜色"
                    }),
                },
            }
        else:
            return {
                "required": {
                    "image": ("IMAGE", {"description": "Input image tensor (B,H,W,C format)"}),
                    "preserve_aspect_ratio": ("BOOLEAN", {
                        "default": True,
                        "description": "Maintain aspect ratio"
                    }),
                    "pixel_budget": (pixel_budget_options, {
                        "default": "1024px²",
                        "description": "Max pixel area"
                    }),
                    "scale_factor": (scale_factor_options, {
                        "default": 64,
                        "description": "Dimension multiple"
                    }),
                    "reference_position": (reference_position_options, {
                        "default": "Center",
                        "description": "Position reference"
                    }),
                    "supersampling_method": (supersampling_methods, {
                        "default": "None",
                        "description": "Resampling method"
                    }),
                    "enable_padding": ("BOOLEAN", {
                        "default": False,
                        "description": "Enable padding"
                    }),
                    "padding": (padding_options, {
                        "default": "Transparent",
                        "description": "Padding color"
                    }),
                },
            }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "scale_image"

    def scale_image(self, image, preserve_aspect_ratio, pixel_budget, scale_factor, reference_position, supersampling_method, enable_padding, padding):
        # Map Chinese options to English for internal processing
        if self.get_language() == 'zh':
            pixel_budget_map = {
                "512像素平方": "512px²",
                "1024像素平方": "1024px²"
            }
            reference_position_map = {
                "居中": "Center", "左上": "Top-Left", "中上": "Top-Center", "右上": "Top-Right",
                "左中": "Middle-Left", "中中": "Middle-Center", "右中": "Middle-Right",
                "左下": "Bottom-Left", "中下": "Bottom-Center", "右下": "Bottom-Right"
            }
            supersampling_map = {
                "无": "None", "兰索斯": "Lanczos", "最近邻": "Nearest", "双线性": "Bilinear",
                "双三次": "Bicubic", "盒式": "Box", "汉明": "Hamming"
            }
            padding_map = {
                "透明": "Transparent", "灰色": "Gray", "黑色": "Black", "白色": "White"
            }
            
            pixel_budget = pixel_budget_map.get(pixel_budget, pixel_budget)
            reference_position = reference_position_map.get(reference_position, reference_position)
            supersampling_method = supersampling_map.get(supersampling_method, supersampling_method)
            padding = padding_map.get(padding, padding)

        # Validate inputs
        max_pixels = {"512px²": 512 ** 2, "1024px²": 1024 ** 2}.get(pixel_budget)
        assert max_pixels, f"Invalid pixel budget: {pixel_budget}"

        # Extract image dimensions
        batch_size, height, width, channels = image.shape
        img = Image.fromarray((image[0].numpy() * 255).astype("uint8"))

        # Ensure image is in RGBA mode for transparent padding
        if padding == "Transparent":
            img = img.convert("RGBA")

        # Calculate target dimensions
        if preserve_aspect_ratio:
            # Case 1: Maintain aspect ratio
            new_width, new_height = self.calculate_dimensions(width, height, max_pixels, scale_factor)
            img = img.resize((new_width, new_height), self.get_interpolation_method(supersampling_method))
        else:
            # Case 2: Don't maintain aspect ratio
            target_size = math.isqrt(max_pixels)
            target_size = (target_size // scale_factor) * scale_factor

            # Check if image can fit within target size while maintaining aspect ratio
            can_fit = (width <= target_size and height <= target_size) or \
                     (width * target_size / height <= target_size and height * target_size / width <= target_size)

            if not can_fit:
                # Case 2.1: Image cannot fit - crop
                img = self.crop_image(img, target_size, target_size, reference_position)
            else:
                if enable_padding:
                    # Case 2.2.1: Image can fit and padding is enabled
                    img = self.scale_with_padding(img, target_size, target_size, reference_position, 
                                                supersampling_method, padding)
                else:
                    # Case 2.2.2: Image can fit but padding is disabled - stretch
                    img = img.resize((target_size, target_size), self.get_interpolation_method(supersampling_method))

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
    "ImageScaler": ImageScaler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageScaler": "ImageScaler 🍿"
}