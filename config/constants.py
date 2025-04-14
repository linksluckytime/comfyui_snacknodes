"""Constants for comfyui_snacknodes."""

from PIL import Image

# Padding colors
PADDING_COLORS = {
    "transparent": (0, 0, 0, 0),
    "black": (0, 0, 0, 255),
    "white": (255, 255, 255, 255),
    "gray": (128, 128, 128, 255),
    "red": (255, 0, 0, 255),
    "green": (0, 255, 0, 255),
    "blue": (0, 0, 255, 255)
}

# Interpolation methods
INTERPOLATION_METHODS = {
    "none": Image.NONE,
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
    "hamming": Image.HAMMING,
    "box": Image.BOX
}

# Relative positions (9-grid layout)
RELATIVE_POSITIONS = [
    "top left",
    "top center",
    "top right",
    "center left",
    "center",
    "center right",
    "bottom left",
    "bottom center",
    "bottom right"
]

# Pixel budgets
PIXEL_BUDGETS = {
    "512px²": 512,
    "768px²": 768,
    "1024px²": 1024,
    "2048px²": 2048
}

# Scaling factors
SCALING_FACTORS = [2, 8, 16, 32, 64] 