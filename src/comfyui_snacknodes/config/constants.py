"""Constants used across the ComfyUI SnackNodes package."""

# Padding colors
PADDING_COLORS = {
    "Transparent": (0, 0, 0, 0),
    "Gray": (128, 128, 128),
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
}

# Interpolation methods
INTERPOLATION_METHODS = {
    "Nearest": "NEAREST",
    "Box": "BOX",
    "Bilinear": "BILINEAR",
    "Hamming": "HAMMING",
    "Bicubic": "BICUBIC",
    "Lanczos": "LANCZOS",
    "None": "NEAREST",
}

# Relative positions
RELATIVE_POSITIONS = [
    "Center", "Top-Left", "Top-Center", "Top-Right",
    "Middle-Left", "Middle-Center", "Middle-Right",
    "Bottom-Left", "Bottom-Center", "Bottom-Right"
]

# Pixel budgets
PIXEL_BUDGETS = {
    "512px²": 512 ** 2,
    "1024px²": 1024 ** 2
}

# Scaling factors
SCALING_FACTORS = [2, 8, 32, 64] 