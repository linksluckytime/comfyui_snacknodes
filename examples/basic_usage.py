"""
Basic usage example for comfyui_snacknodes.
This example demonstrates how to use the ImageInfo and ImageScaler nodes.
"""

import torch
from PIL import Image
import numpy as np

# Create a sample image
def create_sample_image(width=512, height=512):
    """Create a sample image with a gradient."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            img[y, x] = [x % 256, y % 256, (x + y) % 256]
    return img

# Convert PIL Image to torch tensor
def pil_to_tensor(img):
    """Convert PIL Image to torch tensor."""
    img = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)

# Convert torch tensor to PIL Image
def tensor_to_pil(tensor):
    """Convert torch tensor to PIL Image."""
    img = tensor.squeeze(0).numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def main():
    # Create a sample image
    img = create_sample_image()
    img_tensor = pil_to_tensor(img)
    
    # Use ImageInfo node
    from comfyui_snacknodes.image_nodes import ImageInfo
    info_node = ImageInfo()
    batch_size, height, width, channels = info_node.get_image_info(img_tensor)
    print(f"Image Info: {batch_size}x{height}x{width}x{channels}")
    
    # Use ImageScaler node
    from comfyui_snacknodes.image_nodes import ImageScaler
    scaler_node = ImageScaler()
    scaled_img, new_width, new_height = scaler_node.scale_image(
        image=img_tensor,
        KeepProportion=True,
        Pixels="512px²",
        ScalingFactor=64,
        RelativePosition="Center",
        Supersampling="Lanczos",
        Padding=True,
        PaddingElements="Transparent"
    )
    print(f"Scaled Image: {new_width}x{new_height}")
    
    # Save the scaled image
    scaled_pil = tensor_to_pil(scaled_img)
    scaled_pil.save("scaled_image.png")

if __name__ == "__main__":
    main() 