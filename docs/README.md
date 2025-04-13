# ComfyUI SnackNodes Documentation

## Overview

ComfyUI SnackNodes is a collection of custom nodes for ComfyUI that provides various image processing and utility functions.

## Installation

```bash
# Clone the repository
git clone https://github.com/linksluckytime/comfyui_snacknodes.git
cd comfyui_snacknodes

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Usage

### ImageInfo Node

The ImageInfo node provides information about an input image tensor.

```python
from comfyui_snacknodes.image_nodes import ImageInfo

info_node = ImageInfo()
batch_size, height, width, channels = info_node.get_image_info(image_tensor)
```

### ImageScaler Node

The ImageScaler node provides various image scaling and transformation options.

```python
from comfyui_snacknodes.image_nodes import ImageScaler

scaler_node = ImageScaler()
scaled_image, new_width, new_height = scaler_node.scale_image(
    image=image_tensor,
    KeepProportion=True,
    Pixels="512px²",
    ScalingFactor=64,
    RelativePosition="Center",
    Supersampling="Lanczos",
    Padding=True,
    PaddingElements="Transparent"
)
```

## Examples

See the [examples](examples/) directory for more usage examples.

## API Reference

See the [API documentation](api/) for detailed information about each node and its parameters.

## Contributing

Contributions are welcome! Please see our [contributing guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the GNU General Public License v3. See the [LICENSE](LICENSE) file for details. 