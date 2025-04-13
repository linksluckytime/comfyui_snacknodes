# ComfyUI SnackNodes 🍿

A comprehensive collection of ComfyUI nodes designed to reduce reliance on multiple third-party node packages.

> 需要中文说明？[点击这里](./README_CN.md) 🎯

## Installation

1. Clone the repository:
```bash
git clone https://github.com/linksluckytime/comfyui_snacknodes.git
cd comfyui_snacknodes
```

2. Install base dependencies:
```bash
pip install -e .
```

3. For face detection features, install additional dependencies:
```bash
pip install -e ".[face]"
```

Note: On macOS, you may need to install CMake first:
```bash
brew install cmake
```

## Model Download

If you plan to use face detection features, download the following model:

1. Face Detection Model:
   - Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   - Extract the file
   - Place it in: `ComfyUI/models/dlib/`

## Available Nodes

### Image Processing
- **ImageInfo 🍿**  
  Extract image dimensions (width, height, batch size, channels).

- **ImageScaler 🍿**  
  Resize images with various methods (nearest, bilinear, bicubic, lanczos).

- **FaceDetector 🍿**  
  Detect faces and facial landmarks, providing masks and control points.

## Development Plans 🛠️✨

### Functional Components
- **Seed Value:**  
  Add randomization methods outside of the web interface for easier backend integration.

- **String Operations:**  
  Combine and replace strings.

### Mask Nodes
- **Feather Edges:**  
  Smooth the edges of masks.

- **Expand Inward/Outward:**  
  Adjust mask boundaries.

- **Mask Detection:**  
  Detect objects like people or limbs.

## License

GNU General Public License v3
