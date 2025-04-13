# ComfyUI SnackNodes 🍿

一个功能丰富的 ComfyUI 节点集合，就像零食一样，没有也能活，但有的话会更开心 😋

> Need English documentation? [Click here](./README.md) 🎯

## 安装指南

1. 克隆仓库：
```bash
git clone https://github.com/linksluckytime/comfyui_snacknodes.git
cd comfyui_snacknodes
```

2. 安装基础依赖：
```bash
pip install -e .
```

3. 如果需要使用人脸检测功能，安装额外依赖：
```bash
pip install -e ".[face]"
```

小贴士：在 macOS 上安装 dlib 可能需要先安装 CMake：
```bash
brew install cmake
```

## 模型下载

如果你打算使用人脸检测功能，需要下载以下模型：

1. 人脸检测模型：
   - 下载地址：http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   - 解压文件
   - 将文件放在：`ComfyUI/models/dlib/` 目录下

## 现有节点

### 图像处理
- **ImageInfo 🍿**  
  获取图像信息（宽度、高度、批处理大小、通道数）

- **ImageScaler 🍿**  
  图像缩放，支持多种缩放方法（最近邻、双线性、双三次、Lanczos）

- **FaceDetector 🍿**  
  人脸检测和特征点提取，提供蒙版和控制点

## 开发计划 🛠️✨

### 功能组件
- **随机种子：**  
  添加网页界面外的随机化方法，方便后端集成

- **字符串操作：**  
  字符串组合和替换

### 蒙版节点
- **羽化边缘：**  
  平滑蒙版边缘

- **向内/向外扩展：**  
  调整蒙版边界

- **蒙版检测：**  
  检测人物或肢体等对象

## 许可证

GNU General Public License v3 