<div style="font-family: 'Georgia', 'Times New Roman', Times, serif;">

# ComfyUI SnackNodes

一个功能丰富的 ComfyUI 节点集合，旨在减少对多个第三方节点包的依赖。

## 安装

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

注意：在 macOS 上安装 dlib 可能需要先安装 CMake：
```bash
brew install cmake
```

## 模型下载

如果需要使用人脸检测功能，请下载以下模型文件：

1. 下载人脸检测模型：
   - 访问：http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   - 解压文件
   - 将解压后的文件放在：`ComfyUI/models/dlib/` 目录下

## 功能节点

### 图像处理
- ImageInfo：获取图像信息（宽度、高度、批处理大小、通道数）
- ImageScaler：图像缩放，支持多种缩放方法
- FaceDetector：人脸检测和特征点提取（需要安装额外依赖）

## 开发计划
- [ ] 添加更多图像处理节点
- [ ] 优化现有节点性能
- [ ] 添加单元测试
- [ ] 完善文档

## 许可证
GNU General Public License v3
