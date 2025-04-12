from PIL import Image, ImageDraw
import numpy as np

def create_popcorn_icon():
    # 创建一个 200x200 的透明背景图像
    size = 200
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 设置颜色
    yellow = (255, 215, 0)  # 金色
    brown = (139, 69, 19)   # 棕色
    
    # 绘制爆米花形状
    # 中心点
    center_x, center_y = size // 2, size // 2
    
    # 绘制主要爆米花形状
    for i in range(5):
        # 计算角度
        angle = i * 72  # 72度 = 360/5
        # 转换为弧度
        rad = np.radians(angle)
        # 计算点位置
        x = center_x + int(40 * np.cos(rad))
        y = center_y + int(40 * np.sin(rad))
        # 绘制爆米花粒
        draw.ellipse((x-15, y-15, x+15, y+15), fill=yellow)
        # 添加棕色斑点
        draw.ellipse((x-5, y-5, x+5, y+5), fill=brown)
    
    # 保存图像
    img.save('web/icon.png')

if __name__ == '__main__':
    create_popcorn_icon() 