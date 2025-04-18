"""Web resources for comfyui_snacknodes."""

import os
from pathlib import Path

folder_paths = None

# 获取当前文件所在目录
current_dir = Path(__file__).parent

# 定义JS文件路径
js_path = current_dir / "js" / "text_processor.js"

# 确保JS文件存在
if js_path.exists():
    __all__ = ["js_path"]
else:
    __all__ = [] 