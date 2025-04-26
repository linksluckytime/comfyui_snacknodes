"""Text Box node for ComfyUI."""

import logging
from typing import Dict, Tuple
from ..common.base_node import BaseNode

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextBox(BaseNode):
    """A node for handling multiline text input.
    
    This node provides a simple interface for text input with validation
    and error handling capabilities.
    """
    
    CATEGORY = "SnackNodes"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "get_text"
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        """Define the input types for this node."""
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your text here..."
                }),
            }
        }
    
    def get_text(self, text: str) -> Tuple[str]:
        """Get the input text with validation.
        
        Args:
            text: The input text
            
        Returns:
            Tuple containing the validated text
            
        Raises:
            ValueError: If text validation fails
        """
        try:
            # 验证输入
            if text is None:
                raise ValueError("Input text cannot be None")
                
            # 记录调试信息
            logger.debug(f"Processing text input: {text[:50]}...")
            
            # 返回处理后的文本
            return (text,)
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            # 发生错误时返回空字符串
            return ("",) 