"""Text Processor node for ComfyUI."""

import re
import logging
from typing import Dict, Tuple
from ..base_node import BaseNode

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessor(BaseNode):
    """A node for processing strings with various operations.
    
    This node provides two main operations:
    1. Concatenate: Combines multiple text inputs with optional optimization
    2. Search and Replace: Replaces occurrences of search text with replacement text
    """
    
    CATEGORY = "SnackNodes"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process_strings"
    OUTPUT_NODE = True  # 启用输出节点功能
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        """Define the input types for this node."""
        return {
            "required": {
                "operation": (["concatenate", "search_replace"],),
                "optimize_concatenation": ("BOOLEAN", {"default": False}),
                "search_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Concatenate: First part | Search: Text to find"
                }),
                "replace_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Concatenate: Second part | Search: Replacement text"
                }),
                "target_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Concatenate: Third part | Search: Target text to search in"
                }),
            }
        }
    
    def process_strings(self, operation: str, optimize_concatenation: bool, 
                       search_text: str, replace_text: str, target_text: str) -> Tuple[str]:
        """Process the input strings based on the selected operation.
        
        Args:
            operation: The operation to perform ("concatenate" or "search_replace")
            optimize_concatenation: Whether to optimize the concatenated text
            search_text: Text to search for or first part to concatenate
            replace_text: Replacement text or second part to concatenate
            target_text: Target text to search in or third part to concatenate
            
        Returns:
            Tuple containing the processed text and UI display information
            
        Raises:
            Exception: If an error occurs during processing
        """
        try:
            logger.info(f"Processing text with operation: {operation}")
            
            if operation == "concatenate":
                # 使用filter过滤空值，并使用join添加分隔符
                texts = filter(None, [search_text, replace_text, target_text])
                result = (", " if optimize_concatenation else "").join(texts)
                logger.debug(f"Concatenated result: {result}")
            else:  # search_replace
                # 确保所有输入不为None
                search_text = search_text if search_text is not None else ""
                replace_text = replace_text if replace_text is not None else ""
                target_text = target_text if target_text is not None else ""
                
                # 执行替换操作
                result = target_text.replace(search_text, replace_text)
                logger.debug(f"Search and replace result: {result}")
            
            # 返回结果和UI显示信息
            return {"ui": {"text": (result,)}, "result": (result,)}
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            # 如果发生错误，返回空字符串
            return {"ui": {"text": ("",)}, "result": ("",)} 