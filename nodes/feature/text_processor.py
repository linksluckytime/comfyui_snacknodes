"""Text Processor node for ComfyUI."""

import re
from typing import Dict, Tuple
from ..base_node import BaseNode

class TextProcessor(BaseNode):
    """A node for processing strings with various operations."""
    
    CATEGORY = "SnackNodes"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process_strings"
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        """Define the input types for this node."""
        return {
            "required": {
                "operation": (["concatenate", "search_replace"],),
                "optimize_concatenation": ("BOOLEAN", {"default": False}),
                "text1": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Concatenate: First part | Search: Text to find"
                }),
                "text2": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Concatenate: Second part | Search: Replacement text"
                }),
                "text3": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Concatenate: Third part | Search: Target text to search in"
                }),
            }
        }
    
    def process_strings(self, operation: str, optimize_concatenation: bool, 
                       text1: str, text2: str, text3: str) -> Tuple[str]:
        """Process the input strings based on the selected operation.
        
        Args:
            operation: The operation to perform
            optimize_concatenation: Whether to optimize the concatenated text
            text1: First input text
            text2: Second input text
            text3: Third input text
            
        Returns:
            Tuple containing the processed text
        """
        try:
            if operation == "concatenate":
                # 使用filter过滤空值，并使用join添加分隔符
                texts = filter(None, [text1, text2, text3])
                result = (", " if optimize_concatenation else "").join(texts)
            else:  # search_replace
                # 确保所有输入不为None
                text1 = text1 if text1 is not None else ""
                text2 = text2 if text2 is not None else ""
                text3 = text3 if text3 is not None else ""
                
                # 执行替换操作：在text3中搜索text1并替换为text2
                result = text3.replace(text1, text2)
            
            return (result,)
        except Exception as e:
            # 如果发生错误，返回空字符串
            return ("",) 