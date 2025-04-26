"""Text Processor node for ComfyUI."""

import re
import logging
from typing import Dict, Any, Tuple, List
from ..common.base_node import BaseNode

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _concatenate_texts(texts: List[str], optimize: bool = True) -> str:
    """Concatenate multiple texts with optional optimization.
    
    Args:
        texts: List of texts to concatenate
        optimize: Whether to optimize the concatenation
        
    Returns:
        Concatenated text
    """
    # 过滤空值
    valid_texts = [text.strip() for text in texts if text and text.strip()]
    
    if not valid_texts:
        return ""
    
    if optimize:
        # 处理每个文本,只在非句号结尾的文本后添加逗号
        processed_texts = []
        for i, text in enumerate(valid_texts):
            if i < len(valid_texts) - 1:
                if text.endswith('.'):
                    text = text[:-1]
                text += ','
            processed_texts.append(text)
        # 使用空格连接
        return " ".join(processed_texts)
    else:
        # 不优化，直接连接
        return "".join(valid_texts)

def _search_replace(search: str, replace: str, target: str) -> str:
    """Replace occurrences of search text with replacement text.
    
    Args:
        search: Text to search for
        replace: Replacement text
        target: Target text to search in
        
    Returns:
        Text with replacements made
    """
    # 确保输入不为None
    search = search or ""
    replace = replace or ""
    target = target or ""
    
    return target.replace(search, replace)

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
                "operation": (["concatenate", "search_replace"], {"default": "concatenate"}),
                "optimize_concatenation": ("BOOLEAN", {"default": True}),
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
                       search_text: str, replace_text: str, target_text: str) -> Dict[str, Any]:
        """Process the input strings based on the selected operation.
        
        Args:
            operation: The operation to perform ("concatenate" or "search_replace")
            optimize_concatenation: Whether to optimize the concatenated text
            search_text: Text to search for or first part to concatenate
            replace_text: Replacement text or second part to concatenate
            target_text: Target text to search in or third part to concatenate
            
        Returns:
            Dictionary containing the processed text and UI display information
            
        Raises:
            ValueError: If operation is invalid
            Exception: If an error occurs during processing
        """
        try:
            logger.info(f"Processing text with operation: {operation}")
            
            if operation == "concatenate":
                result = _concatenate_texts([search_text, replace_text, target_text], 
                                         optimize_concatenation)
            elif operation == "search_replace":
                result = _search_replace(search_text, replace_text, target_text)
            else:
                raise ValueError(f"Invalid operation: {operation}")
                
            logger.debug(f"Operation result: {result}")
            return {"ui": {"text": (result,)}, "result": (result,)}
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return {"ui": {"text": ("",)}, "result": ("",)} 