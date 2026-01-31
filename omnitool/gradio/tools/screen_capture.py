from pathlib import Path
from uuid import uuid4

import pyautogui
import requests
from PIL import Image
from .base import BaseAnthropicTool, ToolError
from io import BytesIO

OUTPUT_DIR = "./tmp/outputs"

#注释下面的代码

# def get_screenshot(resize: bool = False, target_width: int = 1920, target_height: int = 1080):
#     """Capture screenshot by requesting from HTTP endpoint - returns native resolution unless resized"""
#     output_dir = Path(OUTPUT_DIR)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     path = output_dir / f"screenshot_{uuid4().hex}.png"
#
#     try:
#         response = requests.get('http://localhost:5000/screenshot')
#         if response.status_code != 200:
#             raise ToolError(f"Failed to capture screenshot: HTTP {response.status_code}")
#
#         # (1280, 800)
#         screenshot = Image.open(BytesIO(response.content))
#
#         if resize and screenshot.size != (target_width, target_height):
#             screenshot = screenshot.resize((target_width, target_height))
#         screenshot.save(path)
#         return screenshot, path
#     except Exception as e:
#         raise ToolError(f"Failed to capture screenshot: {str(e)}")


def get_screenshot(resize: bool = False, target_width: int = 1920, target_height: int = 1080):
    """Capture screenshot locally and resize if needed, then save it to a file."""
    # 创建保存截图的目录
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成唯一的文件名
    path = output_dir / f"screenshot_{uuid4().hex}.png"

    try:
        # 使用 pyautogui 获取当前屏幕截图
        screenshot = pyautogui.screenshot()

        # 如果需要调整大小
        if resize and screenshot.size != (target_width, target_height):
            screenshot = screenshot.resize((target_width, target_height))

        # 保存截图
        screenshot.save(path)

        return screenshot, path
    except Exception as e:
        raise Exception(f"Failed to capture screenshot: {str(e)}")