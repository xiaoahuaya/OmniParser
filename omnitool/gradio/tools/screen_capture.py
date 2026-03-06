from pathlib import Path
from uuid import uuid4

import os
import time
import pyautogui
import requests
from PIL import Image
from .base import BaseAnthropicTool, ToolError
from io import BytesIO

OUTPUT_DIR = "./tmp/outputs"
MAX_OUTPUT_FILES = int(os.getenv("OMNITOOL_OUTPUT_MAX_FILES", "800"))
MAX_OUTPUT_AGE_HOURS = float(os.getenv("OMNITOOL_OUTPUT_MAX_AGE_HOURS", "24"))
OUTPUT_CLEANUP_INTERVAL = int(os.getenv("OMNITOOL_OUTPUT_CLEANUP_INTERVAL", "20"))
_CAPTURE_COUNT = 0


def _cleanup_output_dir(output_dir: Path):
    if not output_dir.exists():
        return
    files = sorted(
        output_dir.glob("*.png"),
        key=lambda p: p.stat().st_mtime,
    )
    if not files:
        return

    now = time.time()
    max_age_seconds = MAX_OUTPUT_AGE_HOURS * 3600
    for path in files:
        try:
            if max_age_seconds > 0 and (now - path.stat().st_mtime) > max_age_seconds:
                path.unlink(missing_ok=True)
        except OSError:
            continue

    files = sorted(
        output_dir.glob("*.png"),
        key=lambda p: p.stat().st_mtime,
    )
    overflow = len(files) - MAX_OUTPUT_FILES
    if overflow > 0:
        for old_path in files[:overflow]:
            try:
                old_path.unlink(missing_ok=True)
            except OSError:
                continue

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
    """Capture screenshot from remote host if configured; otherwise local capture."""
    # 创建保存截图的目录
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成唯一的文件名
    path = output_dir / f"screenshot_{uuid4().hex}.png"

    global _CAPTURE_COUNT
    try:
        windows_host_url = os.getenv("OMNITOOL_WINDOWS_HOST_URL", "").strip()
        if windows_host_url:
            if not windows_host_url.startswith("http"):
                windows_host_url = f"http://{windows_host_url}"
            response = requests.get(
                f"{windows_host_url}/screenshot",
                timeout=30,
                proxies={"http": "", "https": ""},
            )
            if response.status_code != 200:
                raise Exception(f"remote screenshot failed: HTTP {response.status_code}")
            screenshot = Image.open(BytesIO(response.content))
        else:
            # 使用 pyautogui 获取当前屏幕截图
            screenshot = pyautogui.screenshot()

        # 如果需要调整大小
        if resize and screenshot.size != (target_width, target_height):
            screenshot = screenshot.resize((target_width, target_height))

        # 保存截图
        screenshot.save(path)
        _CAPTURE_COUNT += 1
        if OUTPUT_CLEANUP_INTERVAL > 0 and (_CAPTURE_COUNT % OUTPUT_CLEANUP_INTERVAL == 0):
            _cleanup_output_dir(output_dir)

        return screenshot, path
    except Exception as e:
        raise Exception(f"Failed to capture screenshot: {str(e)}")
