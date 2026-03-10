import os
import logging
import argparse
import shlex
import subprocess
import json
import contextlib
import io as pyio
from flask import Flask, request, jsonify, send_file
import threading
import traceback
import pyautogui
from PIL import Image
from io import BytesIO
try:
    import mss
except ImportError:
    mss = None
try:
    import win32gui
    import win32process
except ImportError:
    win32gui = None
    win32process = None


def execute_anything(data):
    """Execute any command received in the JSON request.
    WARNING: This function executes commands without any safety checks."""
    # The 'command' key in the JSON request should contain the command to be executed.
    shell = data.get('shell', False)
    command = data.get('command', "" if shell else [])

    if isinstance(command, str) and not shell:
        command = shlex.split(command)

    # Expand user directory
    for i, arg in enumerate(command):
        if arg.startswith("~/"):
            command[i] = os.path.expanduser(arg)

    # Execute the command without any safety checks.
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell, text=True, timeout=120)
        return jsonify({
            'status': 'success',
            'output': result.stdout,
            'error': result.stderr,
            'returncode': result.returncode
        })
    except Exception as e:
        logger.error("\n" + traceback.format_exc() + "\n")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    

def execute(data):
    """
    Action-aware implementation.
    If payload is `python -c "<code>"`, execute inline in this process so pyautogui
    runs in the same desktop session as the server.
    """
    command = data.get("command", [])
    if (
        isinstance(command, list)
        and len(command) >= 3
        and isinstance(command[0], str)
        and isinstance(command[1], str)
        and command[1] == "-c"
    ):
        code = command[2]
        stdout = pyio.StringIO()
        stderr = pyio.StringIO()
        scope = {
            "pyautogui": pyautogui,
            "subprocess": subprocess,
            "time": __import__("time"),
            "os": os,
        }
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(code, scope, scope)
            return jsonify(
                {
                    "status": "success",
                    "output": stdout.getvalue(),
                    "error": stderr.getvalue(),
                    "returncode": 0,
                }
            )
        except Exception:
            logger.error("\n" + traceback.format_exc() + "\n")
            return jsonify(
                {
                    "status": "success",
                    "output": stdout.getvalue(),
                    "error": stderr.getvalue() + traceback.format_exc(),
                    "returncode": 1,
                }
            )
    return execute_anything(data)


execute_impl = execute


parser = argparse.ArgumentParser()
parser.add_argument("--log_file", help="log file path", type=str,
                    default=os.path.join(os.path.dirname(__file__), "server.log"))
parser.add_argument("--port", help="port", type=int, default=5000)
args = parser.parse_args()

logging.basicConfig(filename=args.log_file,level=logging.DEBUG, filemode='w' )
logger = logging.getLogger('werkzeug')

app = Flask(__name__)

computer_control_lock = threading.Lock()
window_target_lock = threading.Lock()
WINDOW_TARGET = {
    "hwnd": None,
    "title": "",
    "process": "",
}


def _normalize_text(value):
    return str(value or "").strip()


def _screen_rect():
    width, height = pyautogui.size()
    return {
        "x": 0,
        "y": 0,
        "width": int(width),
        "height": int(height),
        "title": "",
        "process": "",
        "hwnd": None,
        "target_found": False,
        "target_configured": bool(WINDOW_TARGET.get("hwnd") or WINDOW_TARGET.get("title") or WINDOW_TARGET.get("process")),
    }


def _get_process_name(pid):
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        line = (result.stdout or "").strip().splitlines()
        if not line:
            return ""
        first = line[0].strip()
        if not first or first.startswith("INFO:"):
            return ""
        parts = next(iter([row for row in __import__("csv").reader([first])]), [])
        return parts[0].strip() if parts else ""
    except Exception:
        return ""


def _is_window_candidate(hwnd):
    if win32gui is None:
        return False
    try:
        if not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd):
            return False
        if win32gui.IsIconic(hwnd):
            return False
        title = _normalize_text(win32gui.GetWindowText(hwnd))
        if not title:
            return False
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        return (right - left) > 80 and (bottom - top) > 80
    except Exception:
        return False


def _window_info_from_hwnd(hwnd):
    if win32gui is None or win32process is None or not _is_window_candidate(hwnd):
        return None
    try:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        _thread_id, pid = win32process.GetWindowThreadProcessId(hwnd)
        title = _normalize_text(win32gui.GetWindowText(hwnd))
        process = _get_process_name(pid)
        return {
            "hwnd": int(hwnd),
            "pid": int(pid),
            "title": title,
            "process": process,
            "x": int(left),
            "y": int(top),
            "width": int(max(1, right - left)),
            "height": int(max(1, bottom - top)),
            "label": f"{title} | {process or 'unknown'} | hwnd={int(hwnd)}",
        }
    except Exception:
        return None


def _list_windows():
    if win32gui is None:
        return []
    windows = []

    def _collector(hwnd, _extra):
        info = _window_info_from_hwnd(hwnd)
        if info:
            windows.append(info)

    win32gui.EnumWindows(_collector, None)
    windows.sort(key=lambda item: (item["process"].lower(), item["title"].lower()))
    return windows


def _resolve_window_target():
    configured = {
        "hwnd": WINDOW_TARGET.get("hwnd"),
        "title": _normalize_text(WINDOW_TARGET.get("title")),
        "process": _normalize_text(WINDOW_TARGET.get("process")).lower(),
    }
    if not (configured["hwnd"] or configured["title"] or configured["process"]):
        return _screen_rect()

    if configured["hwnd"]:
        info = _window_info_from_hwnd(int(configured["hwnd"]))
        if info:
            info["target_found"] = True
            info["target_configured"] = True
            return info

    title_key = configured["title"].lower()
    process_key = configured["process"]
    for item in _list_windows():
        title_ok = True if not title_key else title_key in item["title"].lower()
        process_ok = True if not process_key else process_key in item["process"].lower()
        if title_ok and process_ok:
            item["target_found"] = True
            item["target_configured"] = True
            return item

    fallback = _screen_rect()
    fallback["target_configured"] = True
    return fallback


def _apply_window_target(payload):
    if bool(payload.get("clear")):
        WINDOW_TARGET.update({"hwnd": None, "title": "", "process": ""})
        return _resolve_window_target()

    hwnd = payload.get("hwnd")
    WINDOW_TARGET["hwnd"] = int(hwnd) if str(hwnd or "").strip() else None
    WINDOW_TARGET["title"] = _normalize_text(payload.get("title"))
    WINDOW_TARGET["process"] = _normalize_text(payload.get("process"))
    return _resolve_window_target()


def _capture_screenshot_for_target(target_info):
    region = None
    if target_info.get("target_found"):
        region = (
            int(target_info["x"]),
            int(target_info["y"]),
            int(target_info["width"]),
            int(target_info["height"]),
        )
    try:
        screenshot = pyautogui.screenshot(region=region) if region else pyautogui.screenshot()
    except Exception:
        if mss is None:
            raise
        with mss.mss() as sct:
            if region:
                shot = sct.grab(
                    {
                        "left": int(target_info["x"]),
                        "top": int(target_info["y"]),
                        "width": int(target_info["width"]),
                        "height": int(target_info["height"]),
                    }
                )
            else:
                shot = sct.grab(sct.monitors[1])
            screenshot = Image.frombytes("RGB", shot.size, shot.rgb)
    return screenshot

@app.route('/probe', methods=['GET'])
def probe_endpoint():
    with window_target_lock:
        info = _resolve_window_target()
    return jsonify({"status": "Probe successful", "message": "Service is operational", "window": info}), 200


@app.route('/windows', methods=['GET'])
def list_windows_endpoint():
    return jsonify({"windows": _list_windows()}), 200


@app.route('/window_info', methods=['GET'])
def window_info_endpoint():
    with window_target_lock:
        info = _resolve_window_target()
        target = {
            "hwnd": WINDOW_TARGET.get("hwnd"),
            "title": WINDOW_TARGET.get("title"),
            "process": WINDOW_TARGET.get("process"),
        }
    return jsonify({"target": target, "window": info}), 200


@app.route('/window_target', methods=['POST'])
def set_window_target_endpoint():
    data = request.json or {}
    with window_target_lock:
        info = _apply_window_target(data)
        target = {
            "hwnd": WINDOW_TARGET.get("hwnd"),
            "title": WINDOW_TARGET.get("title"),
            "process": WINDOW_TARGET.get("process"),
        }
    return jsonify({"status": "ok", "target": target, "window": info}), 200

@app.route('/execute', methods=['POST'])
def execute_command():
    # Only execute one command at a time
    with computer_control_lock:
        data = request.json
        return execute_impl(data)

@app.route('/screenshot', methods=['GET'])
def capture_screen_with_cursor():    
    with window_target_lock:
        target_info = _resolve_window_target()
    screenshot = _capture_screenshot_for_target(target_info)
    img_io = BytesIO()
    screenshot.save(img_io, "PNG")
    img_io.seek(0)
    response = send_file(img_io, mimetype="image/png")
    response.headers["X-Window-Offset-X"] = str(int(target_info.get("x", 0)))
    response.headers["X-Window-Offset-Y"] = str(int(target_info.get("y", 0)))
    response.headers["X-Window-Width"] = str(int(target_info.get("width", screenshot.size[0])))
    response.headers["X-Window-Height"] = str(int(target_info.get("height", screenshot.size[1])))
    response.headers["X-Window-Title"] = target_info.get("title", "")
    response.headers["X-Window-Process"] = target_info.get("process", "")
    response.headers["X-Window-Found"] = "1" if target_info.get("target_found") else "0"
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=args.port)
