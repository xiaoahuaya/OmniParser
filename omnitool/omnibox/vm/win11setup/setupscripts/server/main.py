import os
import logging
import argparse
import shlex
import subprocess
import contextlib
import io as pyio
from flask import Flask, request, jsonify, send_file
import threading
import traceback
import pyautogui
from PIL import Image
from io import BytesIO
import mss


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

@app.route('/probe', methods=['GET'])
def probe_endpoint():
    return jsonify({"status": "Probe successful", "message": "Service is operational"}), 200

@app.route('/execute', methods=['POST'])
def execute_command():
    # Only execute one command at a time
    with computer_control_lock:
        data = request.json
        return execute_impl(data)

@app.route('/screenshot', methods=['GET'])
def capture_screen_with_cursor():    
    # Keep screenshot endpoint robust in varied desktop sessions.
    try:
        screenshot = pyautogui.screenshot()
    except Exception:
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            shot = sct.grab(monitor)
            screenshot = Image.frombytes("RGB", shot.size, shot.rgb)
    img_io = BytesIO()
    screenshot.save(img_io, "PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=args.port)
