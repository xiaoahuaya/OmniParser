import base64
import time
import os
import pyautogui
from enum import Enum
import sys
from typing import Literal, TypedDict
from io import BytesIO

# Python 3.10 兼容性: StrEnum 在 3.11+ 才有
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    class StrEnum(str, Enum):
        """Python 3.10 的 StrEnum 兼容实现"""
        def __str__(self):
            return str(self.value)

from PIL import Image

from anthropic.types.beta import BetaToolComputerUse20241022Param

from .base import BaseAnthropicTool, ToolError, ToolResult
from .screen_capture import get_screenshot
import requests
import re

OUTPUT_DIR = "./tmp/outputs"

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50
HOVER_DELAY_SEC = 0.3
DRAG_DURATION_SEC = 0.5
REPEAT_CLICK_DISTANCE = 18
REPEAT_CLICK_LIMIT = 3
PREFLIGHT_ENABLED = os.getenv("OMNITOOL_PREFLIGHT_ENABLED", "1").lower() not in ("0", "false", "no")
PREFLIGHT_REPEAT_INTERVAL = int(os.getenv("OMNITOOL_PREFLIGHT_REPEAT_INTERVAL", "12"))
FOCUS_PROBE_ENABLED = os.getenv("OMNITOOL_FOCUS_PROBE_ENABLED", "1").lower() not in ("0", "false", "no")
FOCUS_PROBE_CHAR = os.getenv("OMNITOOL_FOCUS_PROBE_CHAR", "0")
FOCUS_PROBE_SETTLE_SEC = float(os.getenv("OMNITOOL_FOCUS_PROBE_SETTLE_SEC", "0.08"))
FOCUS_PROBE_ROI_HALF_WIDTH = int(os.getenv("OMNITOOL_FOCUS_PROBE_ROI_HALF_WIDTH", "140"))
FOCUS_PROBE_ROI_HALF_HEIGHT = int(os.getenv("OMNITOOL_FOCUS_PROBE_ROI_HALF_HEIGHT", "90"))
FOCUS_PROBE_INSERT_DIFF_MIN = float(os.getenv("OMNITOOL_FOCUS_PROBE_INSERT_DIFF_MIN", "0.0035"))
FOCUS_PROBE_RESTORE_DIFF_MAX = float(os.getenv("OMNITOOL_FOCUS_PROBE_RESTORE_DIFF_MAX", "0.0018"))

Action = Literal[
    "key",
    "type",
    "type_submit",
    "mouse_move",
    "drag",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
    "hover",
    "wait"
]


class Resolution(TypedDict):
    width: int
    height: int


MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),  # 4:3
    "WXGA": Resolution(width=1280, height=800),  # 16:10
    "FWXGA": Resolution(width=1366, height=768),
    "MAX_FWXGA": Resolution(width=25690, height=1440),

}


class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"


class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None


def chunks(s: str, chunk_size: int) -> list[str]:
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]

class ComputerTool(BaseAnthropicTool):
    """
    A tool that allows the agent to interact with the screen, keyboard, and mouse of the current computer.
    Adapted for Windows using 'pyautogui'.
    """

    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20241022"] = "computer_20241022"
    width: int
    height: int
    display_num: int | None

    _screenshot_delay = 2.0
    _scaling_enabled = True

    @property
    def options(self) -> ComputerToolOptions:
        width, height = self.scale_coordinates(
            ScalingSource.COMPUTER, self.width, self.height
        )
        return {
            "display_width_px": width,
            "display_height_px": height,
            "display_number": self.display_num,
        }

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}

    def __init__(self, is_scaling: bool = False):
        super().__init__()

        # Get screen width and height using Windows command
        self.display_num = None
        self.offset_x = 0
        self.offset_y = 0
        self.is_scaling = is_scaling
        self.windows_host_url = os.getenv("OMNITOOL_WINDOWS_HOST_URL", "").strip()
        self.remote_mode = bool(self.windows_host_url)
        if self.remote_mode and not self.windows_host_url.startswith("http"):
            self.windows_host_url = f"http://{self.windows_host_url}"

        self.width, self.height = self.get_screen_size()
        hint = self._screen_profile_hint()
        if hint:
            print(f"[WARN] {hint}")

        self.key_conversion = {"Page_Down": "pagedown",
                               "Page_Up": "pageup",
                               "Super_L": "win",
                               "Escape": "esc"}
        self._last_mouse_target: tuple[int, int] | None = None
        self._repeat_click_count = 0
        self._preflight_done = False
        self._action_count = 0


    async def __call__(
        self,
        *,
        action: Action,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        start_coordinate: tuple[int, int] | None = None,
        end_coordinate: tuple[int, int] | None = None,
        **kwargs,
    ):
        # Keep runtime output minimal; detailed logs are intentionally omitted.
        self._action_count += 1
        self._run_preflight_if_needed(action)

        if action == "drag":
            if start_coordinate is None or end_coordinate is None:
                raise ToolError("start_coordinate and end_coordinate are required for drag")
            if not isinstance(start_coordinate, (list, tuple)) or len(start_coordinate) != 2:
                raise ToolError(f"{start_coordinate} must be a tuple of length 2")
            if not isinstance(end_coordinate, (list, tuple)) or len(end_coordinate) != 2:
                raise ToolError(f"{end_coordinate} must be a tuple of length 2")
            if not all(isinstance(i, int) for i in start_coordinate):
                raise ToolError(f"{start_coordinate} must be a tuple of ints")
            if not all(isinstance(i, int) for i in end_coordinate):
                raise ToolError(f"{end_coordinate} must be a tuple of ints")

            if self.is_scaling:
                start_x, start_y = self.scale_coordinates(
                    ScalingSource.API, start_coordinate[0], start_coordinate[1]
                )
                end_x, end_y = self.scale_coordinates(
                    ScalingSource.API, end_coordinate[0], end_coordinate[1]
                )
            else:
                start_x, start_y = start_coordinate
                end_x, end_y = end_coordinate

            screen_width, screen_height = pyautogui.size()
            if self.remote_mode:
                screen_width, screen_height = self.get_screen_size()
            start_x = min(max(start_x, 0), screen_width - 1)
            start_y = min(max(start_y, 0), screen_height - 1)
            end_x = min(max(end_x, 0), screen_width - 1)
            end_y = min(max(end_y, 0), screen_height - 1)

            self.send_action(f"pyautogui.moveTo({start_x}, {start_y})")
            self.send_action(f"pyautogui.dragTo({end_x}, {end_y}, duration={DRAG_DURATION_SEC})")
            return ToolResult(output=f"Dragged mouse from ({start_x}, {start_y}) to ({end_x}, {end_y})")

        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
                raise ToolError(f"{coordinate} must be a tuple of length 2")
            # if not all(isinstance(i, int) and i >= 0 for i in coordinate):
            if not all(isinstance(i, int) for i in coordinate):
                raise ToolError(f"{coordinate} must be a tuple of non-negative ints")
            
            if self.is_scaling:
                x, y = self.scale_coordinates(
                    ScalingSource.API, coordinate[0], coordinate[1]
                )
            else:
                x, y = coordinate
            screen_width, screen_height = self.get_screen_size()
            # Ensure x and y are within screen bounds
            x = min(max(x, 0), screen_width - 1)
            y = min(max(y, 0), screen_height - 1)

            # print(f"scaled_coordinates: {x}, {y}")
            # print(f"offset: {self.offset_x}, {self.offset_y}")
            
            # x += self.offset_x # TODO - check if this is needed
            # y += self.offset_y

            if action == "mouse_move":
                self._last_mouse_target = (x, y)
                self.send_action(f"pyautogui.moveTo({x}, {y})")
                return ToolResult(output=f"Moved mouse to ({x}, {y})")
            elif action == "left_click_drag":
                current_x, current_y = self.send_action("pyautogui.position()")
                self.send_action(f"pyautogui.dragTo({x}, {y}, duration=0.5)")
                return ToolResult(output=f"Dragged mouse from ({current_x}, {current_y}) to ({x}, {y})")

        if action in ("key", "type", "type_submit"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ToolError(output=f"{text} must be a string")

            if action == "key":
                # Handle key combinations
                keys = text.split('+')
                for key in keys:
                    key = self.key_conversion.get(key.strip(), key.strip())
                    key = key.lower()
                    self.send_action(f"pyautogui.keyDown('{key}')")  # Press down each key
                for key in reversed(keys):
                    key = self.key_conversion.get(key.strip(), key.strip())
                    key = key.lower()
                    self.send_action(f"pyautogui.keyUp('{key}')")    # Release each key in reverse order
                return ToolResult(output=f"Pressed keys: {text}")
            
            elif action in ("type", "type_submit"):
                clean_text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
                if clean_text:
                    is_url = clean_text.lower().startswith(("http://", "https://"))
                    if not is_url and FOCUS_PROBE_ENABLED:
                        probe_ok, probe_meta = self._probe_focus_with_insert_rollback()
                        if not probe_ok:
                            # One retry after a gentle refocus click at current cursor position.
                            self.send_action("pyautogui.click()")
                            time.sleep(0.06)
                            probe_ok, probe_meta = self._probe_focus_with_insert_rollback()
                        if not probe_ok:
                            raise ToolError(
                                "Focus probe failed before typing. "
                                f"{probe_meta}. "
                                "Please click the target input area once and retry."
                            )
                    if is_url:
                        # Enforce deterministic browser navigation path.
                        self.send_action("pyautogui.hotkey('ctrl', '0')")
                        time.sleep(0.1)
                        self.send_action("pyautogui.hotkey('ctrl', 'l')")
                        time.sleep(0.15)
                        self.send_action("pyautogui.hotkey('ctrl', 'a')")
                        time.sleep(0.05)
                    else:
                        self.send_action("pyautogui.hotkey('ctrl', 'a')")
                        time.sleep(0.05)

                    if self.remote_mode:
                        # Prefer clipboard paste for Chinese/long text; fallback to typing.
                        self._remote_paste_or_type(clean_text)
                    else:
                        # 使用剪贴板复制粘贴，支持中文
                        import pyperclip
                        pyperclip.copy(clean_text)
                        time.sleep(0.1)
                        # Ctrl+V 粘贴
                        self.send_action("pyautogui.keyDown('ctrl')")
                        self.send_action("pyautogui.press('v')")
                        self.send_action("pyautogui.keyUp('ctrl')")
                        time.sleep(0.2)

                if action == "type_submit" or clean_text.lower().startswith(("http://", "https://")):
                    self.send_action("pyautogui.press('enter')")
                screenshot_base64 = (await self.screenshot()).base64_image
                return ToolResult(output=text, base64_image=screenshot_base64)

        if action in (
            "left_click",
            "right_click",
            "double_click",
            "middle_click",
            "screenshot",
            "cursor_position",
            "left_press",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")

            if action == "screenshot":
                return await self.screenshot()
            elif action == "cursor_position":
                x, y = self.send_action("pyautogui.position()")
                x, y = self.scale_coordinates(ScalingSource.COMPUTER, x, y)
                return ToolResult(output=f"X={x},Y={y}")
            else:
                self._guard_repeat_click()
                if action == "left_click":
                    self.send_action("pyautogui.click()")
                elif action == "right_click":
                    self.send_action("pyautogui.rightClick()")
                elif action == "middle_click":
                    self.send_action("pyautogui.middleClick()")
                elif action == "double_click":
                    self.send_action("pyautogui.doubleClick()")
                elif action == "left_press":
                    self.send_action("pyautogui.mouseDown()")
                    time.sleep(1)
                    self.send_action("pyautogui.mouseUp()")
                return ToolResult(output=f"Performed {action}")
        if action in ("scroll_up", "scroll_down"):
            if action == "scroll_up":
                self.send_action("pyautogui.scroll(100)")
            elif action == "scroll_down":
                self.send_action("pyautogui.scroll(-100)")
            return ToolResult(output=f"Performed {action}")
        if action == "hover":
            if coordinate is not None:
                if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
                    raise ToolError(f"{coordinate} must be a tuple of length 2")
                if not all(isinstance(i, int) for i in coordinate):
                    raise ToolError(f"{coordinate} must be a tuple of ints")
                if self.is_scaling:
                    x, y = self.scale_coordinates(
                        ScalingSource.API, coordinate[0], coordinate[1]
                    )
                else:
                    x, y = coordinate
                screen_width, screen_height = self.get_screen_size()
                x = min(max(x, 0), screen_width - 1)
                y = min(max(y, 0), screen_height - 1)
                self.send_action(f"pyautogui.moveTo({x}, {y})")
            time.sleep(HOVER_DELAY_SEC)
            return ToolResult(output=f"Performed {action}")
        if action == "wait":
            time.sleep(1)
            return ToolResult(output=f"Performed {action}")
        raise ToolError(f"Invalid action: {action}")

    # def send_to_vm(self, action: str):
    #     """
    #     Executes a python command on the server. Only return tuple of x,y when action is "pyautogui.position()"
    #     """
    #     prefix = "import pyautogui; pyautogui.FAILSAFE = False;"
    #     command_list = ["python", "-c", f"{prefix} {action}"]
    #     parse = action == "pyautogui.position()"
    #     if parse:
    #         command_list[-1] = f"{prefix} print({action})"
    #
    #     try:
    #         print(f"sending to vm: {command_list}")
    #         response = requests.post(
    #             f"http://localhost:5000/execute",
    #             headers={'Content-Type': 'application/json'},
    #             json={"command": command_list},
    #             timeout=90
    #         )
    #         time.sleep(0.7) # avoid async error as actions take time to complete
    #         print(f"action executed")
    #         if response.status_code != 200:
    #             raise ToolError(f"Failed to execute command. Status code: {response.status_code}")
    #         if parse:
    #             output = response.json()['output'].strip()
    #             match = re.search(r'Point\(x=(\d+),\s*y=(\d+)\)', output)
    #             if not match:
    #                 raise ToolError(f"Could not parse coordinates from output: {output}")
    #             x, y = map(int, match.groups())
    #             return x, y
    #     except requests.exceptions.RequestException as e:
    #         raise ToolError(f"An error occurred while trying to execute the command: {str(e)}")
    #

    def send_to_local(self, action: str):
        """ Executes a python command directly on the local machine. """
        prefix = "import pyautogui; pyautogui.FAILSAFE = False;"
        command = f"{prefix} {action}"

        if action == "pyautogui.position()":
            pos = pyautogui.position()
            return pos
        else:
            exec(command)

    def send_to_vm(self, action: str):
        """Execute a python command on remote Windows host."""
        prefix = "import pyautogui; pyautogui.FAILSAFE = False;"
        parse_position = action == "pyautogui.position()"
        if parse_position:
            command_list = ["python", "-c", f"{prefix} p={action}; print('Point(x=%d, y=%d)' % (p.x, p.y))"]
        else:
            command_list = ["python", "-c", f"{prefix} {action}"]
        try:
            response = requests.post(
                f"{self.windows_host_url}/execute",
                headers={"Content-Type": "application/json"},
                json={"command": command_list},
                timeout=90,
                proxies={"http": "", "https": ""},
            )
            if response.status_code != 200:
                raise ToolError(f"Failed to execute command. Status code: {response.status_code}")
            payload = response.json()
            if payload.get("status") == "error":
                raise ToolError(payload.get("message", "remote execute failed"))
            if int(payload.get("returncode", 0)) != 0:
                err = (payload.get("error") or "").strip()
                raise ToolError(f"Remote command failed: {err[:300]}")
            if parse_position:
                output = (payload.get("output") or "").strip()
                match = re.search(r"Point\(x=(\d+),\s*y=(\d+)\)", output)
                if not match:
                    raise ToolError(f"Could not parse coordinates from output: {output}")
                x, y = map(int, match.groups())
                return x, y
            return None
        except requests.exceptions.RequestException as e:
            raise ToolError(f"An error occurred while trying to execute the command: {str(e)}")

    def send_action(self, action: str):
        if self.remote_mode:
            result = self.send_to_vm(action)
            if result is not None:
                x, y = result
                return x, y
            return None
        return self.send_to_local(action)

    async def screenshot(self):
        if self.remote_mode:
            response = requests.get(
                f"{self.windows_host_url}/screenshot",
                timeout=60,
                proxies={"http": "", "https": ""},
            )
            if response.status_code != 200:
                raise ToolError(f"Failed to capture remote screenshot. Status code: {response.status_code}")
            return ToolResult(base64_image=base64.b64encode(response.content).decode())
        if not hasattr(self, 'target_dimension'):
            self.target_dimension = MAX_SCALING_TARGETS["WXGA"]
        width, height = self.target_dimension["width"], self.target_dimension["height"]
        screenshot, path = get_screenshot(resize=True, target_width=width, target_height=height)
        time.sleep(0.7) # avoid async error as actions take time to complete
        return ToolResult(base64_image=base64.b64encode(path.read_bytes()).decode())

    def padding_image(self, screenshot):
        """Pad the screenshot to 16:10 aspect ratio, when the aspect ratio is not 16:10."""
        _, height = screenshot.size
        new_width = height * 16 // 10

        padding_image = Image.new("RGB", (new_width, height), (255, 255, 255))
        # padding to top left
        padding_image.paste(screenshot, (0, 0))
        return padding_image

    def scale_coordinates(self, source: ScalingSource, x: int, y: int):
        """Scale coordinates to a target maximum resolution."""
        if not self._scaling_enabled:
            return x, y
        # 计算当前屏幕的宽高比
        ratio = self.width / self.height
        target_dimension = None

        # 遍历定义的目标分辨率，找到与当前屏幕宽高比相近的目标分辨率
        for target_name, dimension in MAX_SCALING_TARGETS.items():
            # allow some error in the aspect ratio - not ratios are exactly 16:9
            # 允许一定误差范围来匹配宽高比（例如16:9不是严格精确的，可能是16:10）
            if abs(dimension["width"] / dimension["height"] - ratio) < 0.02:
                # 如果目标分辨率的宽度小于当前屏幕的宽度，选择该目标分辨率作为目标
                if dimension["width"] < self.width:
                    target_dimension = dimension
                    self.target_dimension = target_dimension
                    # print(f"target_dimension: {target_dimension}")
                break

        if target_dimension is None:
            # TODO: currently we force the target to be WXGA (16:10), when it cannot find a match
            target_dimension = MAX_SCALING_TARGETS["WXGA"]
            self.target_dimension = MAX_SCALING_TARGETS["WXGA"]

        # should be less than 1
        x_scaling_factor = target_dimension["width"] / self.width
        y_scaling_factor = target_dimension["height"] / self.height
        if source == ScalingSource.API:
            if x > self.width or y > self.height:
                raise ToolError(f"Coordinates {x}, {y} are out of bounds")
            # scale up
            return round(x / x_scaling_factor), round(y / y_scaling_factor)
        # scale down
        return round(x * x_scaling_factor), round(y * y_scaling_factor)


    def get_local_screen_size(self):
        """ Returns width and height of the local screen """
        import pyautogui
        screen_width, screen_height = pyautogui.size()
        return screen_width, screen_height

    def get_screen_size(self):
        if not self.remote_mode:
            return self.get_local_screen_size()
        try:
            response = requests.post(
                f"{self.windows_host_url}/execute",
                headers={"Content-Type": "application/json"},
                json={"command": ["python", "-c", "import pyautogui; print(pyautogui.size())"]},
                timeout=90,
                proxies={"http": "", "https": ""},
            )
            if response.status_code != 200:
                return self.get_local_screen_size()
            output = (response.json().get("output") or "").strip()
            match = re.search(r"Size\(width=(\d+),\s*height=(\d+)\)", output)
            if not match:
                return 1366, 768
            width, height = map(int, match.groups())
            return width, height
        except Exception:
            return 1366, 768

    def _screen_profile_hint(self) -> str:
        common = {(1024, 768), (1280, 800), (1366, 768), (1920, 1080)}
        if (self.width, self.height) not in common:
            return (
                f"Screen resolution is {self.width}x{self.height}. "
                "For best stability, prefer 1366x768 or 1920x1080 and browser zoom 100%."
            )
        return ""

    def _guard_repeat_click(self):
        if not self._last_mouse_target:
            self._repeat_click_count = 0
            return
        current_x, current_y = self.send_action("pyautogui.position()")
        tx, ty = self._last_mouse_target
        near_target = abs(current_x - tx) <= REPEAT_CLICK_DISTANCE and abs(current_y - ty) <= REPEAT_CLICK_DISTANCE
        if near_target:
            self._repeat_click_count += 1
        else:
            self._repeat_click_count = 0
        if self._repeat_click_count >= REPEAT_CLICK_LIMIT:
            self._repeat_click_count = 0
            raise ToolError(
                "Repeated clicking on nearly the same spot. Switch to keyboard navigation: Ctrl+L then type URL and Enter."
            )

    def _run_preflight_if_needed(self, action: str):
        if not PREFLIGHT_ENABLED:
            return
        # Skip pure observation actions.
        if action in ("screenshot", "cursor_position", "wait"):
            return
        should_repeat = (
            PREFLIGHT_REPEAT_INTERVAL > 0
            and self._preflight_done
            and (self._action_count % PREFLIGHT_REPEAT_INTERVAL == 0)
        )
        if self._preflight_done and not should_repeat:
            return
        # Normalize browser interaction environment for better OCR/coordinate stability.
        try:
            self.send_action("pyautogui.hotkey('win', 'up')")   # maximize active window
            time.sleep(0.1)
            self.send_action("pyautogui.hotkey('alt', 'space')")  # open system menu
            time.sleep(0.05)
            self.send_action("pyautogui.press('x')")              # maximize from menu if needed
            time.sleep(0.1)
            self.send_action("pyautogui.hotkey('ctrl', '0')")   # browser zoom reset to 100%
            time.sleep(0.1)
            self.send_action("pyautogui.press('esc')")          # close overlays/dropdowns
        finally:
            self._preflight_done = True

    def _remote_paste_or_type(self, text: str):
        escaped = self._escape_py_str(text)
        # Try clipboard paste first.
        paste_code = (
            f"import pyperclip, time; "
            f"txt='{escaped}'; "
            f"pyperclip.copy(txt); "
            f"time.sleep(0.08); "
            f"pyautogui.hotkey('ctrl','v')"
        )
        try:
            self.send_action(paste_code)
        except ToolError:
            # Fallback to typing if clipboard package or paste fails.
            self.send_action(f"pyautogui.write('{escaped}', interval=0.01)")

    def _escape_py_str(self, s: str) -> str:
        return (
            s.replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace("\n", "\\n")
        )

    def _probe_focus_with_insert_rollback(self) -> tuple[bool, str]:
        """
        Active focus probe:
        1) Capture baseline frame near current cursor position.
        2) Insert one probe character.
        3) Capture changed frame.
        4) Undo via Ctrl+Z.
        5) Capture restored frame and validate insert/restore diffs.
        """
        probe_char = (FOCUS_PROBE_CHAR or "0")[:1]
        try:
            cx, cy = self.send_action("pyautogui.position()")
            before_b64 = self._capture_probe_frame_b64()
            escaped_probe = self._escape_py_str(probe_char)
            self.send_action(f"pyautogui.write('{escaped_probe}', interval=0.01)")
            time.sleep(FOCUS_PROBE_SETTLE_SEC)
            mid_b64 = self._capture_probe_frame_b64()
            self.send_action("pyautogui.hotkey('ctrl', 'z')")
            time.sleep(FOCUS_PROBE_SETTLE_SEC)
            after_b64 = self._capture_probe_frame_b64()
        except Exception as e:
            return False, f"probe execution error: {str(e)}"

        insert_diff = self._roi_diff_score(before_b64, mid_b64, cx, cy)
        restore_diff = self._roi_diff_score(before_b64, after_b64, cx, cy)
        insert_ok = insert_diff >= FOCUS_PROBE_INSERT_DIFF_MIN
        restore_ok = restore_diff <= FOCUS_PROBE_RESTORE_DIFF_MAX
        ok = insert_ok and restore_ok
        meta = (
            f"insert_diff={insert_diff:.4f} (>= {FOCUS_PROBE_INSERT_DIFF_MIN:.4f}), "
            f"restore_diff={restore_diff:.4f} (<= {FOCUS_PROBE_RESTORE_DIFF_MAX:.4f})"
        )
        return ok, meta

    def _capture_probe_frame_b64(self) -> str:
        if self.remote_mode:
            response = requests.get(
                f"{self.windows_host_url}/screenshot",
                timeout=30,
                proxies={"http": "", "https": ""},
            )
            if response.status_code != 200:
                raise ToolError(f"Probe screenshot failed: HTTP {response.status_code}")
            return base64.b64encode(response.content).decode()
        img = pyautogui.screenshot()
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def _roi_diff_score(self, a_b64: str, b_b64: str, cx: int, cy: int) -> float:
        try:
            img_a = Image.open(BytesIO(base64.b64decode(a_b64))).convert("L")
            img_b = Image.open(BytesIO(base64.b64decode(b_b64))).convert("L")
            w = min(img_a.width, img_b.width)
            h = min(img_a.height, img_b.height)
            left = max(0, min(w - 1, cx - FOCUS_PROBE_ROI_HALF_WIDTH))
            right = max(1, min(w, cx + FOCUS_PROBE_ROI_HALF_WIDTH))
            top = max(0, min(h - 1, cy - FOCUS_PROBE_ROI_HALF_HEIGHT))
            bottom = max(1, min(h, cy + FOCUS_PROBE_ROI_HALF_HEIGHT))
            if right <= left or bottom <= top:
                return 1.0
            roi_a = img_a.crop((left, top, right, bottom))
            roi_b = img_b.crop((left, top, right, bottom))
            pa = list(roi_a.getdata())
            pb = list(roi_b.getdata())
            if not pa or len(pa) != len(pb):
                return 1.0
            total = sum(abs(x - y) for x, y in zip(pa, pb))
            return total / (len(pa) * 255)
        except Exception:
            return 1.0

    # def get_screen_size(self):
    #     """Return width and height of the screen"""
    #     try:
    #         response = requests.post(
    #             f"http://localhost:5000/execute",
    #             headers={'Content-Type': 'application/json'},
    #             json={"command": ["python", "-c", "import pyautogui; print(pyautogui.size())"]},
    #             timeout=90
    #         )
    #         if response.status_code != 200:
    #             raise ToolError(f"Failed to get screen size. Status code: {response.status_code}")
    #
    #         output = response.json()['output'].strip()
    #         match = re.search(r'Size\(width=(\d+),\s*height=(\d+)\)', output)
    #         if not match:
    #             raise ToolError(f"Could not parse screen size from output: {output}")
    #         width, height = map(int, match.groups())
    #         return width, height
    #     except requests.exceptions.RequestException as e:
    #         raise ToolError(f"An error occurred while trying to get screen size: {str(e)}")
