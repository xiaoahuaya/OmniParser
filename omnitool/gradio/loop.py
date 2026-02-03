"""
Agentic sampling loop that calls the Anthropic API and local implenmentation of anthropic-defined computer use tools.
"""
from collections.abc import Callable
import base64
from io import BytesIO
from PIL import Image
import time
from enum import Enum
import sys

# Python 3.10 兼容性: StrEnum 在 3.11+ 才有
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    class StrEnum(str, Enum):
        """Python 3.10 的 StrEnum 兼容实现"""
        def __str__(self):
            return str(self.value)

from anthropic import APIResponse
from anthropic.types import (
    TextBlock,
)
from anthropic.types.beta import (
    BetaContentBlock,
    BetaMessage,
    BetaMessageParam
)
from tools import ToolResult

from agent.llm_utils.omniparserclient import OmniParserClient
from agent.anthropic_agent import AnthropicActor
from agent.vlm_agent import VLMAgent
from agent.vlm_agent_with_orchestrator import VLMOrchestratedAgent
from executor.anthropic_executor import AnthropicExecutor

BETA_FLAG = "computer-use-2024-10-22"

SCREEN_DIFF_SIZE = 32
SCREEN_DIFF_THRESHOLD = 0.015
NO_CHANGE_LIMIT = 2
NO_CHANGE_HINT = (
    "The previous action did not change the screen. Try a different approach "
    "(double-click, click a slightly different target, scroll, or wait longer)."
)

REPEAT_COORD_THRESHOLD = 50
REPEAT_COORD_LIMIT = 3
REPEAT_COORD_HINT = (
    "⚠️ 检测到连续 {count} 次点击相同位置 {coord}，但屏幕没有变化。"
    "请尝试完全不同的方法：滚动页面、点击其他元素、使用键盘快捷键、或等待页面加载。"
    "如果目标元素不可交互，请跳过此步骤。"
)

PLAN_STEP_RETRY_LIMIT = 5
PLAN_STEP_SKIP_MSG = "⚠️ 当前步骤重试 {count} 次未成功，自动跳过到下一步。"

OPTIMISTIC_ACTION_LIMIT = 1
OPTIMISTIC_ADVANCE_MSG = "⏩ 动作已执行但未检测到成功条件，乐观前进尝试下一步。"
OPTIMISTIC_CONSECUTIVE_LIMIT = 3
OPTIMISTIC_STOP_MSG = "⚠️ 连续 {count} 步乐观前进均未检测到成功，任务可能出问题，请检查。"

def _is_similar_coord(coord1: list, coord2: list, threshold: int = REPEAT_COORD_THRESHOLD) -> bool:
    """判断两个坐标是否相近"""
    if not coord1 or not coord2:
        return False
    return abs(coord1[0] - coord2[0]) < threshold and abs(coord1[1] - coord2[1]) < threshold

def _check_repeat_coords(recent_coords: list, new_coord: list) -> int:
    """检查新坐标是否与最近的坐标重复，返回重复次数"""
    if not new_coord or not recent_coords:
        return 0
    count = 0
    for coord in reversed(recent_coords):
        if _is_similar_coord(coord, new_coord):
            count += 1
        else:
            break
    return count

def _downsample_gray_pixels(image_b64: str, size: int = SCREEN_DIFF_SIZE) -> list[int]:
    image_bytes = base64.b64decode(image_b64)
    img = Image.open(BytesIO(image_bytes)).convert("L").resize((size, size))
    return list(img.getdata())

def _mean_abs_diff(pixels_a: list[int], pixels_b: list[int]) -> float:
    if not pixels_a or not pixels_b or len(pixels_a) != len(pixels_b):
        return 1.0
    total = sum(abs(a - b) for a, b in zip(pixels_a, pixels_b))
    return total / (len(pixels_a) * 255)

def _screen_change_score(prev_b64: str, curr_b64: str) -> float:
    try:
        prev_pixels = _downsample_gray_pixels(prev_b64)
        curr_pixels = _downsample_gray_pixels(curr_b64)
        return _mean_abs_diff(prev_pixels, curr_pixels)
    except Exception as e:
        print(f"[WARN] Screen diff failed: {e}")
        return 1.0

def _matches_success(screen_info: str, success_groups: list[list[str]]) -> bool:
    if not screen_info or not success_groups:
        return False
    screen_lower = screen_info.lower()
    for group in success_groups:
        if not any(alt.lower() in screen_lower for alt in group if alt):
            return False
    return True

def _advance_plan_if_ready(
    *,
    parsed_screen: dict,
    messages: list,
    output_callback: Callable[[BetaContentBlock], None],
    plan_steps: list[dict] | None,
    plan_state: dict | None,
    plan_update_callback: Callable[[dict], None] | None,
):
    if not plan_steps or not plan_state:
        return
    screen_info = parsed_screen.get("screen_info", "")
    current_index = plan_state.get("current_index", 0)
    progressed = False

    while current_index < len(plan_steps):
        step = plan_steps[current_index]
        success_groups = step.get("success_groups") or []
        if not success_groups:
            break
        if not _matches_success(screen_info, success_groups):
            break
        step_num = step.get("step", current_index + 1)
        success_text = step.get("success", "")
        progress_msg = f"✅ Step {step_num} 已完成，成功条件已匹配：{success_text}"
        messages.append({"role": "assistant", "content": progress_msg})
        output_callback(progress_msg)
        current_index += 1
        progressed = True

    if progressed:
        plan_state["current_index"] = current_index
        if plan_update_callback:
            plan_update_callback(plan_state)
        if current_index < len(plan_steps):
            next_step = plan_steps[current_index]
            next_msg = (
                f"➡️ 下一步: Step {next_step.get('step', current_index + 1)} - "
                f"{next_step.get('action', '')} | Success: {next_step.get('success', '')}"
            )
            messages.append({"role": "assistant", "content": next_msg})
            output_callback(next_msg)
        else:
            done_msg = "✅ 所有计划步骤已完成。若目标已达成，请输出 Next Action: None 结束任务。"
            messages.append({"role": "assistant", "content": done_msg})
            output_callback(done_msg)
            if plan_update_callback:
                plan_update_callback(plan_state)

class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    OPENAI = "openai"
    ZHIPU = "zhipu"
    CODEX_PROXY = "codex_proxy"
    CLAUDE_PROXY = "claude_proxy"


PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
    APIProvider.OPENAI: "gpt-4o",
    APIProvider.ZHIPU: "glm-4.5v",
    APIProvider.CODEX_PROXY: "gpt-4o",
    APIProvider.CLAUDE_PROXY: "claude-3-5-sonnet-20241022",
}

def sampling_loop_sync(
    *,
    model: str,
    provider: APIProvider | None,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlock], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[[APIResponse[BetaMessage]], None],
    api_key: str,
    only_n_most_recent_images: int | None = 2,
    max_tokens: int = 4096,
    omniparser_url: str,
    save_folder: str = "./uploads",
    proxy_base_url: str = None,
    proxy_model: str = None,
    max_steps: int | None = None,
    max_seconds: int | None = None,
    plan_steps: list[dict] | None = None,
    plan_state: dict | None = None,
    plan_update_callback: Callable[[dict], None] | None = None,
):
    """
    Synchronous agentic sampling loop for the assistant/tool interaction of computer use.
    """
    # Keep console output minimal; detailed progress is shown in UI.
    prev_screen_b64 = None
    no_change_count = 0
    step_count = 0
    start_time = time.time()
    omniparser_client = OmniParserClient(url=f"http://{omniparser_url}/parse/")
    if model == "claude-3-5-sonnet-20241022":
        # Register Actor and Executor
        actor = AnthropicActor(
            model=model, 
            provider=provider,
            api_key=api_key, 
            api_response_callback=api_response_callback,
            max_tokens=max_tokens,
            only_n_most_recent_images=only_n_most_recent_images
        )
    elif model in set(["omniparser + gpt-4o", "omniparser + o1", "omniparser + o3-mini", "omniparser + R1", "omniparser + qwen2.5vl", "omniparser + glm-4.5v", "omniparser + glm-4v-plus", "omniparser + glm-4v-flash", "omniparser + glm-4.6", "omniparser + proxy"]):
        actor = VLMAgent(
            model=model,
            provider=provider,
            api_key=api_key,
            api_response_callback=api_response_callback,
            output_callback=output_callback,
            max_tokens=max_tokens,
            only_n_most_recent_images=only_n_most_recent_images,
            proxy_base_url=proxy_base_url,
            proxy_model=proxy_model,
        )
    elif model in set(["omniparser + gpt-4o-orchestrated", "omniparser + o1-orchestrated", "omniparser + o3-mini-orchestrated", "omniparser + R1-orchestrated", "omniparser + qwen2.5vl-orchestrated"]):
        actor = VLMOrchestratedAgent(
            model=model,
            provider=provider,
            api_key=api_key,
            api_response_callback=api_response_callback,
            output_callback=output_callback,
            max_tokens=max_tokens,
            only_n_most_recent_images=only_n_most_recent_images,
            save_folder=save_folder
        )
    elif model == "omniparser-only":
        parsed_screen = omniparser_client()

        screen_width = parsed_screen['width']
        screen_height = parsed_screen['height']
        elements = parsed_screen['parsed_content_list']

        result_text = f"""
## 截图解析完成

**屏幕尺寸**: {screen_width} x {screen_height}
**解析耗时**: {parsed_screen['latency']:.2f}s
**检测到的 UI 元素数量**: {len(elements)}

### UI 元素列表 (前20个):
```
{parsed_screen['screen_info'][:2000]}
```
"""
        output_callback(result_text)

        som_img_html = f'<img src="data:image/png;base64,{parsed_screen["som_image_base64"]}" style="max-width:100%;">'
        output_callback(som_img_html)

        yield result_text
        return messages
    else:
        raise ValueError(f"Model {model} not supported")

    executor = AnthropicExecutor(
        output_callback=output_callback,
        tool_output_callback=tool_output_callback,
    )
    tool_result_content = None

    if model == "claude-3-5-sonnet-20241022": # Anthropic loop
        while True:
            if max_steps is not None and step_count >= max_steps:
                output_callback(f"⚠️ 已达到最大步数 {max_steps}，任务停止。")
                return messages
            if max_seconds is not None and time.time() - start_time >= max_seconds:
                output_callback(f"⚠️ 已达到最大运行时间 {max_seconds} 秒，任务停止。")
                return messages
            step_count += 1
            parsed_screen = omniparser_client() # parsed_screen: {"som_image_base64": dino_labled_img, "parsed_content_list": parsed_content_list, "screen_info"}
            current_b64 = parsed_screen.get("original_screenshot_base64")
            if prev_screen_b64 and current_b64:
                diff_score = _screen_change_score(prev_screen_b64, current_b64)
                if diff_score < SCREEN_DIFF_THRESHOLD:
                    no_change_count += 1
                else:
                    no_change_count = 0
                if no_change_count >= NO_CHANGE_LIMIT:
                    messages.append({"role": "user", "content": NO_CHANGE_HINT})
                    no_change_count = 0
            prev_screen_b64 = current_b64
            _advance_plan_if_ready(
                parsed_screen=parsed_screen,
                messages=messages,
                output_callback=output_callback,
                plan_steps=plan_steps,
                plan_state=plan_state,
                plan_update_callback=plan_update_callback,
            )
            screen_info_block = TextBlock(text='Below is the structured accessibility information of the current UI screen, which includes text and icons you can operate on, take these information into account when you are making the prediction for the next action. Note you will still need to take screenshot to get the image: \n' + parsed_screen['screen_info'], type='text')
            screen_info_dict = {"role": "user", "content": [screen_info_block]}
            messages.append(screen_info_dict)
            tools_use_needed = actor(messages=messages)

            for message, tool_result_content in executor(tools_use_needed, messages):
                yield message
        
            if not tool_result_content:
                return messages

            messages.append({"content": tool_result_content, "role": "user"})
    
    elif model in set(["omniparser + gpt-4o", "omniparser + o1", "omniparser + o3-mini", "omniparser + R1", "omniparser + qwen2.5vl", "omniparser + glm-4.5v", "omniparser + glm-4v-plus", "omniparser + glm-4v-flash", "omniparser + glm-4.6", "omniparser + gpt-4o-orchestrated", "omniparser + o1-orchestrated", "omniparser + o3-mini-orchestrated", "omniparser + R1-orchestrated", "omniparser + qwen2.5vl-orchestrated", "omniparser + proxy"]):
        recent_coords = []

        while True:
            if max_steps is not None and step_count >= max_steps:
                output_callback(f"⚠️ 已达到最大步数 {max_steps}，任务停止。")
                return messages
            if max_seconds is not None and time.time() - start_time >= max_seconds:
                output_callback(f"⚠️ 已达到最大运行时间 {max_seconds} 秒，任务停止。")
                return messages
            step_count += 1
            parsed_screen = omniparser_client()
            current_b64 = parsed_screen.get("original_screenshot_base64")
            if prev_screen_b64 and current_b64:
                diff_score = _screen_change_score(prev_screen_b64, current_b64)
                if diff_score < SCREEN_DIFF_THRESHOLD:
                    no_change_count += 1
                else:
                    no_change_count = 0
                if no_change_count >= NO_CHANGE_LIMIT:
                    messages.append({"role": "user", "content": NO_CHANGE_HINT})
                    no_change_count = 0
            prev_screen_b64 = current_b64

            tools_use_needed, vlm_response_json = actor(messages=messages, parsed_screen=parsed_screen)

            new_coord = vlm_response_json.get("box_centroid_coordinate") if vlm_response_json else None
            if new_coord:
                repeat_count = _check_repeat_coords(recent_coords, new_coord)
                if repeat_count >= REPEAT_COORD_LIMIT - 1:
                    hint = REPEAT_COORD_HINT.format(count=repeat_count + 1, coord=new_coord)
                    output_callback(hint)
                    messages.append({"role": "user", "content": hint})
                recent_coords.append(new_coord)
                if len(recent_coords) > 10:
                    recent_coords.pop(0)

            for message, tool_result_content in executor(tools_use_needed, messages):
                yield message

            if not tool_result_content:
                return messages

            messages.append({"content": tool_result_content, "role": "user"})
