"""
Agentic sampling loop that calls the Anthropic API and local implenmentation of anthropic-defined computer use tools.
"""
from collections.abc import Callable
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
    BetaMessageParam,
)
from tools import ToolResult

from agent.llm_utils.omniparserclient import OmniParserClient
from agent.anthropic_agent import AnthropicActor
from agent.vlm_agent import VLMAgent
from agent.vlm_agent_with_orchestrator import VLMOrchestratedAgent
from executor.anthropic_executor import AnthropicExecutor

BETA_FLAG = "computer-use-2024-10-22"

from loop_helpers import (
    NAVIGATION_RECOVERY_HINT,
    NO_CHANGE_HINT,
    NO_CHANGE_LIMIT,
    REPEAT_COORD_HINT,
    REPEAT_COORD_LIMIT,
    REPEAT_COORD_SCROLL_RECOVERY_HINT,
    SCROLL_KEY_FALLBACK_HINT,
    SCREEN_DIFF_THRESHOLD,
    VIEWPORT_RECOVERY_HINT,
    action_gate_passed as _action_gate_passed,
    advance_plan_if_ready as _advance_plan_if_ready,
    blocked_editor_exit_action as _blocked_editor_exit_action,
    build_action_gate_recovery_hint as _build_action_gate_recovery_hint,
    check_repeat_coords as _check_repeat_coords,
    forced_key_tool_response as _forced_key_tool_response,
    forced_scroll_tool_response as _forced_scroll_tool_response,
    is_click_like_action as _is_click_like_action,
    is_maximize_action as _is_maximize_action,
    is_scroll_like_action as _is_scroll_like_action,
    is_xhs_feed_or_search_context as _is_xhs_feed_or_search_context,
    needs_viewport_recover as _needs_viewport_recover,
    required_field_violation as _required_field_violation,
    screen_change_score as _screen_change_score,
    topic_anchor_evidence as _topic_anchor_evidence,
)

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
    windows_host_url: str | None = None,
    capture_output_dir: str | None = None,
    save_folder: str = "./uploads",
    proxy_base_url: str = None,
    proxy_model: str = None,
    max_steps: int | None = None,
    max_seconds: int | None = None,
    plan_steps: list[dict] | None = None,
    plan_state: dict | None = None,
    plan_update_callback: Callable[[dict], None] | None = None,
    topic_anchor_term: str | None = None,
    topic_anchor_callback: Callable[[dict], None] | None = None,
):
    """
    Synchronous agentic sampling loop for the assistant/tool interaction of computer use.
    """
    # Keep console output minimal; detailed progress is shown in UI.
    prev_screen_b64 = None
    no_change_count = 0
    step_count = 0
    start_time = time.time()
    omniparser_client = OmniParserClient(
        url=f"http://{omniparser_url}/parse/",
        windows_host_url=windows_host_url,
        output_dir=capture_output_dir or "./tmp/outputs",
    )
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
        windows_host_url=windows_host_url,
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
        last_recovery_hint_step = -1
        last_viewport_hint_step = -1
        last_scroll_fallback_step = -1
        anchor_announced = False

        while True:
            if max_steps is not None and step_count >= max_steps:
                output_callback(f"⚠️ 已达到最大步数 {max_steps}，任务停止。")
                return messages
            if max_seconds is not None and time.time() - start_time >= max_seconds:
                output_callback(f"⚠️ 已达到最大运行时间 {max_seconds} 秒，任务停止。")
                return messages
            step_count += 1
            parsed_screen = omniparser_client()
            pre_action_screen = parsed_screen
            if topic_anchor_term:
                anchor_evidence = _topic_anchor_evidence(parsed_screen, topic_anchor_term)
                if topic_anchor_callback:
                    topic_anchor_callback(anchor_evidence)
                if anchor_evidence.get("anchored") and (not anchor_announced):
                    anchor_msg = (
                        f"🎯 页面证据锚定完成: topic={anchor_evidence.get('topic')} "
                        f"input={int(bool(anchor_evidence.get('input_hit')))} "
                        f"results={int(bool(anchor_evidence.get('results_hit')))}"
                    )
                    messages.append({"role": "assistant", "content": anchor_msg})
                    output_callback(anchor_msg)
                    anchor_announced = True
            screen_info_lower = (parsed_screen.get("screen_info", "") or "").lower()
            if any(k in screen_info_lower for k in ["bing", "challenge", "验证", "captcha", "最后一步"]):
                if step_count - last_recovery_hint_step >= 2:
                    messages.append({"role": "user", "content": NAVIGATION_RECOVERY_HINT})
                    output_callback(NAVIGATION_RECOVERY_HINT)
                    last_recovery_hint_step = step_count
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
            required_field_error = _required_field_violation(vlm_response_json or {}, parsed_screen)
            if required_field_error:
                output_callback(required_field_error)
                messages.append({"role": "user", "content": required_field_error})
                continue
            blocked_click_error = _blocked_editor_exit_action(vlm_response_json or {}, parsed_screen)
            if blocked_click_error:
                output_callback(blocked_click_error)
                messages.append({"role": "user", "content": blocked_click_error})
                continue
            if _needs_viewport_recover(parsed_screen) and not _is_maximize_action(vlm_response_json or {}):
                if step_count - last_viewport_hint_step >= 2:
                    output_callback(VIEWPORT_RECOVERY_HINT)
                    messages.append({"role": "user", "content": VIEWPORT_RECOVERY_HINT})
                    last_viewport_hint_step = step_count
                continue

            new_coord = vlm_response_json.get("box_centroid_coordinate") if vlm_response_json else None
            if new_coord:
                repeat_count = _check_repeat_coords(recent_coords, new_coord)
                if repeat_count >= REPEAT_COORD_LIMIT - 1:
                    hint = REPEAT_COORD_HINT.format(count=repeat_count + 1, coord=new_coord)
                    output_callback(hint)
                    messages.append({"role": "user", "content": hint})
                    if _is_click_like_action(vlm_response_json or {}) and _is_xhs_feed_or_search_context(parsed_screen):
                        output_callback(REPEAT_COORD_SCROLL_RECOVERY_HINT)
                        messages.append({"role": "user", "content": REPEAT_COORD_SCROLL_RECOVERY_HINT})
                        # Hard recovery: force one scroll action instead of only提示，避免持续点击同一区域。
                        tools_use_needed = _forced_scroll_tool_response()
                        vlm_response_json = {"Next Action": "scroll_down", "value": "", "Box ID": None}
                        recent_coords.append(new_coord)
                        if len(recent_coords) > 10:
                            recent_coords.pop(0)
                recent_coords.append(new_coord)
                if len(recent_coords) > 10:
                    recent_coords.pop(0)

            for message, tool_result_content in executor(tools_use_needed, messages):
                yield message

            if not tool_result_content:
                return messages

            messages.append({"content": tool_result_content, "role": "user"})

            # Action gate: only continue normally when at least one signal changes.
            post_action_screen = omniparser_client()
            if topic_anchor_term:
                anchor_evidence = _topic_anchor_evidence(post_action_screen, topic_anchor_term)
                if topic_anchor_callback:
                    topic_anchor_callback(anchor_evidence)
                if anchor_evidence.get("anchored") and (not anchor_announced):
                    anchor_msg = (
                        f"🎯 页面证据锚定完成: topic={anchor_evidence.get('topic')} "
                        f"input={int(bool(anchor_evidence.get('input_hit')))} "
                        f"results={int(bool(anchor_evidence.get('results_hit')))}"
                    )
                    messages.append({"role": "assistant", "content": anchor_msg})
                    output_callback(anchor_msg)
                    anchor_announced = True
            gate_passed, gate_reason = _action_gate_passed(
                pre_action_screen,
                post_action_screen,
                vlm_response_json or {},
            )
            if not gate_passed:
                recovery_hint = _build_action_gate_recovery_hint(vlm_response_json or {}, pre_action_screen)
                recovery_msg = f"{recovery_hint}\n[{gate_reason}]"
                messages.append({"role": "user", "content": recovery_msg})
                output_callback(recovery_msg)
                if _is_scroll_like_action(vlm_response_json or {}) and (step_count - last_scroll_fallback_step >= 2):
                    next_action = str((vlm_response_json or {}).get("Next Action", "") or "").lower().strip()
                    fallback_key = "pagedown" if next_action != "scroll_up" else "pageup"
                    output_callback(SCROLL_KEY_FALLBACK_HINT)
                    messages.append({"role": "user", "content": SCROLL_KEY_FALLBACK_HINT})
                    forced_response = _forced_key_tool_response(fallback_key)
                    forced_tool_result_content = None
                    for message, forced_tool_result_content in executor(forced_response, messages):
                        yield message
                    if not forced_tool_result_content:
                        return messages
                    messages.append({"content": forced_tool_result_content, "role": "user"})
                    post_action_screen = omniparser_client()
                    prev_screen_b64 = post_action_screen.get("original_screenshot_base64")
                    last_scroll_fallback_step = step_count
                    continue
            # Keep latest frame as previous baseline.
            prev_screen_b64 = post_action_screen.get("original_screenshot_base64")

