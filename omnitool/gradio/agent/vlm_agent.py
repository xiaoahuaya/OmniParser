import json
import os
from collections.abc import Callable
from typing import cast, Callable
import uuid
from PIL import Image, ImageDraw
import base64
from io import BytesIO

from anthropic import APIResponse
from anthropic.types import ToolResultBlockParam
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock, BetaMessageParam, BetaUsage

from agent.llm_utils.oaiclient import run_oai_interleaved
from agent.llm_utils.groqclient import run_groq_interleaved
from agent.llm_utils.proxy_client import run_proxy_interleaved
from agent.llm_utils.utils import is_image_path
import time
import re

OUTPUT_DIR = "./tmp/outputs"
LLM_ERROR_PATTERNS = (
    "请求失败",
    "connection aborted",
    "connection reset",
    "timed out",
    "timeout",
    "error",
    "错误",
    "失败",
)
DEBUG_LOGS = os.getenv("OMNITOOL_DEBUG", "").lower() in ("1", "true", "yes")

def _debug_print(*args, **kwargs):
    if DEBUG_LOGS:
        print(*args, **kwargs)

def extract_data(input_string, data_type):
    # Regular expression to extract content starting from '```python' until the end if there are no closing backticks
    pattern = f"```{data_type}" + r"(.*?)(```|$)"
    # Extract content
    # re.DOTALL allows '.' to match newlines as well
    matches = re.findall(pattern, input_string, re.DOTALL)
    # Return the first match if exists, trimming whitespace and ignoring potential closing backticks
    return matches[0][0].strip() if matches else input_string

class VLMAgent:
    def __init__(
        self,
        model: str,
        provider: str,
        api_key: str,
        output_callback: Callable,
        api_response_callback: Callable,
        max_tokens: int = 4096,
        only_n_most_recent_images: int | None = None,
        print_usage: bool = True,
        proxy_base_url: str = None,
        proxy_model: str = None,
    ):
        self.proxy_base_url = proxy_base_url
        self.use_proxy = proxy_base_url is not None
        self.proxy_model = proxy_model

        if model == "omniparser + gpt-4o":
            self.model = "gpt-4o-2024-11-20"
        elif model == "omniparser + R1":
            self.model = "deepseek-r1-distill-llama-70b"
        elif model == "omniparser + qwen2.5vl":
            self.model = "qwen2.5-vl-72b-instruct"
        elif model == "omniparser + o1":
            self.model = "o1"
        elif model == "omniparser + o3-mini":
            self.model = "o3-mini"
        elif model == "omniparser + glm-4.5v":
            self.model = "glm-4.5v"
        elif model == "omniparser + glm-4v-plus":
            self.model = "glm-4v-plus-0111"
        elif model == "omniparser + glm-4v-flash":
            self.model = "glm-4v-flash"
        elif model == "omniparser + glm-4.6":
            self.model = "glm-4.6"
        elif model == "omniparser + proxy":
            self.model = proxy_model if proxy_model else "gpt-4o"
        else:
            raise ValueError(f"Model {model} not supported")
        

        self.provider = provider
        self.api_key = api_key
        self.api_response_callback = api_response_callback
        self.max_tokens = max_tokens
        self.only_n_most_recent_images = only_n_most_recent_images
        self.output_callback = output_callback

        self.print_usage = print_usage
        self.total_token_usage = 0
        self.total_cost = 0
        self.step_count = 0

        self.system = ''
        self.retry_limit = int(os.getenv("OMNITOOL_LLM_RETRY_LIMIT", "3"))
        self.retry_backoff = float(os.getenv("OMNITOOL_LLM_RETRY_BACKOFF", "2.0"))
        self.debug = DEBUG_LOGS

    def _should_retry_response(self, response: object) -> bool:
        if not isinstance(response, str):
            return True
        lower = response.lower()
        return any(pattern in lower for pattern in LLM_ERROR_PATTERNS)
           
    def __call__(self, messages: list, parsed_screen: list[str, list, dict]):
        self.step_count += 1
        image_base64 = parsed_screen['original_screenshot_base64']
        latency_omniparser = parsed_screen['latency']
        self.output_callback(f'**🔄 Step {self.step_count}**', sender="bot")
        screen_info = str(parsed_screen['screen_info'])
        screenshot_uuid = parsed_screen['screenshot_uuid']
        screen_width, screen_height = parsed_screen['width'], parsed_screen['height']

        boxids_and_labels = parsed_screen["screen_info"]
        system = self._get_system_prompt(boxids_and_labels)

        # drop looping actions msg, byte image etc
        planner_messages = messages
        _remove_som_images(planner_messages)
        _maybe_filter_to_n_most_recent_images(planner_messages, self.only_n_most_recent_images)

        if isinstance(planner_messages[-1], dict):
            if not isinstance(planner_messages[-1]["content"], list):
                planner_messages[-1]["content"] = [planner_messages[-1]["content"]]
            planner_messages[-1]["content"].append(f"{OUTPUT_DIR}/screenshot_{screenshot_uuid}.png")
            planner_messages[-1]["content"].append(f"{OUTPUT_DIR}/screenshot_som_{screenshot_uuid}.png")

        last_error = None
        vlm_response = None
        token_usage = 0
        latency_vlm = 0.0
        for attempt in range(self.retry_limit):
            start = time.time()
            try:
                if self.use_proxy:
                    vlm_response, token_usage = run_proxy_interleaved(
                        messages=planner_messages,
                        system=system,
                        model_name=self.model,
                        api_key=self.api_key,
                        base_url=self.proxy_base_url,
                        max_tokens=self.max_tokens,
                        temperature=0,
                    )
                    self.total_token_usage += token_usage
                    self.total_cost += (token_usage * 2.5 / 1000000)
                elif "gpt" in self.model or "o1" in self.model or "o3-mini" in self.model:
                    vlm_response, token_usage = run_oai_interleaved(
                        messages=planner_messages,
                        system=system,
                        model_name=self.model,
                        api_key=self.api_key,
                        max_tokens=self.max_tokens,
                        provider_base_url="https://api.openai.com/v1",
                        temperature=0,
                    )
                    self.total_token_usage += token_usage
                    if 'gpt' in self.model:
                        self.total_cost += (token_usage * 2.5 / 1000000)
                    elif 'o1' in self.model:
                        self.total_cost += (token_usage * 15 / 1000000)
                    elif 'o3-mini' in self.model:
                        self.total_cost += (token_usage * 1.1 / 1000000)
                elif "r1" in self.model:
                    vlm_response, token_usage = run_groq_interleaved(
                        messages=planner_messages,
                        system=system,
                        model_name=self.model,
                        api_key=self.api_key,
                        max_tokens=self.max_tokens,
                    )
                    self.total_token_usage += token_usage
                    self.total_cost += (token_usage * 0.99 / 1000000)
                elif "qwen" in self.model:
                    vlm_response, token_usage = run_oai_interleaved(
                        messages=planner_messages,
                        system=system,
                        model_name=self.model,
                        api_key=self.api_key,
                        max_tokens=min(2048, self.max_tokens),
                        provider_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        temperature=0,
                    )
                    self.total_token_usage += token_usage
                    self.total_cost += (token_usage * 2.2 / 1000000)  # https://help.aliyun.com/zh/model-studio/getting-started/models?spm=a2c4g.11186623.0.0.74b04823CGnPv7#fe96cfb1a422a
                elif "glm" in self.model:
                    vlm_response, token_usage = run_oai_interleaved(
                        messages=planner_messages,
                        system=system,
                        model_name=self.model,
                        api_key=self.api_key,
                        max_tokens=self.max_tokens,
                        provider_base_url="https://open.bigmodel.cn/api/paas/v4",
                        temperature=0.6,
                    )
                    self.total_token_usage += token_usage
                    # GLM-4.5V: 输入¥0.05/千tokens, 输出¥0.05/千tokens (平均)
                    # GLM-4V-Plus: 输入¥0.01/千tokens, 输出¥0.01/千tokens
                    # GLM-4V-Flash: 免费
                    if "4.5v" in self.model:
                        self.total_cost += (token_usage * 0.05 / 1000)
                    elif "plus" in self.model:
                        self.total_cost += (token_usage * 0.01 / 1000)
                    # flash 免费,不计费
                else:
                    raise ValueError(f"Model {self.model} not supported")
            except Exception as e:
                last_error = str(e)
                time.sleep(self.retry_backoff * (attempt + 1))
                continue

            latency_vlm = time.time() - start
            if self._should_retry_response(vlm_response):
                last_error = str(vlm_response)[:200]
                time.sleep(self.retry_backoff * (attempt + 1))
                continue

            vlm_response_json = extract_data(vlm_response, "json")
            try:
                vlm_response_json = json.loads(vlm_response_json)
                break
            except json.JSONDecodeError as e:
                last_error = f"JSON parse failed: {e}"
                time.sleep(self.retry_backoff * (attempt + 1))
                continue
        else:
            raise RuntimeError(f"LLM request failed after {self.retry_limit} attempts: {last_error}")

        if self.debug:
            self.output_callback(
                f"LLM: {latency_vlm:.2f}s, OmniParser: {latency_omniparser:.2f}s",
                sender="bot",
            )

        # Keep console output minimal; detailed responses are shown in UI.

        img_to_show_base64 = parsed_screen["som_image_base64"]
        if "Box ID" in vlm_response_json:
            try:
                bbox = parsed_screen["parsed_content_list"][int(vlm_response_json["Box ID"])]["bbox"]
                vlm_response_json["box_centroid_coordinate"] = [int((bbox[0] + bbox[2]) / 2 * screen_width), int((bbox[1] + bbox[3]) / 2 * screen_height)]
                img_to_show_data = base64.b64decode(img_to_show_base64)
                img_to_show = Image.open(BytesIO(img_to_show_data))

                draw = ImageDraw.Draw(img_to_show)
                x, y = vlm_response_json["box_centroid_coordinate"] 
                radius = 10
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red')
                draw.ellipse((x - radius*3, y - radius*3, x + radius*3, y + radius*3), fill=None, outline='red', width=2)

                buffered = BytesIO()
                img_to_show.save(buffered, format="PNG")
                img_to_show_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            except:
                _debug_print(f"Error parsing: {vlm_response_json}")
                pass
        self.output_callback(f'<img src="data:image/png;base64,{img_to_show_base64}">', sender="bot")

        reasoning = vlm_response_json.get("Reasoning", "")
        if reasoning:
            self.output_callback(f"💭 {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}", sender="bot")

        if self.debug:
            self.output_callback(
                        f'<details>'
                        f'  <summary>Parsed Screen elemetns by OmniParser</summary>'
                        f'  <pre>{screen_info}</pre>'
                        f'</details>',
                        sender="bot"
                    )
        vlm_plan_str = ""
        for key, value in vlm_response_json.items():
            if key == "Reasoning":
                vlm_plan_str += f'{value}'
            else:
                vlm_plan_str += f'\n{key}: {value}'

        # construct the response so that anthropicExcutor can execute the tool
        response_content = [BetaTextBlock(text=vlm_plan_str, type='text')]

        next_action = vlm_response_json.get("Next Action", "None")
        box_id = vlm_response_json.get("Box ID", "N/A")
        coordinate = vlm_response_json.get("box_centroid_coordinate", None)
        drag_from_id = vlm_response_json.get("From Box ID")
        drag_to_id = vlm_response_json.get("To Box ID")
        drag_start_coordinate = None
        drag_end_coordinate = None

        if drag_from_id is not None and drag_to_id is not None:
            try:
                from_bbox = parsed_screen["parsed_content_list"][int(drag_from_id)]["bbox"]
                to_bbox = parsed_screen["parsed_content_list"][int(drag_to_id)]["bbox"]
                drag_start_coordinate = [
                    int((from_bbox[0] + from_bbox[2]) / 2 * screen_width),
                    int((from_bbox[1] + from_bbox[3]) / 2 * screen_height),
                ]
                drag_end_coordinate = [
                    int((to_bbox[0] + to_bbox[2]) / 2 * screen_width),
                    int((to_bbox[1] + to_bbox[3]) / 2 * screen_height),
                ]
                vlm_response_json["drag_start_coordinate"] = drag_start_coordinate
                vlm_response_json["drag_end_coordinate"] = drag_end_coordinate
            except Exception as e:
                _debug_print(f"[WARN] Failed to parse drag coordinates: {e}")

        if 'box_centroid_coordinate' in vlm_response_json and next_action in (
            "left_click", "right_click", "double_click", "type", "hover"
        ):
            move_cursor_block = BetaToolUseBlock(
                id=f'toolu_{uuid.uuid4()}',
                input={'action': 'mouse_move', 'coordinate': vlm_response_json["box_centroid_coordinate"]},
                name='computer',
                type='tool_use'
            )
            response_content.append(move_cursor_block)

            # type 动作前自动点击输入框
            if next_action == "type":
                click_block = BetaToolUseBlock(
                    id=f'toolu_{uuid.uuid4()}',
                    input={'action': 'left_click'},
                    name='computer',
                    type='tool_use'
                )
                response_content.append(click_block)

        if next_action == "None":
            action_info = f"🏁 **任务完成/暂停**"
        elif next_action == "type":
            type_value = vlm_response_json.get("value", "")

            # type 动作：输入文本
            sim_content_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                        input={'action': 'type', 'text': type_value},
                                        name='computer', type='tool_use')
            response_content.append(sim_content_block)

            # 自动追加 key enter 发送
            enter_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                        input={'action': 'key', 'text': 'enter'},
                                        name='computer', type='tool_use')
            response_content.append(enter_block)

            action_info = f"⌨️ **输入并发送**: `{type_value}` (Box ID: {box_id})"
        elif next_action == "drag":
            if not drag_start_coordinate or not drag_end_coordinate:
                action_info = "⚠️ **拖拽缺少起止坐标，已暂停**"
            else:
                sim_content_block = BetaToolUseBlock(
                    id=f'toolu_{uuid.uuid4()}',
                    input={
                        'action': next_action,
                        'start_coordinate': drag_start_coordinate,
                        'end_coordinate': drag_end_coordinate,
                    },
                    name='computer',
                    type='tool_use'
                )
                response_content.append(sim_content_block)
                action_info = f"🖱️ **drag** | From Box ID: {drag_from_id} → To Box ID: {drag_to_id}"
        elif next_action == "key":
            key_value = vlm_response_json.get("value", "")
            sim_content_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                        input={'action': next_action, 'text': key_value},
                                        name='computer', type='tool_use')
            response_content.append(sim_content_block)
            action_info = f"⌨️ **快捷键**: `{key_value}`"
        else:
            sim_content_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                            input={'action': next_action},
                                            name='computer', type='tool_use')
            response_content.append(sim_content_block)
            coord_str = f"({coordinate[0]}, {coordinate[1]})" if coordinate else "N/A"
            action_info = f"🖱️ **{next_action}** | Box ID: {box_id} | 坐标: {coord_str}"

        self.output_callback(f"**📋 动作:** {action_info}", sender="bot")
        response_message = BetaMessage(id=f'toolu_{uuid.uuid4()}', content=response_content, model='', role='assistant', type='message', stop_reason='tool_use', usage=BetaUsage(input_tokens=0, output_tokens=0))
        return response_message, vlm_response_json

    def _api_response_callback(self, response: APIResponse):
        self.api_response_callback(response)

    def _get_system_prompt(self, screen_info: str = ""):
        main_section = f"""
你是一个图像标注助手。你的唯一任务是：分析截图，输出JSON告诉程序点击哪个元素。

【重要】你不操作电脑！你只看图片、输出JSON。真正的点击由Python脚本执行，与你无关。

当前屏幕上检测到的UI元素列表：
{screen_info}

根据用户任务和截图，告诉程序下一步动作：点击/输入用 Box ID，拖拽用 From Box ID + To Box ID。

可用动作：left_click, right_click, double_click, drag, type, key, scroll_up, scroll_down, wait, None

动作说明：
- type: 输入文字并自动按回车发送（无需额外操作）
- key: 按快捷键（如 enter, ctrl+a, ctrl+c 等）
- drag: 从一个元素拖拽到另一个元素（需要 From Box ID 和 To Box ID）

【重要】：type 动作会自动发送，不需要再用 key enter 或点击发送按钮

只输出JSON，格式如下：
```json
{{
    "Reasoning": "简述当前屏幕内容和下一步思路",
    "Next Action": "动作类型",
    "Box ID": 元素ID数字,
    "From Box ID": 起点元素ID,  # 仅drag需要
    "To Box ID": 终点元素ID,    # 仅drag需要
    "value": "仅type/key动作需要"
}}
```

示例1 - 点击：
```json
{{
    "Reasoning": "屏幕显示百度搜索结果，需要点击第一个链接",
    "Next Action": "left_click",
    "Box ID": 5
}}
```

示例2 - 输入：
```json
{{
    "Reasoning": "屏幕显示搜索框，需要输入搜索词",
    "Next Action": "type",
    "Box ID": 3,
    "value": "今天天气"
}}
```

示例3 - 键盘快捷键（全选）：
```json
{{
    "Reasoning": "搜索框已获得焦点但包含旧内容，需要先全选再覆盖输入",
    "Next Action": "key",
    "value": "ctrl+a"
}}
```

示例4 - 任务完成：
```json
{{
    "Reasoning": "已完成用户要求的任务",
    "Next Action": "None"
}}
```

规则：
1. 每次只输出一个动作
2. 任务完成时 Next Action 填 "None"
3. 遇到登录页/验证码时 Next Action 填 "None"
4. 不要重复选择同一个元素

【重要 - 输入框规则】：
- 点击输入框后，不要再次点击验证焦点，直接使用 type 输入文字
- 不要通过视觉判断输入框是否"获得焦点"，因为很多输入框获得焦点后外观不变
- 正确流程：点击输入框 → 下一步直接 type 输入内容
- 错误流程：点击输入框 → 再次点击验证 → 再点击...（禁止）
- 如果输入后文字没有出现在屏幕上，才需要重新点击输入框
"""
        return main_section

def _remove_som_images(messages):
    for msg in messages:
        msg_content = msg["content"]
        if isinstance(msg_content, list):
            msg["content"] = [
                cnt for cnt in msg_content 
                if not (isinstance(cnt, str) and 'som' in cnt and is_image_path(cnt))
            ]


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int = 10,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place
    """
    if images_to_keep is None:
        return messages

    total_images = 0
    for msg in messages:
        for cnt in msg.get("content", []):
            if isinstance(cnt, str) and is_image_path(cnt):
                total_images += 1
            elif isinstance(cnt, dict) and cnt.get("type") == "tool_result":
                for content in cnt.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "image":
                        total_images += 1

    images_to_remove = total_images - images_to_keep
    
    for msg in messages:
        msg_content = msg["content"]
        if isinstance(msg_content, list):
            new_content = []
            for cnt in msg_content:
                # Remove images from SOM or screenshot as needed
                if isinstance(cnt, str) and is_image_path(cnt):
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                # VLM shouldn't use anthropic screenshot tool so shouldn't have these but in case it does, remove as needed
                elif isinstance(cnt, dict) and cnt.get("type") == "tool_result":
                    new_tool_result_content = []
                    for tool_result_entry in cnt.get("content", []):
                        if isinstance(tool_result_entry, dict) and tool_result_entry.get("type") == "image":
                            if images_to_remove > 0:
                                images_to_remove -= 1
                                continue
                        new_tool_result_content.append(tool_result_entry)
                    cnt["content"] = new_tool_result_content
                # Append fixed content to current message's content list
                new_content.append(cnt)
            msg["content"] = new_content
