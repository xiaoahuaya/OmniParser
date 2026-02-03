"""
python app.py --windows_host_url localhost:8006 --omniparser_server_url localhost:8000
"""

import os
from datetime import datetime
from enum import Enum
from functools import partial
import sys
import threading
import time
import json
import re

# Python 3.10 兼容性: StrEnum 在 3.11+ 才有
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    class StrEnum(str, Enum):
        """Python 3.10 的 StrEnum 兼容实现"""
        def __str__(self):
            return str(self.value)
from pathlib import Path
from typing import cast
import argparse
import gradio as gr
from anthropic import APIResponse
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from anthropic.types.tool_use_block import ToolUseBlock
from loop import (
    APIProvider,
    sampling_loop_sync,
)
from tools import ToolResult
from llm_config import load_config, save_config, get_provider_config, get_all_providers
from agent.llm_utils.proxy_client import test_proxy_connection, run_proxy_interleaved
from agent.llm_utils.oaiclient import run_oai_interleaved
from agent.llm_utils.groqclient import run_groq_interleaved
import requests
from requests.exceptions import RequestException
import base64

CONFIG_DIR = Path("~/.anthropic").expanduser()
API_KEY_FILE = CONFIG_DIR / "api_key"
DEBUG_LOGS = os.getenv("OMNITOOL_DEBUG", "").lower() in ("1", "true", "yes")

def _debug_print(*args, **kwargs):
    if DEBUG_LOGS:
        print(*args, **kwargs)

TASK_LOCK = threading.Lock()
TASK_STATE = {
    "status": "idle",
    "messages": [],
    "responses": {},
    "tools": {},
    "chatbot_messages": [],
    "stop": False,
    "last_error": None,
    "last_update": None,
    "model": None,
    "provider": None,
    "api_key": None,
    "proxy_base_url": None,
    "proxy_model": None,
    "only_n_most_recent_images": 2,
    "max_steps": None,
    "max_seconds": None,
    "plan": None,
    "plan_steps": None,
    "plan_step_index": 0,
}
TASK_THREAD = None
TASK_STATE_PATH = Path("./tmp/task_state.json")
TASK_STATE_VERSION = 1

INTRO_TEXT = '''
OmniParser 让你可以将任何视觉语言模型转换为 AI 代理。支持 **Codex 中转 / Claude 中转** 以及 OpenAI、DeepSeek、Qwen、Anthropic 等。

输入消息并点击发送开始使用。点击停止暂停，点击垃圾桶图标清除历史。
'''

def _serialize_content_item(item):
    if isinstance(item, (TextBlock, BetaTextBlock)):
        return {"_type": "text_block", "text": item.text}
    if isinstance(item, (ToolUseBlock, BetaToolUseBlock)):
        return {"_type": "tool_use", "name": item.name, "input": item.input}
    return item

def _deserialize_content_item(item):
    if isinstance(item, dict) and item.get("_type") == "text_block":
        return item.get("text", "")
    if isinstance(item, dict) and item.get("_type") == "tool_use":
        name = item.get("name", "tool")
        tool_input = item.get("input", {})
        return f"Tool Use: {name}\nInput: {tool_input}"
    return item

def _serialize_messages(messages):
    serialized = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]
        serialized.append(
            {
                "role": msg.get("role", "user"),
                "content": [_serialize_content_item(c) for c in content],
            }
        )
    return serialized

def _deserialize_messages(messages):
    restored = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]
        restored.append(
            {
                "role": msg.get("role", "user"),
                "content": [_deserialize_content_item(c) for c in content],
            }
        )
    return restored

def _resolve_model_name(model: str, proxy_model: str | None) -> str | None:
    if model == "omniparser + gpt-4o":
        return "gpt-4o-2024-11-20"
    if model == "omniparser + R1":
        return "deepseek-r1-distill-llama-70b"
    if model == "omniparser + qwen2.5vl":
        return "qwen2.5-vl-72b-instruct"
    if model == "omniparser + o1":
        return "o1"
    if model == "omniparser + o3-mini":
        return "o3-mini"
    if model == "omniparser + glm-4.5v":
        return "glm-4.5v"
    if model == "omniparser + glm-4v-plus":
        return "glm-4v-plus-0111"
    if model == "omniparser + glm-4v-flash":
        return "glm-4v-flash"
    if model == "omniparser + glm-4.6":
        return "glm-4.6"
    if model == "omniparser + proxy":
        return proxy_model or "gpt-4o"
    if model == "claude-3-5-sonnet-20241022":
        return None
    return None

_PLAN_STEP_RE = re.compile(
    r"^Step\s*(\d+)\s*:\s*(.*?)\s*\|\s*Success\s*:\s*(.*)$",
    re.IGNORECASE,
)

def _parse_success_groups(success_text: str) -> list[list[str]]:
    if not success_text:
        return []
    groups = re.split(r"[，,;；]+", success_text)
    result: list[list[str]] = []
    for group in groups:
        group = group.strip()
        if not group:
            continue
        alts = [alt.strip() for alt in re.split(r"[|/]", group) if alt.strip()]
        if alts:
            result.append(alts)
    return result

def _parse_plan_steps(plan_text: str | None) -> list[dict]:
    if not plan_text:
        return []
    steps: list[dict] = []
    for raw_line in plan_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = line.lstrip("-*• ").strip()
        match = _PLAN_STEP_RE.match(line)
        if not match:
            continue
        step_num = int(match.group(1))
        action = match.group(2).strip()
        success = match.group(3).strip()
        success_groups = _parse_success_groups(success)
        steps.append(
            {
                "step": step_num,
                "action": action,
                "success": success,
                "success_groups": success_groups,
            }
        )
    return steps

def _generate_plan(task: str, state: dict) -> str | None:
    model = state.get("model")
    provider = state.get("provider")
    api_key = state.get("api_key")
    proxy_base_url = state.get("proxy_base_url")
    proxy_model = state.get("proxy_model")
    model_name = _resolve_model_name(model, proxy_model)
    if not model_name or not api_key:
        return None

    plan_prompt = (
        "请把用户目标拆分成 3-7 个简短步骤。每步给出可观察的成功条件，"
        "成功条件必须是屏幕上可见的文字或控件标签关键词，用逗号分隔。"
        "输出为项目列表，每行格式："
        "Step N: 动作描述 | Success: 关键词1, 关键词2"
        f"\n用户目标：{task}"
    )

    try:
        if model == "omniparser + proxy":
            response, _tokens = run_proxy_interleaved(
                messages=[{"content": [plan_prompt]}],
                system="You are a planning assistant.",
                model_name=model_name,
                api_key=api_key,
                base_url=proxy_base_url,
                max_tokens=512,
                temperature=0,
            )
            return response if isinstance(response, str) else None
        if model == "omniparser + R1":
            response, _tokens = run_groq_interleaved(
                messages=[{"content": [plan_prompt]}],
                system="You are a planning assistant.",
                model_name=model_name,
                api_key=api_key,
                max_tokens=512,
            )
            return response if isinstance(response, str) else None
        if model == "omniparser + qwen2.5vl":
            response, _tokens = run_oai_interleaved(
                messages=[{"content": [plan_prompt]}],
                system="You are a planning assistant.",
                model_name=model_name,
                api_key=api_key,
                max_tokens=512,
                provider_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                temperature=0,
            )
            return response if isinstance(response, str) else None
        if model and model.startswith("omniparser + glm"):
            response, _tokens = run_oai_interleaved(
                messages=[{"content": [plan_prompt]}],
                system="You are a planning assistant.",
                model_name=model_name,
                api_key=api_key,
                max_tokens=512,
                provider_base_url="https://open.bigmodel.cn/api/paas/v4",
                temperature=0,
            )
            return response if isinstance(response, str) else None
        if model and (model.startswith("omniparser + gpt") or model.startswith("omniparser + o")):
            response, _tokens = run_oai_interleaved(
                messages=[{"content": [plan_prompt]}],
                system="You are a planning assistant.",
                model_name=model_name,
                api_key=api_key,
                max_tokens=512,
                provider_base_url="https://api.openai.com/v1",
                temperature=0,
            )
            return response if isinstance(response, str) else None
    except Exception:
        return None
    return None

def _save_task_state():
    TASK_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": TASK_STATE_VERSION,
        "status": TASK_STATE.get("status"),
        "messages": _serialize_messages(TASK_STATE.get("messages", [])),
        "chatbot_messages": TASK_STATE.get("chatbot_messages", []),
        "last_error": TASK_STATE.get("last_error"),
        "last_update": TASK_STATE.get("last_update"),
        "model": TASK_STATE.get("model"),
        "provider": TASK_STATE.get("provider"),
        "api_key": TASK_STATE.get("api_key"),
        "proxy_base_url": TASK_STATE.get("proxy_base_url"),
        "proxy_model": TASK_STATE.get("proxy_model"),
        "only_n_most_recent_images": TASK_STATE.get("only_n_most_recent_images"),
        "max_steps": TASK_STATE.get("max_steps"),
        "max_seconds": TASK_STATE.get("max_seconds"),
        "plan": TASK_STATE.get("plan"),
        "plan_steps": TASK_STATE.get("plan_steps"),
        "plan_step_index": TASK_STATE.get("plan_step_index"),
    }
    TASK_STATE_PATH.write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )

def _load_task_state():
    if not TASK_STATE_PATH.exists():
        return
    try:
        data = json.loads(TASK_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        try:
            data = json.loads(TASK_STATE_PATH.read_text())
        except Exception:
            return
    status = data.get("status", "idle")
    if status in ("running", "starting", "stopping"):
        status = "interrupted"
    TASK_STATE.update(
        {
            "status": status,
            "messages": _deserialize_messages(data.get("messages", [])),
            "chatbot_messages": data.get("chatbot_messages", []),
            "last_error": data.get("last_error"),
            "last_update": data.get("last_update"),
            "model": data.get("model"),
            "provider": data.get("provider"),
            "api_key": data.get("api_key"),
            "proxy_base_url": data.get("proxy_base_url"),
            "proxy_model": data.get("proxy_model"),
            "only_n_most_recent_images": data.get("only_n_most_recent_images", 2),
            "max_steps": data.get("max_steps"),
            "max_seconds": data.get("max_seconds"),
            "plan": data.get("plan"),
            "plan_steps": data.get("plan_steps"),
            "plan_step_index": data.get("plan_step_index", 0),
        }
    )

with TASK_LOCK:
    _load_task_state()

def get_proxy_choices():
    """获取中转 provider 选项"""
    providers = get_all_providers()
    return [(v["name"], k) for k, v in providers.items()]

def get_proxy_models(provider_key: str):
    """获取指定中转 provider 的可用模型"""
    provider = get_provider_config(provider_key)
    if provider:
        return provider.get("available_models", [])
    return []

def test_llm_connection(base_url: str, api_key: str, model: str):
    """
    测试 LLM 连通性
    返回测试结果字符串
    """
    if not base_url or not api_key:
        return "❌ 请先填写 API Base URL 和 API Key"

    try:
        response, tokens = run_proxy_interleaved(
            messages=[{"content": ["Hello, please respond with 'OK' only."]}],
            system="You are a helpful assistant. Respond with only 'OK'.",
            model_name=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=10,
            temperature=0,
        )

        if "错误" in response or "失败" in response or "Error" in response:
            return f"❌ 连接失败: {response}"

        return f"✅ 连接成功!\n模型: {model}\n响应: {response}\nTokens: {tokens}"

    except Exception as e:
        return f"❌ 连接失败: {str(e)}"

def parse_arguments():

    parser = argparse.ArgumentParser(description="Gradio App")
    parser.add_argument("--windows_host_url", type=str, default='localhost:8006')
    parser.add_argument("--omniparser_server_url", type=str, default="localhost:8000")
    parser.add_argument("--local", action="store_true", help="本地模式，直接控制本机桌面")
    parser.add_argument("--max_steps", type=int, default=int(os.getenv("OMNITOOL_MAX_STEPS", "80")))
    parser.add_argument("--max_seconds", type=int, default=int(os.getenv("OMNITOOL_MAX_SECONDS", "900")))
    return parser.parse_args()
args = parse_arguments()


class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"


def setup_state(state):
    if "messages" not in state:
        state["messages"] = []

    llm_config = load_config()
    default_provider_key = llm_config.get("default_provider", "codex_proxy")
    default_provider = llm_config.get("providers", {}).get(default_provider_key, {})

    if "model" not in state:
        state["model"] = "omniparser + proxy"
    if "provider" not in state:
        state["provider"] = default_provider_key
    if "proxy_provider" not in state:
        state["proxy_provider"] = default_provider_key
    if "proxy_base_url" not in state:
        state["proxy_base_url"] = default_provider.get("base_url", "")
    if "proxy_model" not in state:
        state["proxy_model"] = default_provider.get("default_model", "gpt-4o")

    if "openai_api_key" not in state:
        state["openai_api_key"] = os.getenv("OPENAI_API_KEY", "")
    if "anthropic_api_key" not in state:
        state["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY", "")
    if "zhipu_api_key" not in state:
        state["zhipu_api_key"] = "70931c95e7c24296a004b4288da87d79.6lNzdgSQgyLUrMvP"

    if "api_key" not in state:
        state["api_key"] = default_provider.get("api_key", "")

    if "auth_validated" not in state:
        state["auth_validated"] = False
    if "responses" not in state:
        state["responses"] = {}
    if "tools" not in state:
        state["tools"] = {}
    if "only_n_most_recent_images" not in state:
        state["only_n_most_recent_images"] = 2
    if "max_steps" not in state:
        state["max_steps"] = args.max_steps
    if "max_seconds" not in state:
        state["max_seconds"] = args.max_seconds
    if 'chatbot_messages' not in state:
        state['chatbot_messages'] = []
    if 'stop' not in state:
        state['stop'] = False

async def main(state):
    """Render loop for Gradio"""
    setup_state(state)
    return "Setup completed"

def validate_auth(provider: APIProvider, api_key: str | None):
    if provider == APIProvider.ANTHROPIC:
        if not api_key:
            return "Enter your Anthropic API key to continue."
    if provider == APIProvider.BEDROCK:
        import boto3

        if not boto3.Session().get_credentials():
            return "You must have AWS credentials set up to use the Bedrock API."
    if provider == APIProvider.VERTEX:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError

        if not os.environ.get("CLOUD_ML_REGION"):
            return "Set the CLOUD_ML_REGION environment variable to use the Vertex API."
        try:
            google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        except DefaultCredentialsError:
            return "Your google cloud credentials are not set up correctly."

def load_from_storage(filename: str) -> str | None:
    """Load data from a file in the storage directory."""
    try:
        file_path = CONFIG_DIR / filename
        if file_path.exists():
            data = file_path.read_text().strip()
            if data:
                return data
    except Exception as e:
        _debug_print(f"Debug: Error loading {filename}: {e}")
    return None

def save_to_storage(filename: str, data: str) -> None:
    """Save data to a file in the storage directory."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = CONFIG_DIR / filename
        file_path.write_text(data)
        # Ensure only user can read/write the file
        file_path.chmod(0o600)
    except Exception as e:
        _debug_print(f"Debug: Error saving {filename}: {e}")

def _api_response_callback(response: APIResponse[BetaMessage], response_state: dict):
    response_id = datetime.now().isoformat()
    response_state[response_id] = response

def _tool_output_callback(tool_output: ToolResult, tool_id: str, tool_state: dict):
    tool_state[tool_id] = tool_output

def chatbot_output_callback(message, chatbot_state, hide_images=False, sender="bot"):
    def _render_message(message: str | BetaTextBlock | BetaToolUseBlock | ToolResult, hide_images=False):
        _debug_print(f"_render_message: {str(message)[:100]}")
        
        if isinstance(message, str):
            return message
        
        is_tool_result = not isinstance(message, str) and (
            isinstance(message, ToolResult)
            or message.__class__.__name__ == "ToolResult"
        )
        if not message or (
            is_tool_result
            and hide_images
            and not hasattr(message, "error")
            and not hasattr(message, "output")
        ):  # return None if hide_images is True
            return
        # render tool result
        if is_tool_result:
            message = cast(ToolResult, message)
            if message.output:
                return message.output
            if message.error:
                return f"Error: {message.error}"
            if message.base64_image and not hide_images:
                # somehow can't display via gr.Image
                # image_data = base64.b64decode(message.base64_image)
                # return gr.Image(value=Image.open(io.BytesIO(image_data)))
                return f'<img src="data:image/png;base64,{message.base64_image}">'

        elif isinstance(message, BetaTextBlock) or isinstance(message, TextBlock):
            return f"Analysis: {message.text}"
        elif isinstance(message, BetaToolUseBlock) or isinstance(message, ToolUseBlock):
            # return f"Tool Use: {message.name}\nInput: {message.input}"
            return f"Next I will perform the following action: {message.input}"
        else:  
            return message

    def _truncate_string(s, max_length=500):
        """Truncate long strings for concise printing."""
        if isinstance(s, str) and len(s) > max_length:
            return s[:max_length] + "..."
        return s
    # processing Anthropic messages
    message = _render_message(message, hide_images)
    
    if sender == "bot":
        chatbot_state.append((None, message))
    else:
        chatbot_state.append((message, None))
    
    if DEBUG_LOGS:
        concise_state = [(_truncate_string(user_msg), _truncate_string(bot_msg))
                            for user_msg, bot_msg in chatbot_state]
        _debug_print(f"chatbot_output_callback chatbot_state: {concise_state} (truncated)")

STATUS_ICONS = {
    "idle": "⚪",
    "starting": "🟡",
    "running": "🟢",
    "stopping": "🟠",
    "stopped": "🔴",
    "completed": "✅",
    "error": "❌",
    "interrupted": "⚠️",
}

def _task_status_text():
    status = TASK_STATE.get("status", "idle")
    last_error = TASK_STATE.get("last_error")
    plan_steps = TASK_STATE.get("plan_steps") or []
    plan_index = TASK_STATE.get("plan_step_index", 0)

    icon = STATUS_ICONS.get(status, "⚪")

    plan_part = ""
    if plan_steps:
        plan_part = f" | 📋 步骤 {min(plan_index + 1, len(plan_steps))}/{len(plan_steps)}"

    last_update = TASK_STATE.get("last_update")
    updated_part = ""
    if last_update:
        updated_part = f" | 🕐 {datetime.fromtimestamp(last_update).strftime('%H:%M:%S')}"

    messages_count = len(TASK_STATE.get("chatbot_messages", []))
    msg_part = f" | 💬 {messages_count} 条消息" if messages_count > 0 else ""

    if status == "error" and last_error:
        return f"{icon} **状态: 错误** - {last_error}{plan_part}{msg_part}{updated_part}"

    status_text_map = {
        "idle": "空闲",
        "starting": "启动中...",
        "running": "执行中...",
        "stopping": "停止中...",
        "stopped": "已停止",
        "completed": "已完成",
        "interrupted": "已中断",
    }
    status_cn = status_text_map.get(status, status)
    return f"{icon} **状态: {status_cn}**{plan_part}{msg_part}{updated_part}"

def _reset_task_state():
    TASK_STATE.update(
        {
            "status": "idle",
            "messages": [],
            "responses": {},
            "tools": {},
            "chatbot_messages": [],
            "stop": False,
            "last_error": None,
            "last_update": None,
            "plan": None,
            "plan_steps": None,
            "plan_step_index": 0,
        }
    )

def _extract_llm_output(message) -> str:
    """提取 LLM 输出的简洁文本"""
    if isinstance(message, str):
        text = message
    elif hasattr(message, "text"):
        text = message.text
    elif hasattr(message, "input"):
        action = message.input
        if isinstance(action, dict):
            action_type = action.get("action", "unknown")
            coord = action.get("coordinate", "")
            text_input = action.get("text", "")
            if coord:
                return f"[{action_type}] {coord}"
            elif text_input:
                return f"[{action_type}] {text_input[:30]}"
            return f"[{action_type}]"
        return str(action)[:50]
    elif hasattr(message, "output"):
        return f"[result] {str(message.output)[:50]}"
    elif hasattr(message, "error"):
        return f"[error] {str(message.error)[:50]}"
    else:
        text = str(message)

    text = text.replace("\n", " ").strip()
    if len(text) > 80:
        text = text[:80] + "..."
    return text

def _background_output_callback(message, sender="bot", hide_images=False):
    llm_text = _extract_llm_output(message)
    if llm_text and not llm_text.startswith("<img"):
        print(f"  → {llm_text}")

    with TASK_LOCK:
        chatbot_output_callback(
            message, TASK_STATE["chatbot_messages"], hide_images=hide_images, sender=sender
        )
        TASK_STATE["last_update"] = time.time()
        _save_task_state()

def _plan_update_callback(plan_state: dict):
    with TASK_LOCK:
        TASK_STATE["plan_step_index"] = plan_state.get("current_index", 0)
        TASK_STATE["last_update"] = time.time()
        _save_task_state()

def _run_task():
    try:
        with TASK_LOCK:
            messages = TASK_STATE["messages"]
            api_key = TASK_STATE["api_key"]
            provider = TASK_STATE["provider"]
            model = TASK_STATE["model"]
            only_n_images = TASK_STATE["only_n_most_recent_images"]
            proxy_base_url = TASK_STATE.get("proxy_base_url")
            proxy_model = TASK_STATE.get("proxy_model")
            max_steps = TASK_STATE.get("max_steps")
            max_seconds = TASK_STATE.get("max_seconds")
            plan_steps = TASK_STATE.get("plan_steps") or []
            plan_state = None
            if plan_steps:
                plan_state = {"current_index": TASK_STATE.get("plan_step_index", 0)}
            TASK_STATE["status"] = "running"
            TASK_STATE["last_update"] = time.time()
            _save_task_state()

        plan_total = len(plan_steps) if plan_steps else 0
        print(f"[TASK] 🚀 开始 | {model} | 计划:{plan_total}步")

        step_counter = 0
        for loop_msg in sampling_loop_sync(
            model=model,
            provider=provider,
            messages=messages,
            output_callback=partial(_background_output_callback, hide_images=False),
            tool_output_callback=partial(_tool_output_callback, tool_state=TASK_STATE["tools"]),
            api_response_callback=partial(_api_response_callback, response_state=TASK_STATE["responses"]),
            api_key=api_key,
            only_n_most_recent_images=only_n_images,
            max_tokens=16384,
            omniparser_url=args.omniparser_server_url,
            proxy_base_url=proxy_base_url,
            proxy_model=proxy_model,
            max_steps=max_steps,
            max_seconds=max_seconds,
            plan_steps=plan_steps,
            plan_state=plan_state,
            plan_update_callback=_plan_update_callback if plan_state else None,
        ):
            step_counter += 1
            print(f"[{step_counter}]", end=" ", flush=True)

            with TASK_LOCK:
                if TASK_STATE.get("stop"):
                    print(f"\n[TASK] ⏹ 停止")
                    TASK_STATE["status"] = "stopped"
                    _save_task_state()
                    return
            if loop_msg is None:
                break

        print(f"\n[TASK] ✅ 完成 | 共{step_counter}步")
        with TASK_LOCK:
            if TASK_STATE.get("status") != "stopped":
                TASK_STATE["status"] = "completed"
                _save_task_state()
    except Exception as e:
        print(f"\n[TASK] ❌ 错误: {e}")
        with TASK_LOCK:
            TASK_STATE["status"] = "error"
            TASK_STATE["last_error"] = str(e)
            _save_task_state()

def valid_params(user_input, state):
    """Validate all requirements and return a list of error messages."""
    errors = []

    # 本地模式只需要检查 OmniParser Server,不需要 Windows Host
    try:
        url = f'http://{args.omniparser_server_url}/probe'
        response = requests.get(url, timeout=3)
        if response.status_code != 200:
            errors.append(f"OmniParser Server is not responding")
    except RequestException as e:
        errors.append(f"OmniParser Server is not responding. Please start it first: cd omnitool/omniparserserver && python -m omniparserserver")

    if state.get("model") != "omniparser-only" and not state["api_key"].strip():
        errors.append("LLM API Key is not set")

    if not user_input:
        errors.append("no computer use request provided")
    
    return errors

def process_input(user_input, state):
    # Reset the stop flag
    if state["stop"]:
        state["stop"] = False

    errors = valid_params(user_input, state)
    if errors:
        raise gr.Error("Validation errors: " + ", ".join(errors))
    
    # Append the user message to state["messages"]
    state["messages"].append(
        {
            "role": Sender.USER,
            "content": [TextBlock(type="text", text=user_input)],
        }
    )

    # Append the user's message to chatbot_messages with None for the assistant's reply
    state['chatbot_messages'].append((user_input, None))
    yield state['chatbot_messages']  # Yield to update the chatbot UI with the user's message

    _debug_print("=" * 60)
    _debug_print(f"[PROCESS_INPUT] 当前模型: {state.get('model')}")
    _debug_print(f"[PROCESS_INPUT] Provider: {state.get('provider')}")
    _debug_print(f"[PROCESS_INPUT] API Key: {state.get('api_key', '')[:20]}...")
    _debug_print(f"[PROCESS_INPUT] Proxy Base URL: {state.get('proxy_base_url')}")
    _debug_print(f"[PROCESS_INPUT] Proxy Model: {state.get('proxy_model')}")
    _debug_print("=" * 60)

    is_proxy_mode = state.get("model") == "omniparser + proxy"
    proxy_base_url = state.get("proxy_base_url") if is_proxy_mode else None
    proxy_model = state.get("proxy_model") if is_proxy_mode else None

    _debug_print(f"[PROCESS_INPUT] is_proxy_mode: {is_proxy_mode}")
    _debug_print(f"[PROCESS_INPUT] 传递给 loop 的 proxy_base_url: {proxy_base_url}")
    _debug_print(f"[PROCESS_INPUT] 传递给 loop 的 proxy_model: {proxy_model}")

    for loop_msg in sampling_loop_sync(
        model=state["model"],
        provider=state["provider"],
        messages=state["messages"],
        output_callback=partial(chatbot_output_callback, chatbot_state=state['chatbot_messages'], hide_images=False),
        tool_output_callback=partial(_tool_output_callback, tool_state=state["tools"]),
        api_response_callback=partial(_api_response_callback, response_state=state["responses"]),
        api_key=state["api_key"],
        only_n_most_recent_images=state["only_n_most_recent_images"],
        max_tokens=16384,
        omniparser_url=args.omniparser_server_url,
        proxy_base_url=proxy_base_url,
        proxy_model=proxy_model,
        max_steps=state.get("max_steps"),
        max_seconds=state.get("max_seconds"),
    ):  
        if loop_msg is None or state.get("stop"):
            yield state['chatbot_messages']
            _debug_print("End of task. Close the loop.")
            break
            
        yield state['chatbot_messages']  # Yield the updated chatbot_messages to update the chatbot UI

def start_background_task(user_input, state):
    errors = valid_params(user_input, state)
    if errors:
        raise gr.Error("验证错误: " + ", ".join(errors))

    with TASK_LOCK:
        current_status = TASK_STATE.get("status", "idle")
        if current_status == "running":
            raise gr.Error("任务正在运行，请先停止或等待完成")

        print(f"[START] 新任务")

        _reset_task_state()

        TASK_STATE["model"] = state.get("model")
        TASK_STATE["provider"] = state.get("provider")
        TASK_STATE["api_key"] = state.get("api_key")
        TASK_STATE["proxy_base_url"] = state.get("proxy_base_url")
        TASK_STATE["proxy_model"] = state.get("proxy_model")
        TASK_STATE["only_n_most_recent_images"] = state.get("only_n_most_recent_images", 2)
        TASK_STATE["max_steps"] = state.get("max_steps")
        TASK_STATE["max_seconds"] = state.get("max_seconds")

        TASK_STATE["messages"].append(
            {
                "role": Sender.USER,
                "content": [TextBlock(type="text", text=user_input)],
            }
        )
        TASK_STATE["chatbot_messages"].append((user_input, None))

        TASK_STATE["plan"] = None
        TASK_STATE["plan_steps"] = None
        TASK_STATE["plan_step_index"] = 0

        TASK_STATE["status"] = "starting"
        TASK_STATE["last_update"] = time.time()
        _save_task_state()

        global TASK_THREAD
        TASK_THREAD = threading.Thread(target=_run_task, daemon=True)
        TASK_THREAD.start()

        return TASK_STATE["chatbot_messages"], _task_status_text(), ""

def resume_task():
    with TASK_LOCK:
        if TASK_STATE.get("status") == "running":
            return TASK_STATE["chatbot_messages"], _task_status_text()
        if not TASK_STATE.get("messages"):
            raise gr.Error("没有可恢复的任务")
        TASK_STATE["stop"] = False
        TASK_STATE["status"] = "starting"
        TASK_STATE["last_update"] = time.time()
        _save_task_state()

        global TASK_THREAD
        TASK_THREAD = threading.Thread(target=_run_task, daemon=True)
        TASK_THREAD.start()

        return TASK_STATE["chatbot_messages"], _task_status_text()

def refresh_task():
    with TASK_LOCK:
        return TASK_STATE["chatbot_messages"], _task_status_text()

def stop_task():
    with TASK_LOCK:
        TASK_STATE["stop"] = True
        if TASK_STATE.get("status") == "running":
            TASK_STATE["status"] = "stopping"
        _save_task_state()
        return _task_status_text()

def stop_app(state):
    state["stop"] = True
    return "App stopped"

def get_header_image_base64():
    try:
        # Get the absolute path to the image relative to this script
        script_dir = Path(__file__).parent
        image_path = script_dir.parent.parent / "imgs" / "header_bar_thin.png"
        
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            return f'data:image/png;base64,{encoded_string}'
    except Exception as e:
        _debug_print(f"Failed to load header image: {e}")
        return None

with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.HTML("""
        <style>
        .no-padding {
            padding: 0 !important;
        }
        .no-padding > div {
            padding: 0 !important;
        }
        .markdown-text p {
            font-size: 18px;
        }
        </style>
        <script>
        /* 自动滚动 Chatbot 到底部 */
        function scrollChatbotToBottom() {
            const chatbots = document.querySelectorAll('.chatbot, [data-testid="chatbot"]');
            chatbots.forEach(chatbot => {
                const scrollContainer = chatbot.querySelector('.overflow-y-auto, .messages-wrapper, [class*="messages"]');
                if (scrollContainer) {
                    scrollContainer.scrollTop = scrollContainer.scrollHeight;
                }
                /* 备用方案：直接滚动 chatbot 本身 */
                if (chatbot.scrollHeight > chatbot.clientHeight) {
                    chatbot.scrollTop = chatbot.scrollHeight;
                }
            });
        }

        /* 使用 MutationObserver 监听 DOM 变化并自动滚动 */
        const observer = new MutationObserver((mutations) => {
            let shouldScroll = false;
            mutations.forEach(mutation => {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    shouldScroll = true;
                }
            });
            if (shouldScroll) {
                setTimeout(scrollChatbotToBottom, 50);
            }
        });

        /* 页面加载后启动 observer */
        document.addEventListener('DOMContentLoaded', () => {
            setTimeout(() => {
                const chatbots = document.querySelectorAll('.chatbot, [data-testid="chatbot"]');
                chatbots.forEach(chatbot => {
                    observer.observe(chatbot, { childList: true, subtree: true });
                });
                scrollChatbotToBottom();
            }, 1000);
        });

        /* 每次 Gradio 更新后也尝试滚动 */
        setInterval(scrollChatbotToBottom, 2000);
        </script>
    """)
    state = gr.State({})
    
    setup_state(state.value)
    
    header_image = get_header_image_base64()
    if header_image:
        gr.HTML(f'<img src="{header_image}" alt="OmniTool Header" width="100%">', elem_classes="no-padding")
        gr.HTML('<h1 style="text-align: center; font-weight: normal;">Omni<span style="font-weight: bold;">Tool</span></h1>')
    else:
        gr.Markdown("# OmniTool")

    if not os.getenv("HIDE_WARNING", False):
        gr.Markdown(INTRO_TEXT, elem_classes="markdown-text")


    llm_config = load_config()
    default_provider_key = llm_config.get("default_provider", "codex_proxy")
    default_provider = llm_config.get("providers", {}).get(default_provider_key, {})

    with gr.Accordion("中转 API 设置", open=True):
        with gr.Row():
            with gr.Column():
                model = gr.Dropdown(
                    label="Model",
                    choices=[
                        "omniparser + proxy",
                        "omniparser-only",
                        "omniparser + glm-4.6",
                        "omniparser + gpt-4o",
                        "claude-3-5-sonnet-20241022",
                    ],
                    value="omniparser + proxy",
                    interactive=True,
                )
        with gr.Row():
            with gr.Column():
                proxy_provider = gr.Dropdown(
                    label="选择中转服务",
                    choices=get_proxy_choices(),
                    value=default_provider_key,
                    interactive=True,
                )
            with gr.Column():
                proxy_model = gr.Dropdown(
                    label="选择模型",
                    choices=default_provider.get("available_models", []),
                    value=default_provider.get("default_model", "gpt-4o"),
                    interactive=True,
                )
        with gr.Row():
            with gr.Column():
                proxy_base_url = gr.Textbox(
                    label="API Base URL",
                    value=default_provider.get("base_url", ""),
                    interactive=True,
                )
            with gr.Column():
                api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    value=default_provider.get("api_key", ""),
                    placeholder="输入 API Key",
                    interactive=True,
                )
        with gr.Row():
            with gr.Column():
                only_n_images = gr.Slider(
                    label="保留最近截图数量",
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=2,
                    interactive=True
                )
            with gr.Column():
                test_btn = gr.Button(
                    value="🔌 测试连接",
                    variant="secondary",
                    size="sm",
                )
        with gr.Row():
            test_result = gr.Textbox(
                label="连接测试结果",
                value="",
                interactive=False,
                lines=3,
                visible=True,
            )
    provider = gr.Dropdown(
        label="API Provider",
        choices=[option.value for option in APIProvider],
        value="codex_proxy",
        interactive=False,
        visible=False,
    )

    with gr.Row():
        with gr.Column(scale=8):
            chat_input = gr.Textbox(show_label=False, placeholder="Type a message to send to Omniparser + X ...", container=False)
        with gr.Column(scale=1, min_width=50):
            submit_button = gr.Button(value="Send", variant="primary")
        with gr.Column(scale=1, min_width=50):
            stop_button = gr.Button(value="Stop", variant="secondary")
        with gr.Column(scale=1, min_width=60):
            resume_button = gr.Button(value="Resume", variant="secondary")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chatbot History", autoscroll=True, height=580, type="tuples", allow_tags=True)
        with gr.Column(scale=3):
            if args.local:
                local_info = gr.HTML(
                    '''
                    <div style="height: 580px; display: flex; flex-direction: column; align-items: center; justify-content: center; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 8px; color: white;">
                        <h2 style="margin-bottom: 20px;">🖥️ 本地控制模式</h2>
                        <p style="color: #888; margin-bottom: 10px;">AI 将直接控制您的电脑桌面</p>
                        <p style="color: #666; font-size: 12px;">请确保重要工作已保存</p>
                        <div style="margin-top: 30px; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 8px;">
                            <p style="color: #4ade80; margin: 0;">✅ 本地模式已启用</p>
                        </div>
                    </div>
                    ''',
                    container=False,
                    elem_classes="no-padding"
                )
            else:
                iframe = gr.HTML(
                    f'<iframe src="http://{args.windows_host_url}/vnc.html?view_only=1&autoconnect=1&resize=scale" width="100%" height="580" allow="fullscreen"></iframe>',
                    container=False,
                    elem_classes="no-padding"
                )

    status_bar = gr.Markdown("Status: idle")

    def update_model(model_selection, state):
        old_model = state.get("model", "未设置")
        state["model"] = model_selection
        _debug_print("=" * 60)
        _debug_print(f"[UPDATE_MODEL] 模型切换: {old_model} → {model_selection}")
        _debug_print(f"[UPDATE_MODEL] state['model'] 已更新为: {state['model']}")
        _debug_print("=" * 60)

        if model_selection == "claude-3-5-sonnet-20241022":
            provider_choices = [option.value for option in APIProvider if option.value != "openai"]
        elif model_selection in set(["omniparser + gpt-4o", "omniparser + o1", "omniparser + o3-mini", "omniparser + gpt-4o-orchestrated", "omniparser + o1-orchestrated", "omniparser + o3-mini-orchestrated"]):
            provider_choices = ["openai"]
        elif model_selection == "omniparser + R1":
            provider_choices = ["groq"]
        elif model_selection == "omniparser + qwen2.5vl":
            provider_choices = ["dashscope"]
        elif model_selection in set(["omniparser + glm-4.5v", "omniparser + glm-4v-plus", "omniparser + glm-4v-flash", "omniparser + glm-4.6"]):
            provider_choices = ["zhipu"]
        elif model_selection == "omniparser + proxy":
            provider_choices = [state.get("proxy_provider", "codex_proxy")]
        else:
            provider_choices = [option.value for option in APIProvider]
        default_provider_value = provider_choices[0]

        provider_interactive = len(provider_choices) > 1
        api_key_placeholder = f"{default_provider_value.title()} API Key"

        # Update state
        state["provider"] = default_provider_value
        if model_selection == "omniparser + proxy":
            proxy_config = get_provider_config(state.get("proxy_provider", "codex_proxy"))
            state["api_key"] = proxy_config.get("api_key", "")
        else:
            state["api_key"] = state.get(f"{default_provider_value}_api_key", "")

        # Calls to update other components UI
        provider_update = gr.update(
            choices=provider_choices,
            value=default_provider_value,
            interactive=provider_interactive
        )
        api_key_update = gr.update(
            placeholder=api_key_placeholder,
            value=state["api_key"]
        )

        return provider_update, api_key_update

    def update_only_n_images(only_n_images_value, state):
        state["only_n_most_recent_images"] = only_n_images_value
   
    def update_provider(provider_value, state):
        # Update state
        state["provider"] = provider_value
        state["api_key"] = state.get(f"{provider_value}_api_key", "")
        
        # Calls to update other components UI
        api_key_update = gr.update(
            placeholder=f"{provider_value.title()} API Key",
            value=state["api_key"]
        )
        return api_key_update
                
    def update_api_key(api_key_value, state):
        state["api_key"] = api_key_value
        state[f'{state["provider"]}_api_key'] = api_key_value

    def update_proxy_provider(proxy_provider_value, state):
        """切换中转 provider 时更新配置"""
        provider_config = get_provider_config(proxy_provider_value)
        if provider_config:
            state["proxy_provider"] = proxy_provider_value
            state["proxy_base_url"] = provider_config.get("base_url", "")
            state["api_key"] = provider_config.get("api_key", "")
            state["proxy_model"] = provider_config.get("default_model", "gpt-4o")
            state["provider"] = proxy_provider_value

            return (
                gr.update(choices=provider_config.get("available_models", []),
                         value=provider_config.get("default_model", "gpt-4o")),
                gr.update(value=provider_config.get("base_url", "")),
                gr.update(value=provider_config.get("api_key", "")),
            )
        return gr.update(), gr.update(), gr.update()

    def update_proxy_model(proxy_model_value, state):
        """切换模型时更新 state"""
        state["proxy_model"] = proxy_model_value

    def update_proxy_base_url(base_url_value, state):
        """更新 base_url"""
        state["proxy_base_url"] = base_url_value

    def do_test_connection(base_url, api_key_value, model, state):
        """执行连接测试"""
        result = test_llm_connection(base_url, api_key_value, model)
        return result

    def clear_chat(state):
        # Reset message-related state
        state["messages"] = []
        state["responses"] = {}
        state["tools"] = {}
        state['chatbot_messages'] = []
        with TASK_LOCK:
            _reset_task_state()
            _save_task_state()
        return state['chatbot_messages'], _task_status_text()

    proxy_provider.change(
        fn=update_proxy_provider,
        inputs=[proxy_provider, state],
        outputs=[proxy_model, proxy_base_url, api_key]
    )
    proxy_model.change(fn=update_proxy_model, inputs=[proxy_model, state], outputs=None)
    proxy_base_url.change(fn=update_proxy_base_url, inputs=[proxy_base_url, state], outputs=None)

    test_btn.click(
        fn=do_test_connection,
        inputs=[proxy_base_url, api_key, proxy_model, state],
        outputs=[test_result]
    )

    model.change(fn=update_model, inputs=[model, state], outputs=[provider, api_key])
    only_n_images.change(fn=update_only_n_images, inputs=[only_n_images, state], outputs=None)
    provider.change(fn=update_provider, inputs=[provider, state], outputs=api_key)
    api_key.change(fn=update_api_key, inputs=[api_key, state], outputs=None)
    chatbot.clear(fn=clear_chat, inputs=[state], outputs=[chatbot, status_bar])

    submit_button.click(start_background_task, [chat_input, state], [chatbot, status_bar, chat_input])
    stop_button.click(stop_task, [], [status_bar])
    resume_button.click(resume_task, [], [chatbot, status_bar])

    auto_refresh_timer = gr.Timer(value=1.5)
    auto_refresh_timer.tick(refresh_task, [], [chatbot, status_bar])
    
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7888)
