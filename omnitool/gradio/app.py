"""
python app.py --windows_host_url localhost:8006 --omniparser_server_url localhost:8000
"""

import os
from datetime import datetime
from enum import Enum
from functools import partial
import sys

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
import requests
from requests.exceptions import RequestException
import base64

CONFIG_DIR = Path("~/.anthropic").expanduser()
API_KEY_FILE = CONFIG_DIR / "api_key"

INTRO_TEXT = '''
OmniParser 让你可以将任何视觉语言模型转换为 AI 代理。支持 **Codex 中转 / Claude 中转** 以及 OpenAI、DeepSeek、Qwen、Anthropic 等。

输入消息并点击发送开始使用。点击停止暂停，点击垃圾桶图标清除历史。
'''

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
        print(f"Debug: Error loading {filename}: {e}")
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
        print(f"Debug: Error saving {filename}: {e}")

def _api_response_callback(response: APIResponse[BetaMessage], response_state: dict):
    response_id = datetime.now().isoformat()
    response_state[response_id] = response

def _tool_output_callback(tool_output: ToolResult, tool_id: str, tool_state: dict):
    tool_state[tool_id] = tool_output

def chatbot_output_callback(message, chatbot_state, hide_images=False, sender="bot"):
    def _render_message(message: str | BetaTextBlock | BetaToolUseBlock | ToolResult, hide_images=False):
    
        print(f"_render_message: {str(message)[:100]}")
        
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
    
    # Create a concise version of the chatbot state for printing
    concise_state = [(_truncate_string(user_msg), _truncate_string(bot_msg))
                        for user_msg, bot_msg in chatbot_state]
    # print(f"chatbot_output_callback chatbot_state: {concise_state} (truncated)")

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

    print("=" * 60)
    print(f"[PROCESS_INPUT] 当前模型: {state.get('model')}")
    print(f"[PROCESS_INPUT] Provider: {state.get('provider')}")
    print(f"[PROCESS_INPUT] API Key: {state.get('api_key', '')[:20]}...")
    print(f"[PROCESS_INPUT] Proxy Base URL: {state.get('proxy_base_url')}")
    print(f"[PROCESS_INPUT] Proxy Model: {state.get('proxy_model')}")
    print("=" * 60)

    is_proxy_mode = state.get("model") == "omniparser + proxy"
    proxy_base_url = state.get("proxy_base_url") if is_proxy_mode else None
    proxy_model = state.get("proxy_model") if is_proxy_mode else None

    print(f"[PROCESS_INPUT] is_proxy_mode: {is_proxy_mode}")
    print(f"[PROCESS_INPUT] 传递给 loop 的 proxy_base_url: {proxy_base_url}")
    print(f"[PROCESS_INPUT] 传递给 loop 的 proxy_model: {proxy_model}")

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
    ):  
        if loop_msg is None or state.get("stop"):
            yield state['chatbot_messages']
            print("End of task. Close the loop.")
            break
            
        yield state['chatbot_messages']  # Yield the updated chatbot_messages to update the chatbot UI

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
        print(f"Failed to load header image: {e}")
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
            font-size: 18px;  /* Adjust the font size as needed */
        }
        </style>
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

    with gr.Accordion("原始设置（高级）", open=False):
        with gr.Row():
            with gr.Column():
                model = gr.Dropdown(
                    label="Model",
                    choices=["omniparser + proxy", "omniparser-only", "omniparser + glm-4.6", "omniparser + gpt-4o", "claude-3-5-sonnet-20241022"],
                    value="omniparser + proxy",
                    interactive=True,
                )
            with gr.Column():
                provider = gr.Dropdown(
                    label="API Provider",
                    choices=[option.value for option in APIProvider],
                    value="codex_proxy",
                    interactive=True,
                )

    with gr.Row():
        with gr.Column(scale=8):
            chat_input = gr.Textbox(show_label=False, placeholder="Type a message to send to Omniparser + X ...", container=False)
        with gr.Column(scale=1, min_width=50):
            submit_button = gr.Button(value="Send", variant="primary")
        with gr.Column(scale=1, min_width=50):
            stop_button = gr.Button(value="Stop", variant="secondary")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chatbot History", autoscroll=True, height=580)
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

    def update_model(model_selection, state):
        old_model = state.get("model", "未设置")
        state["model"] = model_selection
        print("=" * 60)
        print(f"[UPDATE_MODEL] 模型切换: {old_model} → {model_selection}")
        print(f"[UPDATE_MODEL] state['model'] 已更新为: {state['model']}")
        print("=" * 60)

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
        return state['chatbot_messages']

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
    chatbot.clear(fn=clear_chat, inputs=[state], outputs=[chatbot])

    submit_button.click(process_input, [chat_input, state], chatbot)
    stop_button.click(stop_app, [state], None)
    
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7888)