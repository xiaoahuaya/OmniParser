"""
Agentic sampling loop that calls the Anthropic API and local implenmentation of anthropic-defined computer use tools.
"""
from collections.abc import Callable
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
):
    """
    Synchronous agentic sampling loop for the assistant/tool interaction of computer use.
    """
    print('in sampling_loop_sync, model:', model)
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
        print("[INFO] OmniParser-only mode: 仅解析截图，不调用 LLM")
        parsed_screen = omniparser_client()

        screen_width = parsed_screen['width']
        screen_height = parsed_screen['height']
        elements = parsed_screen['parsed_content_list']

        print("=" * 60)
        print("[PARSED ELEMENTS] 解析到的 UI 元素:")
        print(f"屏幕尺寸: {screen_width} x {screen_height}")
        print(f"元素数量: {len(elements)}")
        print("-" * 60)

        for elem in elements[:20]:
            idx = elem.get('idx', elem.get('id', '?'))
            elem_type = elem.get('type', 'unknown')
            content = elem.get('content', '')[:50]
            bbox = elem.get('bbox', [])

            if bbox:
                center_x = int((bbox[0] + bbox[2]) / 2 * screen_width)
                center_y = int((bbox[1] + bbox[3]) / 2 * screen_height)
                print(f"  ID:{idx} | {elem_type} | '{content}' | 坐标:({center_x}, {center_y})")
            else:
                print(f"  ID:{idx} | {elem_type} | '{content}' | 无坐标")

        if len(elements) > 20:
            print(f"  ... 还有 {len(elements) - 20} 个元素")
        print("=" * 60)

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
    print(f"Model Inited: {model}, Provider: {provider}")

    tool_result_content = None

    print(f"Start the message loop. User messages: {messages}")

    if model == "claude-3-5-sonnet-20241022": # Anthropic loop
        while True:
            parsed_screen = omniparser_client() # parsed_screen: {"som_image_base64": dino_labled_img, "parsed_content_list": parsed_content_list, "screen_info"}
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
        while True:
            print("[DEBUG] Calling omniparser_client()...")
            parsed_screen = omniparser_client()
            print(f"[DEBUG] Got parsed_screen, calling actor...")
            tools_use_needed, vlm_response_json = actor(messages=messages, parsed_screen=parsed_screen)

            for message, tool_result_content in executor(tools_use_needed, messages):
                yield message

            if not tool_result_content:
                return messages

            messages.append({"content": tool_result_content, "role": "user"})