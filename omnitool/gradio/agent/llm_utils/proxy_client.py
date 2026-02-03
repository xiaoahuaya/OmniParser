"""
统一的 LLM 中转 API 客户端
支持 OpenAI 兼容格式的中转 API（包括 Claude 中转）
"""
import os
import requests
import base64
from typing import Optional, Tuple, List, Dict, Any
from .utils import is_image_path, encode_image

DEBUG_LOGS = os.getenv("OMNITOOL_DEBUG", "").lower() in ("1", "true", "yes")

def _debug_print(*args, **kwargs):
    if DEBUG_LOGS:
        print(*args, **kwargs)


def run_proxy_interleaved(
    messages: list,
    system: str,
    model_name: str,
    api_key: str,
    base_url: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> Tuple[str, int]:
    """
    通过中转 API 调用 LLM
    支持 OpenAI 兼容格式的所有中转服务

    Args:
        messages: 消息列表
        system: 系统提示词
        model_name: 模型名称
        api_key: API Key
        base_url: 中转 API 地址
        max_tokens: 最大 token 数
        temperature: 温度参数

    Returns:
        (响应文本, token 使用量)
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    final_messages = [{"role": "system", "content": system}]

    if isinstance(messages, list):
        for item in messages:
            contents = []
            if isinstance(item, dict):
                for cnt in item.get("content", []):
                    if isinstance(cnt, str):
                        if is_image_path(cnt):
                            base64_image = encode_image(cnt)
                            content = {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        else:
                            content = {"type": "text", "text": cnt}
                    elif hasattr(cnt, 'text'):
                        content = {"type": "text", "text": cnt.text}
                    elif hasattr(cnt, 'input'):
                        action_info = cnt.input
                        content = {"type": "text", "text": f"[已执行动作] {action_info}"}
                    elif isinstance(cnt, dict):
                        if cnt.get("type") == "tool_result":
                            tool_output = ""
                            for tc in cnt.get("content", []):
                                if isinstance(tc, dict) and tc.get("type") == "text":
                                    tool_output += tc.get("text", "")
                            content = {"type": "text", "text": f"[执行结果] {tool_output}"}
                        else:
                            content = {"type": "text", "text": str(cnt)}
                    else:
                        content = {"type": "text", "text": str(cnt)}
                    contents.append(content)
                message = {"role": "user", "content": contents}
            else:
                contents.append({"type": "text", "text": item})
                message = {"role": "user", "content": contents}
            final_messages.append(message)
    elif isinstance(messages, str):
        final_messages = [{"role": "user", "content": messages}]

    payload = {
        "model": model_name,
        "messages": final_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    endpoint = f"{base_url.rstrip('/')}/chat/completions"

    _debug_print("=" * 60)
    _debug_print("[PROXY REQUEST]")
    _debug_print(f"Endpoint: {endpoint}")
    _debug_print(f"Model: {model_name}")
    _debug_print(f"System: {system[:200]}..." if len(system) > 200 else f"System: {system}")
    _debug_print(f"Messages count: {len(final_messages)}")
    if DEBUG_LOGS:
        for i, msg in enumerate(final_messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if isinstance(content, list):
                text_parts = [c.get('text', '')[:100] for c in content if c.get('type') == 'text']
                img_count = sum(1 for c in content if c.get('type') == 'image_url')
                _debug_print(f"  [{i}] {role}: {text_parts} + {img_count} images")
            else:
                _debug_print(f"  [{i}] {role}: {str(content)[:100]}...")
    _debug_print("=" * 60)

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()
        text = result['choices'][0]['message']['content']
        token_usage = int(result.get('usage', {}).get('total_tokens', 0))

        _debug_print("[PROXY RESPONSE]")
        _debug_print(f"Status: {response.status_code}")
        _debug_print(f"Tokens: {token_usage}")
        _debug_print(f"Response: {text}")
        _debug_print("=" * 60)

        return text, token_usage

    except requests.exceptions.RequestException as e:
        error_msg = f"中转 API 请求失败: {e}"
        _debug_print(error_msg)
        if DEBUG_LOGS and hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                _debug_print(f"错误详情: {error_detail}")
            except Exception:
                _debug_print(f"响应内容: {e.response.text}")
        return error_msg, 0
    except (KeyError, IndexError) as e:
        error_msg = f"解析响应失败: {e}"
        _debug_print(error_msg)
        return error_msg, 0


def test_proxy_connection(base_url: str, api_key: str, model_name: str = "gpt-4o") -> Dict[str, Any]:
    """
    测试中转 API 连接

    Returns:
        {"success": bool, "message": str, "models": list}
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    try:
        models_endpoint = f"{base_url.rstrip('/')}/models"
        response = requests.get(models_endpoint, headers=headers, timeout=10)

        if response.status_code == 200:
            models = response.json().get('data', [])
            model_ids = [m.get('id', '') for m in models]
            return {
                "success": True,
                "message": "连接成功",
                "models": model_ids
            }
        else:
            return {
                "success": False,
                "message": f"连接失败: HTTP {response.status_code}",
                "models": []
            }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "message": f"连接失败: {e}",
            "models": []
        }
