from __future__ import annotations

import base64
import sys
import types
from pathlib import Path


GRADIO_DIR = Path(__file__).resolve().parent
if str(GRADIO_DIR) not in sys.path:
    sys.path.insert(0, str(GRADIO_DIR))
AGENT_DIR = GRADIO_DIR / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


def _ns_factory(**kwargs):
    return types.SimpleNamespace(**kwargs)


if "anthropic" not in sys.modules:
    anthropic_mod = types.ModuleType("anthropic")
    anthropic_mod.APIResponse = object

    anthropic_types_mod = types.ModuleType("anthropic.types")
    anthropic_types_mod.ToolResultBlockParam = dict

    anthropic_beta_mod = types.ModuleType("anthropic.types.beta")
    anthropic_beta_mod.BetaContentBlock = object
    anthropic_beta_mod.BetaMessage = _ns_factory
    anthropic_beta_mod.BetaTextBlock = _ns_factory
    anthropic_beta_mod.BetaToolUseBlock = _ns_factory
    anthropic_beta_mod.BetaMessageParam = dict
    anthropic_beta_mod.BetaUsage = _ns_factory

    anthropic_types_mod.beta = anthropic_beta_mod
    anthropic_mod.types = anthropic_types_mod

    sys.modules["anthropic"] = anthropic_mod
    sys.modules["anthropic.types"] = anthropic_types_mod
    sys.modules["anthropic.types.beta"] = anthropic_beta_mod


if "agent.llm_utils.oaiclient" not in sys.modules:
    oaiclient_mod = types.ModuleType("agent.llm_utils.oaiclient")
    oaiclient_mod.run_oai_interleaved = lambda *args, **kwargs: ("", 0)
    sys.modules["agent.llm_utils.oaiclient"] = oaiclient_mod

if "agent.llm_utils.groqclient" not in sys.modules:
    groqclient_mod = types.ModuleType("agent.llm_utils.groqclient")
    groqclient_mod.run_groq_interleaved = lambda *args, **kwargs: ("", 0)
    sys.modules["agent.llm_utils.groqclient"] = groqclient_mod

if "agent.llm_utils.proxy_client" not in sys.modules:
    proxyclient_mod = types.ModuleType("agent.llm_utils.proxy_client")
    proxyclient_mod.run_proxy_interleaved = lambda *args, **kwargs: ("", 0)
    sys.modules["agent.llm_utils.proxy_client"] = proxyclient_mod

if "agent.llm_utils.utils" not in sys.modules:
    utils_mod = types.ModuleType("agent.llm_utils.utils")
    utils_mod.is_image_path = lambda value: False
    sys.modules["agent.llm_utils.utils"] = utils_mod


import vlm_agent as agent_mod


def test_non_strict_json_with_failure_word_is_executed():
    raw_response = (
        "Reasoning: 上一步失败后先走恢复链路，用 Ctrl+L 收敛到目标站点\n"
        "Next Action: key\n"
        "value: ctrl+l"
    )
    output_events: list[tuple[str, str]] = []

    agent_mod.run_oai_interleaved = lambda *args, **kwargs: (raw_response, 42)

    agent = agent_mod.VLMAgent(
        model="omniparser + gpt-4o",
        provider="openai",
        api_key="test",
        output_callback=lambda message, sender="bot": output_events.append((sender, str(message))),
        api_response_callback=lambda response: None,
    )

    response_message, response_json = agent(
        messages=[{"role": "user", "content": ["打开小红书"]}],
        parsed_screen={
            "original_screenshot_base64": "",
            "latency": 0.12,
            "screen_info": "1: 搜索框\n2: 地址栏",
            "screenshot_uuid": "test-shot",
            "screenshot_path": str(GRADIO_DIR / "tmp" / "dummy.png"),
            "som_screenshot_path": str(GRADIO_DIR / "tmp" / "dummy_som.png"),
            "width": 1280,
            "height": 720,
            "som_image_base64": base64.b64encode(b"dummy").decode("ascii"),
            "parsed_content_list": [],
        },
    )

    assert agent._should_retry_response(raw_response) is True
    assert response_json["Next Action"] == "key"
    assert response_json["value"] == "ctrl+l"
    assert response_message.content[-1].input == {"action": "key", "text": "ctrl+l"}
    assert any("快捷键" in message for _sender, message in output_events)
