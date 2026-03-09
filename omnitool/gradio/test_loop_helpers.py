from __future__ import annotations

import sys
import types
from pathlib import Path


GRADIO_DIR = Path(__file__).resolve().parent
if str(GRADIO_DIR) not in sys.path:
    sys.path.insert(0, str(GRADIO_DIR))

# loop_helpers depends on anthropic types; provide lightweight stubs for unit tests.
if "anthropic.types.beta" not in sys.modules:
    anthropic_mod = types.ModuleType("anthropic")
    anthropic_types_mod = types.ModuleType("anthropic.types")
    anthropic_beta_mod = types.ModuleType("anthropic.types.beta")
    anthropic_beta_mod.BetaContentBlock = object
    anthropic_beta_mod.BetaToolUseBlock = lambda **kwargs: types.SimpleNamespace(**kwargs)
    anthropic_types_mod.beta = anthropic_beta_mod
    anthropic_mod.types = anthropic_types_mod

    sys.modules["anthropic"] = anthropic_mod
    sys.modules["anthropic.types"] = anthropic_types_mod
    sys.modules["anthropic.types.beta"] = anthropic_beta_mod

import loop_helpers as helpers


def _screen(screen_info: str, contents: list[str] | None = None) -> dict:
    parsed = [{"type": "text", "content": c} for c in (contents or [])]
    return {
        "screen_info": screen_info,
        "parsed_content_list": parsed,
        "original_screenshot_base64": None,
    }


def test_action_gate_passed_for_non_visual_key():
    pre = _screen("https://www.xiaohongshu.com/explore", [])
    post = _screen("https://www.xiaohongshu.com/explore", [])
    passed, reason = helpers.action_gate_passed(pre, post, {"Next Action": "key", "value": "tab"})
    assert passed
    assert "non_visual_key_ok=True" in reason


def test_action_gate_passed_for_focus_click_target():
    pre = _screen("https://www.xiaohongshu.com/explore", ["search input"])
    post = _screen("https://www.xiaohongshu.com/explore", ["search input"])
    passed, reason = helpers.action_gate_passed(pre, post, {"Next Action": "left_click", "Box ID": 0})
    assert passed
    assert "focus_click_ok=True" in reason


def test_topic_anchor_evidence_requires_input_and_results():
    parsed_screen = _screen(
        "search openclaw explore results page",
        ["search openclaw", "openclaw guide", "openclaw usage"],
    )
    evidence = helpers.topic_anchor_evidence(parsed_screen, "openclaw")
    assert evidence["input_hit"] is True
    assert evidence["results_hit"] is True
    assert evidence["anchored"] is True


def test_build_action_gate_recovery_hint_by_action_type():
    type_hint = helpers.build_action_gate_recovery_hint({"Next Action": "type"}, _screen("", []))
    assert "Ctrl+A" in type_hint
    assert "仅当确认页面偏航时再使用 Ctrl+L" in type_hint

    click_hint = helpers.build_action_gate_recovery_hint(
        {"Next Action": "left_click", "Box ID": 0},
        _screen("", ["发送"]),
    )
    assert "按钮点击未生效" in click_hint


def test_advance_plan_if_ready_updates_index_and_emits_next_step():
    outputs: list[str] = []
    messages: list[dict] = []
    plan_steps = [
        {"step": 1, "success_groups": [["publish"]], "success": "publish is visible", "action": "noop"},
        {"step": 2, "success_groups": [["done"]], "success": "done is visible", "action": "next"},
    ]
    plan_state = {"current_index": 0}

    helpers.advance_plan_if_ready(
        parsed_screen=_screen("publish success page", []),
        messages=messages,
        output_callback=lambda msg: outputs.append(str(msg)),
        plan_steps=plan_steps,
        plan_state=plan_state,
        plan_update_callback=None,
    )

    assert plan_state["current_index"] == 1
    joined = "\n".join(str(m.get("content", "")) for m in messages)
    assert "Step 1 已完成" in joined
    assert "下一步" in joined
    assert any("下一步" in out for out in outputs)
