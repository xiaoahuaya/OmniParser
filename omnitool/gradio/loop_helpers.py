"""Helper utilities for loop.py.

This module isolates UI action-gate/recovery and screen-diff logic from the
sampling loop to keep loop.py focused on orchestration.
"""

from __future__ import annotations

from collections.abc import Callable
import base64
import re
import time
from io import BytesIO
from types import SimpleNamespace

from PIL import Image
from anthropic.types.beta import BetaContentBlock, BetaToolUseBlock


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
REPEAT_COORD_SCROLL_RECOVERY_HINT = (
    "⚙️ 重复点击拦截：当前处于内容流/搜索结果页。下一步必须先执行滚轮下滑"
    "（scroll_down）或按 PageDown，再选择新卡片；禁止继续点击当前区域。"
)
SCROLL_KEY_FALLBACK_HINT = "⚙️ 滚动兜底：滚轮变化不明显，自动尝试键盘翻页（PageDown/PageUp）。"
NAVIGATION_RECOVERY_HINT = (
    "检测到导航异常（如 bing/challenge/验证页）。下一步必须使用键盘恢复："
    "先按 Ctrl+L，再输入完整目标 URL（https://www.xiaohongshu.com）并回车。"
    "禁止点击地址栏下拉建议。"
)

ACTION_GATE_RECOVERY_HINT = (
    "⚠️ 动作确认门: 刚才动作后未检测到有效变化（URL未变、画面差异过小、元素状态未变）。"
    "禁止重复同一点击。请执行恢复动作："
    "1) Esc 关闭弹层 2) Ctrl+L 3) 输入目标URL并回车 或切换到可编辑输入框后 Ctrl+A + 粘贴。"
)
ACTION_GATE_DIFF_THRESHOLD = 0.012
ACTION_GATE_DIFF_THRESHOLD_BY_ACTION = {
    # 输入/滚动动作的可见变化通常更细微，阈值适当降低以减少误判。
    "type": 0.004,
    "type_submit": 0.004,
    "scroll_down": 0.007,
    "scroll_up": 0.007,
    "drag": 0.007,
    "mouse_wheel": 0.007,
    "left_click": 0.009,
    "double_click": 0.009,
}
ACTION_GATE_NON_VISUAL_KEYS = {
    "esc",
    "tab",
    "shift+tab",
    "ctrl+a",
    "ctrl+c",
    "ctrl+v",
    "ctrl+x",
    "backspace",
    "delete",
    "left",
    "right",
    "up",
    "down",
    "home",
    "end",
}
ACTION_GATE_FOCUS_TARGET_KEYWORDS = (
    "输入",
    "粘贴",
    "搜索",
    "评论",
    "输入框",
    "textbox",
    "search",
    "title",
    "正文",
)
REQUIRED_FIELD_RECOVERY_HINT = (
    "⚠️ 提交前校验未通过：检测到标题可能缺失。"
    "请先定位标题输入框并填写标题，再继续“下一步/一键排版/发布”类动作。"
)
SUBMIT_ACTION_KEYWORDS = (
    "一键排版",
    "排版",
    "发布",
    "下一步",
    "提交",
    "保存",
    "确认",
    "publish",
    "submit",
    "next",
    "save",
    "confirm",
)
TITLE_MISSING_PATTERNS = (
    "请输入标题",
    "输入标题",
    "标题不能为空",
    "please enter title",
    "title required",
    "missing title",
)
VIEWPORT_RECOVERY_HINT = (
    "⚠️ 视口疑似未展开（底部存在大面积空白/黑边），可能导致“下一步/发布”按钮不可见。"
    "请先执行窗口恢复动作：key=win+up（必要时再按一次），然后继续操作。"
)
WINDOW_MAXIMIZE_KEYS = ("win+up", "alt+space,x")
VIEWPORT_BLACK_RATIO_THRESHOLD = 0.90
VIEWPORT_BAND_HEIGHT_RATIO = 0.18
XHS_EDITOR_BLOCKED_CLICK_HINT = (
    "⚠️ 当前处于长文编辑器，禁止点击左侧“发布笔记”入口按钮（该按钮会返回发布入口页并重开流程）。"
    "请改为编辑器内流程：先“一键排版”再“下一步/发布”。"
)


def extract_url_like(screen_info: str) -> str:
    if not screen_info:
        return ""
    m = re.search(r"https?://[^\s]+", screen_info, flags=re.IGNORECASE)
    if m:
        return m.group(0).lower()
    m = re.search(
        r"\b(?:www\.)?[a-z0-9.-]+\.[a-z]{2,}(?:/[^\s]*)?\b",
        screen_info,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(0).lower()
    return ""


def element_signature(parsed_screen: dict) -> str:
    items = parsed_screen.get("parsed_content_list", []) or []
    parts = []
    for item in items[:40]:
        t = str(item.get("type", ""))
        c = str(item.get("content", ""))
        parts.append(f"{t}:{c}")
    return "|".join(parts)


def safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_box_content(parsed_screen: dict, box_id: int | None) -> str:
    if box_id is None:
        return ""
    items = parsed_screen.get("parsed_content_list", []) or []
    if box_id < 0 or box_id >= len(items):
        return ""
    return str(items[box_id].get("content", "") or "")


def downsample_gray_pixels(image_b64: str, size: int = SCREEN_DIFF_SIZE) -> list[int]:
    image_bytes = base64.b64decode(image_b64)
    img = Image.open(BytesIO(image_bytes)).convert("L").resize((size, size))
    return list(img.getdata())


def mean_abs_diff(pixels_a: list[int], pixels_b: list[int]) -> float:
    if not pixels_a or not pixels_b or len(pixels_a) != len(pixels_b):
        return 1.0
    total = sum(abs(a - b) for a, b in zip(pixels_a, pixels_b))
    return total / (len(pixels_a) * 255)


def screen_change_score(prev_b64: str, curr_b64: str) -> float:
    try:
        prev_pixels = downsample_gray_pixels(prev_b64)
        curr_pixels = downsample_gray_pixels(curr_b64)
        return mean_abs_diff(prev_pixels, curr_pixels)
    except Exception as e:
        print(f"[WARN] Screen diff failed: {e}")
        return 1.0


def action_gate_passed(
    pre_screen: dict,
    post_screen: dict,
    vlm_response_json: dict | None = None,
) -> tuple[bool, str]:
    pre_info = pre_screen.get("screen_info", "") or ""
    post_info = post_screen.get("screen_info", "") or ""
    pre_url = extract_url_like(pre_info)
    post_url = extract_url_like(post_info)
    url_changed = bool(pre_url and post_url and pre_url != post_url)

    action = ""
    key_value = ""
    target_text = ""
    if isinstance(vlm_response_json, dict):
        action = str(vlm_response_json.get("Next Action", "") or "").lower().strip()
        key_value = str(vlm_response_json.get("value", "") or "").lower().replace(" ", "")
        box_id = safe_int(vlm_response_json.get("Box ID"))
        target_text = get_box_content(pre_screen, box_id).lower()

    diff_threshold = ACTION_GATE_DIFF_THRESHOLD_BY_ACTION.get(action, ACTION_GATE_DIFF_THRESHOLD)
    pre_b64 = pre_screen.get("original_screenshot_base64")
    post_b64 = post_screen.get("original_screenshot_base64")
    diff_score = screen_change_score(pre_b64, post_b64) if pre_b64 and post_b64 else 0.0
    visual_changed = diff_score >= diff_threshold

    pre_sig = element_signature(pre_screen)
    post_sig = element_signature(post_screen)
    element_changed = pre_sig != post_sig

    non_visual_key_ok = action == "key" and (
        key_value in ACTION_GATE_NON_VISUAL_KEYS or key_value.startswith("ctrl+l")
    )
    focus_click_ok = action in ("left_click", "double_click") and any(
        token in target_text for token in ACTION_GATE_FOCUS_TARGET_KEYWORDS
    )

    passed = url_changed or visual_changed or element_changed or non_visual_key_ok or focus_click_ok
    reason = (
        f"url_changed={url_changed}, visual_changed={visual_changed}"
        f"(diff={diff_score:.4f}, threshold={diff_threshold:.4f}), "
        f"element_changed={element_changed}, non_visual_key_ok={non_visual_key_ok}, "
        f"focus_click_ok={focus_click_ok}, action={action or 'unknown'}"
    )
    return passed, reason


def is_submit_like_action(vlm_response_json: dict, parsed_screen: dict) -> bool:
    if not isinstance(vlm_response_json, dict):
        return False
    action = str(vlm_response_json.get("Next Action", "") or "").lower().strip()
    if action == "type_submit":
        return True
    if action not in ("left_click", "double_click", "key"):
        return False
    if action == "key":
        key_value = str(vlm_response_json.get("value", "") or "").lower()
        return key_value in ("enter", "ctrl+enter")

    box_id = safe_int(vlm_response_json.get("Box ID"))
    target_text = get_box_content(parsed_screen, box_id).lower()
    return any(keyword.lower() in target_text for keyword in SUBMIT_ACTION_KEYWORDS)


def is_title_missing(parsed_screen: dict) -> bool:
    screen_info = str(parsed_screen.get("screen_info", "") or "").lower()
    return any(token.lower() in screen_info for token in TITLE_MISSING_PATTERNS)


def required_field_violation(vlm_response_json: dict, parsed_screen: dict) -> str | None:
    if is_submit_like_action(vlm_response_json, parsed_screen) and is_title_missing(parsed_screen):
        return REQUIRED_FIELD_RECOVERY_HINT
    return None


def is_maximize_action(vlm_response_json: dict) -> bool:
    if not isinstance(vlm_response_json, dict):
        return False
    if str(vlm_response_json.get("Next Action", "") or "").lower().strip() != "key":
        return False
    key_value = str(vlm_response_json.get("value", "") or "").lower().replace(" ", "")
    return any(k in key_value for k in WINDOW_MAXIMIZE_KEYS)


def has_bottom_black_band(image_b64: str | None) -> bool:
    if not image_b64:
        return False
    try:
        image_bytes = base64.b64decode(image_b64)
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        width, height = img.size
        if width < 200 or height < 200:
            return False
        band_h = max(1, int(height * VIEWPORT_BAND_HEIGHT_RATIO))
        band = img.crop((0, height - band_h, width, height)).convert("L")
        pixels = list(band.getdata())
        if not pixels:
            return False
        dark_pixels = sum(1 for p in pixels if p < 22)
        dark_ratio = dark_pixels / len(pixels)
        return dark_ratio >= VIEWPORT_BLACK_RATIO_THRESHOLD
    except Exception:
        return False


def needs_viewport_recover(parsed_screen: dict) -> bool:
    screen_info = str(parsed_screen.get("screen_info", "") or "").lower()
    browser_like = any(k in screen_info for k in ("http://", "https://", "xiaohongshu", "edge", "chrome"))
    if not browser_like:
        return False
    return has_bottom_black_band(parsed_screen.get("original_screenshot_base64"))


def is_xhs_editor_context(parsed_screen: dict) -> bool:
    screen_info = str(parsed_screen.get("screen_info", "") or "")
    return ("新的创作" in screen_info) and ("输入标题" in screen_info or "粘贴到这里或输入文字" in screen_info)


def blocked_editor_exit_action(vlm_response_json: dict, parsed_screen: dict) -> str | None:
    if not isinstance(vlm_response_json, dict):
        return None
    action = str(vlm_response_json.get("Next Action", "") or "").lower().strip()
    if action not in ("left_click", "double_click"):
        return None
    if not is_xhs_editor_context(parsed_screen):
        return None
    box_id = safe_int(vlm_response_json.get("Box ID"))
    target_text = get_box_content(parsed_screen, box_id).strip()
    if target_text == "发布笔记":
        return XHS_EDITOR_BLOCKED_CLICK_HINT
    return None


def normalize_text_for_match(text: str) -> str:
    return re.sub(r"\s+", "", (text or "")).lower()


def topic_anchor_evidence(parsed_screen: dict, topic_term: str) -> dict:
    topic_raw = (topic_term or "").strip().lower()
    if not topic_raw:
        return {"topic": "", "input_hit": False, "results_hit": False, "anchored": False}

    screen_info = str(parsed_screen.get("screen_info", "") or "")
    screen_norm = normalize_text_for_match(screen_info)
    topic_norm = normalize_text_for_match(topic_raw)

    items = parsed_screen.get("parsed_content_list", []) or []
    contents = [normalize_text_for_match(str(item.get("content", "") or "")) for item in items]

    input_hit = bool(
        re.search(fr"(搜索|search).{{0,24}}{re.escape(topic_norm)}", screen_norm)
        or re.search(fr"{re.escape(topic_norm)}.{{0,24}}(搜索|search)", screen_norm)
    )
    if not input_hit:
        for content in contents:
            if topic_norm in content and ("搜索" in content or "search" in content):
                input_hit = True
                break

    results_markers = ("结果", "话题", "笔记", "综合", "相关", "discover", "explore")
    has_results_marker = any(marker in screen_norm for marker in results_markers)
    result_hits = sum(
        1 for content in contents if topic_norm in content and ("搜索" not in content and "search" not in content)
    )
    results_hit = (has_results_marker and result_hits >= 1) or (result_hits >= 2)

    return {
        "topic": topic_raw,
        "input_hit": input_hit,
        "results_hit": results_hit,
        "anchored": input_hit and results_hit,
    }


def build_action_gate_recovery_hint(vlm_response_json: dict, parsed_screen: dict) -> str:
    base = "⚠️ 动作确认门: 刚才动作后未检测到有效变化（URL未变、画面差异过小、元素状态未变）。禁止重复同一点击。"
    if not isinstance(vlm_response_json, dict):
        return base + " 请先按 Esc 重置状态，再改用不同位置点击或键盘 Tab+Enter。"

    action = str(vlm_response_json.get("Next Action", "") or "").lower().strip()
    box_id = safe_int(vlm_response_json.get("Box ID"))
    target_text = get_box_content(parsed_screen, box_id).lower()
    key_value = str(vlm_response_json.get("value", "") or "").lower().replace(" ", "")

    if action in ("type", "type_submit"):
        return (
            base
            + " 输入动作未生效。请执行恢复动作：1) Esc 2) 单击目标输入框本体 3) Ctrl+A 后重新输入/粘贴。"
            " 仅当确认页面偏航时再使用 Ctrl+L。"
        )
    if action in ("scroll_down", "scroll_up", "drag", "mouse_wheel"):
        return (
            base
            + " 滚动动作未生效。请执行恢复动作：1) Esc 2) 单击内容区域中心 3) 再次滚动或改用翻页按钮/方向键。"
            " 不要直接 Ctrl+L。"
        )
    if action == "key":
        if key_value in ("enter", "ctrl+enter"):
            return (
                base
                + " 提交按键未生效。请执行恢复动作：1) Esc 2) 重新聚焦目标控件（按钮/输入框）3) 再按 Enter。"
            )
        if "ctrl+l" in key_value:
            return (
                base
                + " 地址栏恢复未生效。请继续输入完整目标 URL 回车；若仍无变化，Esc 后回到页面控件。"
            )
        return base + " 键盘动作未生效。请 Esc 后改用同语义的不同动作。"
    if action in ("left_click", "double_click", "right_click"):
        if any(token in target_text for token in ("搜索", "search")):
            return (
                base
                + " 搜索框点击未生效。请执行恢复动作：1) Esc 2) 单击搜索输入框本体 3) 输入关键词并回车。"
            )
        if any(token in target_text for token in ("评论", "发送", "发布", "下一步", "一键排版", "保存", "确认")):
            return (
                base
                + " 按钮点击未生效。请执行恢复动作：1) Esc 2) 改点按钮内不同位置或双击 3) 必要时用 Enter 触发。"
            )
        return (
            base
            + " 点击未生效。请执行恢复动作：1) Esc 2) 改点同控件不同区域/双击 3) 或用 Tab+Enter。"
            " 仅导航异常时使用 Ctrl+L。"
        )
    return ACTION_GATE_RECOVERY_HINT


def is_similar_coord(coord1: list, coord2: list, threshold: int = REPEAT_COORD_THRESHOLD) -> bool:
    if not coord1 or not coord2:
        return False
    return abs(coord1[0] - coord2[0]) < threshold and abs(coord1[1] - coord2[1]) < threshold


def check_repeat_coords(recent_coords: list, new_coord: list) -> int:
    if not new_coord or not recent_coords:
        return 0
    count = 0
    for coord in reversed(recent_coords):
        if is_similar_coord(coord, new_coord):
            count += 1
        else:
            break
    return count


def is_click_like_action(vlm_response_json: dict) -> bool:
    if not isinstance(vlm_response_json, dict):
        return False
    action = str(vlm_response_json.get("Next Action", "") or "").lower().strip()
    return action in ("left_click", "double_click", "right_click")


def is_scroll_like_action(vlm_response_json: dict) -> bool:
    if not isinstance(vlm_response_json, dict):
        return False
    action = str(vlm_response_json.get("Next Action", "") or "").lower().strip()
    return action in ("scroll_down", "scroll_up", "mouse_wheel")


def is_xhs_feed_or_search_context(parsed_screen: dict) -> bool:
    screen_info = str(parsed_screen.get("screen_info", "") or "").lower()
    if not screen_info:
        return False
    if "creator.xiaohongshu.com" in screen_info:
        return False
    if "xiaohongshu" not in screen_info and "小红书" not in screen_info:
        return False
    markers = (
        "search_result",
        "搜索结果",
        "发现",
        "推荐",
        "explore",
        "话题",
        "相关内容",
    )
    return any(marker in screen_info for marker in markers)


def forced_scroll_tool_response() -> object:
    tool_block = BetaToolUseBlock(
        type="tool_use",
        id=f"toolu_forced_scroll_{int(time.time() * 1000)}",
        name="computer",
        input={"action": "scroll_down"},
    )
    return SimpleNamespace(content=[tool_block])


def forced_key_tool_response(key_text: str) -> object:
    tool_block = BetaToolUseBlock(
        type="tool_use",
        id=f"toolu_forced_key_{int(time.time() * 1000)}",
        name="computer",
        input={"action": "key", "text": key_text},
    )
    return SimpleNamespace(content=[tool_block])


def matches_success(screen_info: str, success_groups: list[list[str]]) -> bool:
    if not screen_info or not success_groups:
        return False
    screen_lower = screen_info.lower()
    for group in success_groups:
        if not any(alt.lower() in screen_lower for alt in group if alt):
            return False
    return True


def advance_plan_if_ready(
    *,
    parsed_screen: dict,
    messages: list,
    output_callback: Callable[[BetaContentBlock], None],
    plan_steps: list[dict] | None,
    plan_state: dict | None,
    plan_update_callback: Callable[[dict], None] | None,
) -> None:
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
        if not matches_success(screen_info, success_groups):
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

