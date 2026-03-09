"""Task-cycle policy helpers extracted from app.py.

These helpers are intentionally pure/lightweight so they can be reused and
tested without importing the full Gradio app runtime.
"""

from __future__ import annotations


DEFAULT_PUBLISH_SUCCESS_PATTERNS = (
    "发布成功",
    "已发布",
    "发布完成",
    "发布后状态确认",
)

DEFAULT_PUBLISH_MARKERS = (
    "小红书发布",
    "发布纯文本笔记",
    "标题与正文非空后再发布",
    "creator.xiaohongshu.com/publish",
    "发布笔记",
)

DEFAULT_INTERACTION_TASK_TEMPLATE = (
    "围绕“{topic}”方向自然浏览推荐内容，并完成点赞/收藏/评论/关注中的 2-4 项互动"
)

DEFAULT_ANCHOR_MARKERS = (
    "搜索",
    "search",
    "话题",
    "结果页",
    "搜索结果",
    "performed left_click",
    "pressed keys: enter",
)


def chat_entry_text(entry) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, (tuple, list)):
        parts: list[str] = []
        for part in entry:
            if isinstance(part, str) and part.strip():
                parts.append(part)
        return "\n".join(parts).strip()
    return ""


def collect_chat_text_since(state: dict, start_index: int, max_items: int = 120) -> str:
    entries = state.get("chatbot_messages") or []
    if not isinstance(entries, list):
        return ""
    safe_index = max(0, int(start_index))
    slice_items = entries[safe_index:]
    if max_items > 0:
        slice_items = slice_items[-max_items:]
    parts: list[str] = []
    for item in slice_items:
        text = chat_entry_text(item)
        if text:
            parts.append(text)
    return "\n".join(parts).lower()


def detect_publish_success(
    state: dict,
    start_index: int,
    success_patterns: tuple[str, ...] = DEFAULT_PUBLISH_SUCCESS_PATTERNS,
) -> bool:
    text = collect_chat_text_since(state, start_index=start_index, max_items=160)
    return any(pattern in text for pattern in success_patterns)


def is_publish_task(
    state: dict,
    publish_markers: tuple[str, ...] = DEFAULT_PUBLISH_MARKERS,
) -> bool:
    task_type = str(state.get("task_type") or "").strip()
    profile_id = str(state.get("profile_id") or "").strip().lower()
    goal_task = str(state.get("goal_task") or "").strip().lower()

    if task_type == "小红书发布":
        return True
    if profile_id in {"xhs_publish"} or profile_id.endswith("_publish"):
        return True

    return any(marker in goal_task for marker in publish_markers)


def consume_interaction_task(
    state: dict,
    default_template: str = DEFAULT_INTERACTION_TASK_TEMPLATE,
) -> str:
    tasks = state.get("interaction_tasks") or []
    topic = str(state.get("profile_topic") or "通用")
    if not isinstance(tasks, list) or not tasks:
        return default_template.format(topic=topic)

    index = int(state.get("interaction_task_index") or 0)
    task = str(tasks[index % len(tasks)]).strip()
    state["interaction_task_index"] = (index + 1) % len(tasks)
    try:
        return task.format(topic=topic)
    except Exception:
        return task


def decide_next_cycle_strategy(state: dict, next_cycle: int, now_ts: float) -> dict:
    blocked_reasons: list[str] = []

    publish_every = max(1, int(state.get("publish_every_cycles") or 1))
    if publish_every > 1 and next_cycle % publish_every != 0:
        blocked_reasons.append(f"发布节奏为每 {publish_every} 轮 1 次")

    max_publish = int(state.get("max_publish_per_session") or 0)
    publish_count = int(state.get("publish_count") or 0)
    if max_publish > 0 and publish_count >= max_publish:
        blocked_reasons.append(f"本次会话发布已达上限 {max_publish} 次")

    cooldown_sec = max(0.0, float(state.get("publish_cooldown_sec") or 0.0))
    last_publish_ts = state.get("last_publish_ts")
    if cooldown_sec > 0 and last_publish_ts:
        remain = cooldown_sec - (now_ts - float(last_publish_ts))
        if remain > 0:
            blocked_reasons.append(f"发布冷却剩余约 {int(remain)} 秒")

    interaction_task = consume_interaction_task(state)
    return {
        "allow_publish": len(blocked_reasons) == 0,
        "blocked_reasons": blocked_reasons,
        "interaction_task": interaction_task,
    }


def build_next_cycle_prompt(state: dict, next_cycle: int, strategy: dict) -> str:
    topic = str(state.get("profile_topic") or "通用")
    interaction_task = str(strategy.get("interaction_task") or "").strip()
    base_prompt = str(state.get("continuous_prompt") or "").strip()
    allow_publish = bool(strategy.get("allow_publish"))
    anchor_required = bool(state.get("topic_anchor_required", False))
    anchor_done = bool(state.get("topic_anchor_done", False))

    if anchor_required and (not anchor_done):
        prompt = (
            f"进入第 {next_cycle} 轮任务（方向：{topic}）。"
            f"本轮首要目标：主题锚定，必须先完成站内搜索“{topic}”并进入相关结果/话题页。"
            "硬约束：在锚定完成前，禁止点赞/收藏/评论/关注/发布。"
            "若搜索框焦点失败，先执行恢复链路（Esc -> 重新定位搜索框 -> 输入主题 -> Enter）。"
        )
        if base_prompt:
            prompt += "\n补充约束：" + base_prompt
        return prompt

    if allow_publish:
        prompt = (
            f"进入第 {next_cycle} 轮小红书养号（方向：{topic}）。"
            "本轮策略：互动优先，可发布最多 1 条。"
            f"先执行：{interaction_task}。"
            "如果完成发布，后续动作回到浏览/点赞/收藏/评论，不要再次发布。"
        )
    else:
        blocked = strategy.get("blocked_reasons") or []
        reason_text = "；".join(str(item) for item in blocked if str(item).strip()) or "策略限制"
        prompt = (
            f"进入第 {next_cycle} 轮小红书养号（方向：{topic}）。"
            f"本轮策略：仅互动，禁止发布（原因：{reason_text}）。"
            f"执行任务：{interaction_task}。"
            "本轮不得执行“一键排版/下一步/发布”相关动作。"
        )

    prompt += (
        " 浏览新内容硬约束：若连续 2 次点击同一区域仍未打开新内容，"
        "必须先执行 scroll_down（或 PageDown）1-3 次，再选择新的卡片。"
    )

    if base_prompt:
        prompt += "\n补充约束：" + base_prompt
        if not allow_publish:
            prompt += "；若与“禁止发布”冲突，以本轮策略为准。"
    return prompt


def plan_is_incomplete(state: dict) -> bool:
    steps = state.get("plan_steps") or []
    if not isinstance(steps, list) or not steps:
        return False
    idx = int(state.get("plan_step_index") or 0)
    return idx < len(steps)


def looks_topic_anchored_from_text(
    text: str,
    topic: str,
    markers: tuple[str, ...] = DEFAULT_ANCHOR_MARKERS,
) -> bool:
    normalized_text = (text or "").lower()
    normalized_topic = (topic or "").strip().lower()
    if not normalized_text or not normalized_topic:
        return False
    if normalized_topic not in normalized_text:
        return False
    return any(marker in normalized_text for marker in markers)

