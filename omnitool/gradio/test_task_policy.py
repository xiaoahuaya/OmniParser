from __future__ import annotations

import sys
from pathlib import Path


GRADIO_DIR = Path(__file__).resolve().parent
if str(GRADIO_DIR) not in sys.path:
    sys.path.insert(0, str(GRADIO_DIR))

import task_policy as policy


def test_detect_publish_success_scoped_from_start_index():
    state = {
        "chatbot_messages": [
            (None, "noise"),
            (None, "发布成功"),
            (None, "extra"),
        ]
    }
    assert policy.detect_publish_success(state, start_index=1)
    assert not policy.detect_publish_success(state, start_index=2)


def test_is_publish_task_by_type_profile_and_goal_marker():
    assert policy.is_publish_task({"task_type": "小红书发布"})
    assert policy.is_publish_task({"profile_id": "xhs_publish"})
    assert policy.is_publish_task({"goal_task": "open creator.xiaohongshu.com/publish now"})
    assert not policy.is_publish_task({"task_type": "小红书养号", "goal_task": "browse and interact"})


def test_decide_next_cycle_strategy_with_all_publish_blocks():
    now_ts = 1_000.0
    state = {
        "publish_every_cycles": 4,
        "max_publish_per_session": 1,
        "publish_count": 1,
        "publish_cooldown_sec": 300.0,
        "last_publish_ts": 900.0,
        "interaction_tasks": ["look around {topic}"],
        "interaction_task_index": 0,
        "profile_topic": "openclaw",
    }
    strategy = policy.decide_next_cycle_strategy(state, next_cycle=3, now_ts=now_ts)
    assert not strategy["allow_publish"]
    reasons = " | ".join(strategy["blocked_reasons"])
    assert "发布节奏" in reasons
    assert "发布已达上限" in reasons
    assert "发布冷却剩余" in reasons
    assert strategy["interaction_task"] == "look around openclaw"


def test_build_next_cycle_prompt_prefers_anchor_when_required():
    state = {
        "profile_topic": "openclaw",
        "continuous_prompt": "avoid repeated clicks",
        "topic_anchor_required": True,
        "topic_anchor_done": False,
    }
    prompt = policy.build_next_cycle_prompt(
        state=state,
        next_cycle=2,
        strategy={"allow_publish": False, "blocked_reasons": ["cooldown"], "interaction_task": "do stuff"},
    )
    assert "主题锚定" in prompt
    assert "禁止点赞/收藏/评论/关注/发布" in prompt
    assert "avoid repeated clicks" in prompt


def test_plan_is_incomplete():
    assert policy.plan_is_incomplete({"plan_steps": [{"step": 1}, {"step": 2}], "plan_step_index": 1})
    assert not policy.plan_is_incomplete({"plan_steps": [{"step": 1}], "plan_step_index": 1})
    assert not policy.plan_is_incomplete({"plan_steps": [], "plan_step_index": 0})

