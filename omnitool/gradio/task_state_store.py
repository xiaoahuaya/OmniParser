"""Task-state creation/reset/persistence helpers.

This module keeps state-shape details out of app.py, reducing duplicated
field definitions across new/reset/save/load paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


def state_file_path(node_id: str) -> Path:
    return Path(f"./tmp/task_state_{node_id}.json")


def build_new_task_state(
    *,
    node_id: str,
    windows_host_url: str,
    auto_proxy_failover_max_switches: int,
    auto_replan_max_rounds: int,
    auto_recover_max_cycles: int,
) -> dict:
    return {
        "node_id": node_id,
        "windows_host_url": windows_host_url,
        "status": "idle",
        "phase": "idle",
        "task_type": "自动识别",
        "profile_id": "",
        "profile_topic": "",
        "continuous_mode": False,
        "continuous_cycle": 0,
        "continuous_max_cycles": 1,
        "continuous_interval_sec": 0.0,
        "continuous_prompt": "",
        "publish_every_cycles": 4,
        "publish_cooldown_sec": 1800.0,
        "max_publish_per_session": 2,
        "publish_count": 0,
        "last_publish_ts": None,
        "interaction_tasks": [],
        "interaction_task_index": 0,
        "topic_anchor_required": False,
        "topic_anchor_done": False,
        "topic_anchor_input_hit": False,
        "topic_anchor_results_hit": False,
        "failover_budget": auto_proxy_failover_max_switches,
        "failover_switches": 0,
        "run_attempt": 0,
        "consecutive_failures": 0,
        "next_retry_at": None,
        "health": "unknown",
        "health_detail": "",
        "health_last_checked": None,
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
        "goal_task": "",
        "auto_replan_budget": auto_replan_max_rounds,
        "auto_replan_used": 0,
        "stagnation_rounds": 0,
        "auto_recover_budget": auto_recover_max_cycles,
        "auto_recover_used": 0,
        "plan": None,
        "plan_steps": None,
        "plan_step_index": 0,
    }


def reset_task_state_fields(
    state: dict,
    *,
    auto_proxy_failover_max_switches: int,
    auto_replan_max_rounds: int,
    auto_recover_max_cycles: int,
) -> None:
    state.update(
        {
            "status": "idle",
            "phase": "idle",
            "task_type": "自动识别",
            "profile_id": "",
            "profile_topic": "",
            "continuous_mode": False,
            "continuous_cycle": 0,
            "continuous_max_cycles": 1,
            "continuous_interval_sec": 0.0,
            "continuous_prompt": "",
            "publish_every_cycles": 4,
            "publish_cooldown_sec": 1800.0,
            "max_publish_per_session": 2,
            "publish_count": 0,
            "last_publish_ts": None,
            "interaction_tasks": [],
            "interaction_task_index": 0,
            "topic_anchor_required": False,
            "topic_anchor_done": False,
            "topic_anchor_input_hit": False,
            "topic_anchor_results_hit": False,
            "failover_budget": auto_proxy_failover_max_switches,
            "failover_switches": 0,
            "run_attempt": 0,
            "consecutive_failures": 0,
            "next_retry_at": None,
            "messages": [],
            "responses": {},
            "tools": {},
            "chatbot_messages": [],
            "stop": False,
            "last_error": None,
            "last_update": None,
            "goal_task": "",
            "auto_replan_budget": auto_replan_max_rounds,
            "auto_replan_used": 0,
            "stagnation_rounds": 0,
            "auto_recover_budget": auto_recover_max_cycles,
            "auto_recover_used": 0,
            "plan": None,
            "plan_steps": None,
            "plan_step_index": 0,
        }
    )


def serialize_task_state(
    *,
    state: dict,
    version: int,
    serialize_messages: Callable[[list], list],
    auto_proxy_failover_max_switches: int,
    auto_replan_max_rounds: int,
    auto_recover_max_cycles: int,
) -> dict:
    return {
        "version": version,
        "status": state.get("status"),
        "phase": state.get("phase", "idle"),
        "task_type": state.get("task_type", "自动识别"),
        "profile_id": state.get("profile_id", ""),
        "profile_topic": state.get("profile_topic", ""),
        "continuous_mode": state.get("continuous_mode", False),
        "continuous_cycle": state.get("continuous_cycle", 0),
        "continuous_max_cycles": state.get("continuous_max_cycles", 1),
        "continuous_interval_sec": state.get("continuous_interval_sec", 0.0),
        "continuous_prompt": state.get("continuous_prompt", ""),
        "publish_every_cycles": state.get("publish_every_cycles", 4),
        "publish_cooldown_sec": state.get("publish_cooldown_sec", 1800.0),
        "max_publish_per_session": state.get("max_publish_per_session", 2),
        "publish_count": state.get("publish_count", 0),
        "last_publish_ts": state.get("last_publish_ts"),
        "interaction_tasks": state.get("interaction_tasks", []),
        "interaction_task_index": state.get("interaction_task_index", 0),
        "topic_anchor_required": state.get("topic_anchor_required", False),
        "topic_anchor_done": state.get("topic_anchor_done", False),
        "topic_anchor_input_hit": state.get("topic_anchor_input_hit", False),
        "topic_anchor_results_hit": state.get("topic_anchor_results_hit", False),
        "failover_budget": state.get("failover_budget", auto_proxy_failover_max_switches),
        "failover_switches": state.get("failover_switches", 0),
        "run_attempt": state.get("run_attempt", 0),
        "consecutive_failures": state.get("consecutive_failures", 0),
        "next_retry_at": state.get("next_retry_at"),
        "health": state.get("health", "unknown"),
        "health_detail": state.get("health_detail", ""),
        "health_last_checked": state.get("health_last_checked"),
        "messages": serialize_messages(state.get("messages", [])),
        "chatbot_messages": state.get("chatbot_messages", []),
        "last_error": state.get("last_error"),
        "last_update": state.get("last_update"),
        "model": state.get("model"),
        "provider": state.get("provider"),
        "api_key": state.get("api_key"),
        "proxy_base_url": state.get("proxy_base_url"),
        "proxy_model": state.get("proxy_model"),
        "only_n_most_recent_images": state.get("only_n_most_recent_images"),
        "max_steps": state.get("max_steps"),
        "max_seconds": state.get("max_seconds"),
        "goal_task": state.get("goal_task", ""),
        "auto_replan_budget": state.get("auto_replan_budget", auto_replan_max_rounds),
        "auto_replan_used": state.get("auto_replan_used", 0),
        "stagnation_rounds": state.get("stagnation_rounds", 0),
        "auto_recover_budget": state.get("auto_recover_budget", auto_recover_max_cycles),
        "auto_recover_used": state.get("auto_recover_used", 0),
        "plan": state.get("plan"),
        "plan_steps": state.get("plan_steps"),
        "plan_step_index": state.get("plan_step_index"),
        "windows_host_url": state.get("windows_host_url"),
    }


def save_task_state(
    *,
    node_id: str,
    state: dict,
    version: int,
    serialize_messages: Callable[[list], list],
    auto_proxy_failover_max_switches: int,
    auto_replan_max_rounds: int,
    auto_recover_max_cycles: int,
) -> None:
    path = state_file_path(node_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = serialize_task_state(
        state=state,
        version=version,
        serialize_messages=serialize_messages,
        auto_proxy_failover_max_switches=auto_proxy_failover_max_switches,
        auto_replan_max_rounds=auto_replan_max_rounds,
        auto_recover_max_cycles=auto_recover_max_cycles,
    )
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def read_task_state(node_id: str) -> dict | None:
    path = state_file_path(node_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def apply_loaded_task_state(
    *,
    state: dict,
    data: dict,
    deserialize_messages: Callable[[list], list],
    as_int: Callable[[object, int], int],
    as_float: Callable[[object, float], float],
    auto_proxy_failover_max_switches: int,
    auto_replan_max_rounds: int,
    auto_recover_max_cycles: int,
) -> None:
    status = data.get("status", "idle")
    if status in ("running", "starting", "stopping"):
        status = "interrupted"

    state.update(
        {
            "status": status,
            "phase": data.get("phase", "idle"),
            "task_type": str(data.get("task_type", "自动识别") or "自动识别"),
            "profile_id": str(data.get("profile_id", "") or ""),
            "profile_topic": str(data.get("profile_topic", "") or ""),
            "continuous_mode": bool(data.get("continuous_mode", False)),
            "continuous_cycle": as_int(data.get("continuous_cycle", 0), 0),
            "continuous_max_cycles": as_int(data.get("continuous_max_cycles", 1), 1),
            "continuous_interval_sec": as_float(data.get("continuous_interval_sec", 0.0), 0.0),
            "continuous_prompt": str(data.get("continuous_prompt", "") or ""),
            "publish_every_cycles": as_int(data.get("publish_every_cycles", 4), 4),
            "publish_cooldown_sec": as_float(data.get("publish_cooldown_sec", 1800.0), 1800.0),
            "max_publish_per_session": as_int(data.get("max_publish_per_session", 2), 2),
            "publish_count": as_int(data.get("publish_count", 0), 0),
            "last_publish_ts": data.get("last_publish_ts"),
            "interaction_tasks": data.get("interaction_tasks", []),
            "interaction_task_index": as_int(data.get("interaction_task_index", 0), 0),
            "topic_anchor_required": bool(data.get("topic_anchor_required", False)),
            "topic_anchor_done": bool(data.get("topic_anchor_done", False)),
            "topic_anchor_input_hit": bool(data.get("topic_anchor_input_hit", False)),
            "topic_anchor_results_hit": bool(data.get("topic_anchor_results_hit", False)),
            "failover_budget": as_int(
                data.get("failover_budget", auto_proxy_failover_max_switches),
                auto_proxy_failover_max_switches,
            ),
            "failover_switches": as_int(data.get("failover_switches", 0), 0),
            "run_attempt": data.get("run_attempt", 0),
            "consecutive_failures": data.get("consecutive_failures", 0),
            "next_retry_at": data.get("next_retry_at"),
            "health": data.get("health", "unknown"),
            "health_detail": data.get("health_detail", ""),
            "health_last_checked": data.get("health_last_checked"),
            "messages": deserialize_messages(data.get("messages", [])),
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
            "goal_task": str(data.get("goal_task", "") or ""),
            "auto_replan_budget": as_int(
                data.get("auto_replan_budget", auto_replan_max_rounds),
                auto_replan_max_rounds,
            ),
            "auto_replan_used": as_int(data.get("auto_replan_used", 0), 0),
            "stagnation_rounds": as_int(data.get("stagnation_rounds", 0), 0),
            "auto_recover_budget": as_int(
                data.get("auto_recover_budget", auto_recover_max_cycles),
                auto_recover_max_cycles,
            ),
            "auto_recover_used": as_int(data.get("auto_recover_used", 0), 0),
            "plan": data.get("plan"),
            "plan_steps": data.get("plan_steps"),
            "plan_step_index": as_int(data.get("plan_step_index", 0), 0),
        }
    )

