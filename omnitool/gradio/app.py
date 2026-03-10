"""
python app.py --windows_host_url localhost:8006 --omniparser_server_url localhost:9000
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
import hashlib
import random

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
from llm_config import load_config, get_provider_config, get_all_providers
from agent.llm_utils.proxy_client import test_proxy_connection, run_proxy_interleaved
from agent.llm_utils.oaiclient import run_oai_interleaved
from agent.llm_utils.groqclient import run_groq_interleaved
import requests
from requests.exceptions import RequestException
import base64
from task_policy import (
    build_next_cycle_prompt as _build_next_cycle_prompt_locked,
    decide_next_cycle_strategy as _decide_next_cycle_strategy_locked,
    detect_publish_success as _detect_publish_success_locked,
    is_publish_task as _is_publish_task_locked,
    plan_is_incomplete as _plan_is_incomplete_locked,
)
from task_state_store import (
    apply_loaded_task_state,
    build_new_task_state,
    read_task_state,
    reset_task_state_fields,
    save_task_state,
)
from run_limits import max_seconds_label, normalize_max_seconds
from node_config import (
    DEFAULT_LOCAL_NODE_HOST,
    DEFAULT_REMOTE_NODE_HOSTS,
    node_id_from_host,
    node_label,
    resolve_node_hosts,
)
from runtime_log_monitor import RuntimeLogMonitor


def _ensure_localhost_no_proxy():
    local_hosts = {"localhost", "127.0.0.1", "::1"}
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        parts = [p.strip() for p in current.split(",") if p.strip()]
        merged = set(parts)
        merged.update(local_hosts)
        os.environ[key] = ",".join(sorted(merged))


_ensure_localhost_no_proxy()

DEBUG_LOGS = os.getenv("OMNITOOL_DEBUG", "").lower() in ("1", "true", "yes")

def _debug_print(*args, **kwargs):
    if DEBUG_LOGS:
        print(*args, **kwargs)

TASK_STATE_VERSION = 2
NODE_TASK_LOCKS: dict[str, threading.Lock] = {}
NODE_TASK_STATES: dict[str, dict] = {}
NODE_TASK_THREADS: dict[str, threading.Thread | None] = {}
MONITOR_CACHE: dict[str, dict] = {}
MONITOR_CACHE_LOCK = threading.Lock()
MONITOR_REFRESH_INTERVAL_SEC = float(os.getenv("OMNITOOL_MONITOR_REFRESH_INTERVAL_SEC", "2.5"))
MONITOR_REQUEST_TIMEOUT_SEC = float(os.getenv("OMNITOOL_MONITOR_REQUEST_TIMEOUT_SEC", "2.5"))
RUN_MAX_ATTEMPTS = max(1, int(os.getenv("OMNITOOL_RUN_MAX_ATTEMPTS", "3")))
RUN_RETRY_BASE_SEC = max(1.0, float(os.getenv("OMNITOOL_RUN_RETRY_BASE_SEC", "4")))
RUN_RETRY_MAX_SEC = max(RUN_RETRY_BASE_SEC, float(os.getenv("OMNITOOL_RUN_RETRY_MAX_SEC", "45")))
RUN_RETRY_JITTER = max(0.0, min(0.5, float(os.getenv("OMNITOOL_RUN_RETRY_JITTER", "0.15"))))
HEALTH_CHECK_INTERVAL_SEC = max(3.0, float(os.getenv("OMNITOOL_HEALTH_CHECK_INTERVAL_SEC", "12")))
MAINTENANCE_INTERVAL_SEC = max(30.0, float(os.getenv("OMNITOOL_MAINTENANCE_INTERVAL_SEC", "300")))
OUTPUT_RETENTION_HOURS = max(1.0, float(os.getenv("OMNITOOL_OUTPUT_RETENTION_HOURS", "24")))
OUTPUT_MAX_FILES_PER_NODE = max(50, int(os.getenv("OMNITOOL_OUTPUT_MAX_FILES_PER_NODE", "2000")))
LAST_HEALTH_CHECK_TS = 0.0
LAST_MAINTENANCE_TS = 0.0
MAINTENANCE_LOCK = threading.Lock()
OUTPUT_ROOT = Path("./tmp/outputs")
SUPERVISOR_INTERVAL_SEC = max(2.0, float(os.getenv("OMNITOOL_SUPERVISOR_INTERVAL_SEC", "2.5")))
SUPERVISOR_THREAD: threading.Thread | None = None
SUPERVISOR_STOP_EVENT = threading.Event()
AUTO_REPLAN_MAX_ROUNDS = max(0, int(os.getenv("OMNITOOL_AUTO_REPLAN_MAX_ROUNDS", "2")))
AUTO_REPLAN_STAGNATION_ROUNDS = max(1, int(os.getenv("OMNITOOL_AUTO_REPLAN_STAGNATION_ROUNDS", "2")))
AUTO_REPLAN_MIN_STEP_COUNT = max(2, int(os.getenv("OMNITOOL_AUTO_REPLAN_MIN_STEP_COUNT", "6")))
AUTO_RECOVER_MAX_CYCLES = max(0, int(os.getenv("OMNITOOL_AUTO_RECOVER_MAX_CYCLES", "2")))
CONTINUOUS_IGNORE_MAX_SECONDS = os.getenv("OMNITOOL_CONTINUOUS_IGNORE_MAX_SECONDS", "1").strip().lower() in ("1", "true", "yes", "on")
# 中转自动切换已禁用：保持用户选择的中转服务不被后台改写。
AUTO_PROXY_FAILOVER_ENABLED = False
AUTO_PROXY_FAILOVER_MAX_SWITCHES = 0
BACKEND_LOG_MODE = str(os.getenv("OMNITOOL_BACKEND_LOG_MODE", "compact") or "compact").strip().lower()
BACKEND_LOG_TEXT_MAX = max(80, int(os.getenv("OMNITOOL_BACKEND_LOG_TEXT_MAX", "220")))
BACKEND_LOG_STEP_HEARTBEAT_EVERY = max(0, int(os.getenv("OMNITOOL_BACKEND_LOG_STEP_HEARTBEAT_EVERY", "0")))
BACKEND_LOG_SUMMARY_INTERVAL_SEC = max(8.0, float(os.getenv("OMNITOOL_BACKEND_LOG_SUMMARY_INTERVAL_SEC", "25")))
RUNTIME_LOG_MONITOR = RuntimeLogMonitor(
    debug_logs=DEBUG_LOGS,
    backend_log_mode=BACKEND_LOG_MODE,
    text_max=BACKEND_LOG_TEXT_MAX,
    summary_interval_sec=BACKEND_LOG_SUMMARY_INTERVAL_SEC,
)

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


def _extract_task_keywords(task: str) -> set[str]:
    keywords: set[str] = set()
    lowered = (task or "").lower()
    for w in re.findall(r"[a-z0-9_]{3,}", lowered):
        keywords.add(w)
    for zh in re.findall(r"[\u4e00-\u9fff]{2,}", task or ""):
        keywords.add(zh)
    return keywords


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = PROJECT_ROOT / "docs"
FLOW_INDEX_PATH = DOCS_ROOT / "index" / "flow_index.md"
FLOW_DOCS_DIR = DOCS_ROOT / "flows"
FLOW_GUIDELINE_PATH = FLOW_DOCS_DIR / "flow_reference_guidelines.md"
INTENT_PROFILE_PATH = DOCS_ROOT / "index" / "intent_profiles.json"
TOPIC_CATALOG_PATH = DOCS_ROOT / "index" / "topic_catalog.json"
SHORT_INTENT_MAX_CHARS = max(12, int(os.getenv("OMNITOOL_SHORT_INTENT_MAX_CHARS", "36")))


def _split_index_tokens(raw_value: str) -> list[str]:
    if not raw_value:
        return []
    return [token.strip() for token in re.split(r"[,;，；|]+", raw_value) if token.strip()]


def _as_int(value, default: int) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return int(value)
    except Exception:
        return default


def _as_float(value, default: float) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return float(value)
    except Exception:
        return default


def _resolve_doc_path(path_text: str) -> Path:
    raw = str(path_text or "").strip()
    if not raw:
        return Path("")
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _display_doc_path(path_obj: Path | str) -> str:
    try:
        p = Path(path_obj)
        if not p.is_absolute():
            p = _resolve_doc_path(str(p))
        rel = p.relative_to(PROJECT_ROOT)
        return rel.as_posix()
    except Exception:
        return str(path_obj)


def _load_intent_profiles(profile_path: Path = INTENT_PROFILE_PATH) -> list[dict]:
    if not profile_path.exists():
        return []
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    profiles_raw = raw.get("profiles") if isinstance(raw, dict) else raw
    if not isinstance(profiles_raw, list):
        return []

    profiles: list[dict] = []
    for item in profiles_raw:
        if not isinstance(item, dict):
            continue
        required_tokens = item.get("required_tokens", [])
        task_template = item.get("task_template")
        if not isinstance(required_tokens, list) or not required_tokens:
            continue
        if not isinstance(task_template, str) or not task_template.strip():
            continue
        profiles.append(item)
    return profiles


def _contains_required_tokens(task_lower: str, required_tokens: list[str]) -> bool:
    return all(str(token).lower() in task_lower for token in required_tokens if str(token).strip())


def _looks_like_short_intent(task: str) -> bool:
    text = (task or "").strip()
    if not text:
        return False
    if len(text) <= SHORT_INTENT_MAX_CHARS:
        return True
    detail_markers = (
        "按 docs/",
        "正文",
        "标题",
        "四段",
        "动作确认门",
        "发布前",
        "success:",
        "step ",
    )
    lowered = text.lower()
    return not any(marker in lowered for marker in detail_markers)


def _resolve_profile_topic(task_lower: str, topic_aliases: dict, default_topic: str) -> str:
    if not isinstance(topic_aliases, dict):
        return default_topic
    for topic, aliases in topic_aliases.items():
        alias_list = aliases if isinstance(aliases, list) else []
        for alias in alias_list:
            alias_text = str(alias).strip().lower()
            if alias_text and alias_text in task_lower:
                return str(topic)
    return default_topic


def _extract_explicit_topic(task: str) -> str | None:
    text = (task or "").strip()
    if not text:
        return None

    for pattern in (
        r"(?:topic|主题|方向)\s*[:：=]\s*([a-zA-Z0-9_\-\u4e00-\u9fff]{2,40})",
        r"([a-zA-Z0-9_\-\u4e00-\u9fff]{2,40})\s*相关",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = str(match.group(1)).strip().strip(".,，。;；:：")
        candidate = re.sub(r"\s+", " ", candidate)
        candidate = re.sub(r"(?:小红书|养号|发布|笔记)$", "", candidate, flags=re.IGNORECASE).strip()
        if not candidate:
            continue
        if candidate.lower() in {"相关", "通用", "默认"}:
            continue
        return candidate
    return None


def _load_topic_catalog(path: Path = TOPIC_CATALOG_PATH) -> list[dict]:
    default_topics = [
        {"id": "openclaw", "label": "openclaw"},
        {"id": "dog_care", "label": "养狗"},
        {"id": "cooking", "label": "烹饪"},
        {"id": "education", "label": "教育"},
        {"id": "dev", "label": "开发"},
    ]
    if not path.exists():
        return default_topics
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_topics

    items = raw.get("topics", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return default_topics

    parsed: list[dict] = []
    for item in items:
        if isinstance(item, str):
            label = item.strip()
            if label:
                parsed.append({"id": label.lower(), "label": label})
            continue
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        if not label:
            continue
        parsed.append(
            {
                "id": str(item.get("id", label.lower())).strip(),
                "label": label,
                "aliases": item.get("aliases", []),
            }
        )
    return parsed or default_topics


def _expand_task_from_profiles(task: str) -> tuple[str, str | None, dict]:
    raw_task = (task or "").strip()
    if not raw_task:
        return raw_task, None, {}

    task_lower = raw_task.lower()
    profiles = _load_intent_profiles(INTENT_PROFILE_PATH)
    if not profiles:
        return raw_task, None, {}

    for profile in profiles:
        required_tokens = [str(token).strip() for token in profile.get("required_tokens", [])]
        if not _contains_required_tokens(task_lower, required_tokens):
            continue

        profile_id = str(profile.get("id", "unknown_profile"))
        default_topic = str(profile.get("default_topic", "通用"))
        explicit_topic = _extract_explicit_topic(raw_task)
        topic = explicit_topic or _resolve_profile_topic(task_lower, profile.get("topic_aliases", {}), default_topic)
        meta = {
            "profile_id": profile_id,
            "topic": topic,
            "continuous_mode": bool(profile.get("continuous", False)),
            "continuous_max_cycles": _as_int(profile.get("continuous_max_cycles", 1), 1),
            "continuous_interval_sec": _as_float(profile.get("continuous_interval_sec", 0.0), 0.0),
            "continuous_prompt_template": str(profile.get("continuous_prompt_template", "") or ""),
            "topic_anchor_required": bool(profile.get("topic_anchor_required", False)),
            "publish_every_cycles": _as_int(profile.get("publish_every_cycles", 4), 4),
            "publish_cooldown_sec": _as_float(profile.get("publish_cooldown_sec", 1800.0), 1800.0),
            "max_publish_per_session": _as_int(profile.get("max_publish_per_session", 2), 2),
            "interaction_tasks": [
                str(item).strip()
                for item in profile.get("interaction_tasks", [])
                if str(item).strip()
            ],
        }

        if not _looks_like_short_intent(raw_task):
            status = (
                f"🧩 意图模板命中：{profile_id} | 领域：{topic}\n"
                "检测到已提供详细描述，保留原始任务执行。"
            )
            return raw_task, status, meta

        template = str(profile.get("task_template", "")).strip()
        if not template:
            continue
        expanded = template.format(
            topic=topic,
            flow_doc=str(profile.get("flow_doc", "docs/flows/小红书_发布纯文本笔记流程.md")),
            strategy_doc=str(profile.get("strategy_doc", "docs/flows/flow_reference_guidelines.md")),
            original=raw_task,
        ).strip()
        if not expanded:
            continue
        status = (
            f"🧩 意图模板命中：{profile_id} | 领域：{topic}\n"
            "已将短指令扩展为标准执行任务（已注入模型上下文）。"
        )
        return expanded, status, meta

    return raw_task, None, {}


def _load_flow_index_entries(index_path: Path = FLOW_INDEX_PATH) -> list[dict]:
    if not index_path.exists():
        return []
    try:
        text = index_path.read_text(encoding="utf-8")
    except Exception:
        return []

    entries: list[dict] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue

        cols = [col.strip() for col in line.strip("|").split("|")]
        if len(cols) < 4:
            continue

        # Skip markdown table header and divider rows.
        if cols[0].lower() == "id":
            continue
        if all(re.fullmatch(r"[:\-\s]+", col or "") for col in cols[:4]):
            continue

        entry_id = cols[0]
        description = cols[1]
        keywords = _split_index_tokens(cols[2])
        docs_raw = _split_index_tokens(cols[3])
        if not entry_id or not keywords or not docs_raw:
            continue

        entries.append(
            {
                "id": entry_id,
                "description": description,
                "keywords": keywords,
                "docs": docs_raw,
            }
        )

    return entries


def _match_flow_index_entries(
    task: str,
    entries: list[dict],
    limit: int = 3,
    platform_target: str | None = None,
) -> list[dict]:
    task_text = (task or "").strip().lower()
    if not task_text or not entries:
        return []

    scored: list[tuple[int, dict]] = []
    for entry in entries:
        keywords = entry.get("keywords", [])
        hits = [kw for kw in keywords if str(kw).lower() in task_text]
        if not hits:
            continue
        # Hits + tie breaker for longer keyword matches.
        score = len(hits) * 10 + sum(len(h) for h in hits)
        enriched = {**entry, "hits": hits}
        if platform_target and (not _entry_matches_platform(enriched, platform_target)):
            continue
        scored.append((score, enriched))

    scored.sort(key=lambda item: (-item[0], item[1].get("id", "")))
    return [item[1] for item in scored[:limit]]


def _resolve_docs_from_index_matches(index_matches: list[dict]) -> list[Path]:
    resolved: list[Path] = []
    for match in index_matches:
        for rel_path in match.get("docs", []):
            path_obj = _resolve_doc_path(str(rel_path).strip())
            if path_obj.exists():
                resolved.append(path_obj)
    return resolved


PLATFORM_TARGETS: dict[str, dict[str, object]] = {
    "xiaohongshu": {
        "label": "小红书",
        "url": "https://www.xiaohongshu.com",
        "tokens": ("小红书", "xiaohongshu", "xhs", "红书"),
    },
    "douyin": {
        "label": "抖音",
        "url": "https://www.douyin.com",
        "tokens": ("抖音", "douyin", "iesdouyin"),
    },
    "kuaishou": {
        "label": "快手",
        "url": "https://www.kuaishou.com",
        "tokens": ("快手", "kuaishou", "kwai"),
    },
}


def _detect_platform_target(task: str) -> str | None:
    task_text = (task or "").strip().lower()
    if not task_text:
        return None

    explicit = re.search(
        r"(?:目标平台|platform)\s*[:：=]\s*(抖音|快手|小红书|douyin|kuaishou|kwai|xiaohongshu|xhs)",
        task_text,
        flags=re.IGNORECASE,
    )
    if explicit:
        value = str(explicit.group(1)).lower()
        if value in {"抖音", "douyin"}:
            return "douyin"
        if value in {"快手", "kuaishou", "kwai"}:
            return "kuaishou"
        if value in {"小红书", "xiaohongshu", "xhs"}:
            return "xiaohongshu"

    best_key = None
    best_score = 0
    for key, cfg in PLATFORM_TARGETS.items():
        tokens = cfg.get("tokens", ())
        score = sum(1 for token in tokens if str(token).lower() in task_text)
        if score > best_score:
            best_score = score
            best_key = key
    return best_key if best_score > 0 else None


def _entry_matches_platform(entry: dict, platform_target: str | None) -> bool:
    if not platform_target:
        return True
    if platform_target == "xiaohongshu":
        return True

    target_tokens = tuple(str(x).lower() for x in PLATFORM_TARGETS.get(platform_target, {}).get("tokens", ()))
    xhs_tokens = tuple(str(x).lower() for x in PLATFORM_TARGETS.get("xiaohongshu", {}).get("tokens", ()))

    merged = " ".join(
        [
            str(entry.get("id", "")),
            str(entry.get("description", "")),
            " ".join(str(x) for x in (entry.get("keywords") or [])),
            " ".join(str(x) for x in (entry.get("hits") or [])),
            " ".join(str(x) for x in (entry.get("docs") or [])),
        ]
    ).lower()

    has_target = any(token in merged for token in target_tokens)
    has_xhs = any(token in merged for token in xhs_tokens)
    if has_target:
        return True
    if has_xhs:
        return False
    return True


def _is_xhs_doc_path(path_obj: Path) -> bool:
    name = path_obj.name.lower()
    text = str(path_obj).lower()
    return ("xhs_" in name) or ("xiaohongshu" in text) or ("xhs_text_note" in name) or ("xhs_nurture" in name)


def _doc_matches_platform(path_obj: Path, platform_target: str | None) -> bool:
    if not platform_target or platform_target == "xiaohongshu":
        return True
    return not _is_xhs_doc_path(path_obj)


INTENT_FLOW_SOCKETS: dict[str, dict[str, object]] = {
    "xhs_nurture": {
        "description": "小红书养号流程（互动优先，按节奏发布）",
        "platform_tokens": (
            "小红书", "xiaohongshu", "xhs", "红书",
        ),
        "action_tokens": (
            "养号", "日常运营", "浏览", "点赞", "收藏", "评论", "互动",
            "nurture", "engage",
        ),
        "docs": (
            "docs/flows/小红书_养号流程.md",
            "docs/flows/flow_reference_guidelines.md",
            "docs/flows/vm134_quick_flow.md",
        ),
    },
    "xhs_publish": {
        "description": "小红书发布/创作流程",
        "platform_tokens": (
            "小红书", "xiaohongshu", "xhs", "红书",
        ),
        "action_tokens": (
            "发布", "发笔记", "发帖", "创作", "写长文", "图文", "内容发布",
            "publish", "post",
        ),
        "docs": (
            "docs/flows/小红书_发布纯文本笔记流程.md",
            "docs/flows/flow_reference_guidelines.md",
            "docs/flows/vm134_quick_flow.md",
        ),
    },
    "dy_publish": {
        "description": "抖音通用发布内容流程",
        "platform_tokens": (
            "抖音", "douyin", "iesdouyin",
        ),
        "action_tokens": (
            "发布", "内容", "文案", "文章", "图文", "流程", "创作", "publish", "post",
        ),
        "docs": (
            "docs/flows/抖音_发布内容索引.md",
            "docs/flows/抖音_通用发布内容流程.md",
            "docs/flows/flow_reference_guidelines.md",
        ),
    },
    "ks_publish": {
        "description": "快手发布流程学习与总结",
        "platform_tokens": (
            "快手", "kuaishou", "kwai",
        ),
        "action_tokens": (
            "发布", "文章", "图文", "流程", "总结", "学习", "创作", "publish", "post",
        ),
        "docs": (
            "docs/flows/快手_通用发布内容流程.md",
            "docs/flows/flow_reference_guidelines.md",
        ),
    },
}


def _detect_task_intent(task: str) -> str | None:
    task_text = (task or "").strip().lower()
    if not task_text:
        return None

    for intent, cfg in INTENT_FLOW_SOCKETS.items():
        platform_tokens = cfg.get("platform_tokens", ())
        action_tokens = cfg.get("action_tokens", ())
        has_platform = any(str(token).lower() in task_text for token in platform_tokens)
        has_action = any(str(token).lower() in task_text for token in action_tokens)
        if has_platform and has_action:
            return intent
    return None


def _resolve_intent_docs(task_intent: str | None) -> list[Path]:
    if not task_intent:
        return []
    cfg = INTENT_FLOW_SOCKETS.get(task_intent)
    if not cfg:
        return []
    resolved: list[Path] = []
    for rel in cfg.get("docs", ()):
        p = _resolve_doc_path(str(rel))
        if p.exists():
            resolved.append(p)
    return resolved


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def _find_matching_flow_docs(task: str, docs_dir: Path = FLOW_DOCS_DIR) -> list[Path]:
    if not docs_dir.exists():
        return []
    keywords = _extract_task_keywords(task)
    if not keywords:
        return []

    scored: list[tuple[int, Path]] = []
    for p in docs_dir.glob("*.md"):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        filename = p.name.lower()
        body = text.lower()
        score = 0
        for k in keywords:
            kl = k.lower()
            if kl in filename:
                score += 3
            if kl in body:
                score += 1
        if score > 0:
            scored.append((score, p))

    scored.sort(key=lambda x: (-x[0], len(x[1].name)))
    return [p for _, p in scored[:3]]


def _build_flow_reference_message(task: str) -> tuple[str | None, list[str], str]:
    platform_target = _detect_platform_target(task)
    platform_label = str(PLATFORM_TARGETS.get(platform_target, {}).get("label", "")) if platform_target else ""

    index_entries = _load_flow_index_entries(FLOW_INDEX_PATH)
    index_matches = _match_flow_index_entries(task, index_entries, platform_target=platform_target)
    index_docs = _resolve_docs_from_index_matches(index_matches)

    task_intent = _detect_task_intent(task)
    if platform_target and platform_target != "xiaohongshu" and task_intent and task_intent.startswith("xhs_"):
        task_intent = None
    intent_docs = _resolve_intent_docs(task_intent)
    matched = index_docs + intent_docs + _find_matching_flow_docs(task)
    if platform_target and platform_target != "xiaohongshu":
        matched = [p for p in matched if _doc_matches_platform(p, platform_target)]
    guideline = FLOW_GUIDELINE_PATH
    if guideline.exists():
        matched = [guideline] + matched
    matched = _dedupe_paths(matched)

    index_display = _display_doc_path(FLOW_INDEX_PATH)
    index_hit_ids = [str(match.get("id", "")) for match in index_matches if match.get("id")]
    index_state = "命中" if index_hit_ids else "未命中"
    status_lines = [
        "📚 文档预读检查",
        f"索引: {index_display} ({index_state})",
    ]
    if platform_label:
        status_lines.append(f"平台判定: {platform_label}")
    if index_hit_ids:
        status_lines.append("命中条目: " + ", ".join(index_hit_ids))

    if not matched:
        status_lines.append("加载文档: 无")
        return None, [], "\n".join(status_lines)

    snippets: list[str] = []
    used_paths: list[str] = []
    if FLOW_INDEX_PATH.exists():
        used_paths.append(index_display)
    for p in matched:
        try:
            raw = p.read_text(encoding="utf-8")
        except Exception:
            continue
        normalized = re.sub(r"\s+", " ", raw).strip()
        if not normalized:
            continue
        used_paths.append(_display_doc_path(p))
        snippets.append(f"[{_display_doc_path(p)}] {normalized[:1000]}")

    if not snippets:
        loaded_docs = [p for p in used_paths if p != index_display]
        status_lines.append("加载文档: " + ("; ".join(loaded_docs) if loaded_docs else "无"))
        return None, used_paths, "\n".join(status_lines)

    intent_tip = ""
    if task_intent:
        cfg = INTENT_FLOW_SOCKETS.get(task_intent, {})
        desc = str(cfg.get("description", task_intent))
        intent_tip = f"识别到任务意图：{desc}（{task_intent}），已优先注入对应流程插座。\n"

    index_tip = ""
    if FLOW_INDEX_PATH.exists():
        if index_matches:
            hit_lines = [
                f"- {match.get('id')}: {match.get('description', '')} | 命中关键词: {', '.join(match.get('hits', []))}"
                for match in index_matches
            ]
            index_tip = (
                f"已检查索引文档：{index_display}\n"
                "索引命中条目：\n"
                + "\n".join(hit_lines)
                + "\n"
            )
        else:
            index_tip = f"已检查索引文档：{index_display}，未命中专用条目，使用通用流程文档。\n"

    platform_tip = ""
    if platform_target and platform_target != "xiaohongshu":
        platform_url = str(PLATFORM_TARGETS.get(platform_target, {}).get("url", "") or "")
        if platform_label and platform_url:
            platform_tip = (
                f"平台硬约束：目标平台为{platform_label}，第一步必须先打开 {platform_url}。"
                "禁止沿用小红书创作页上下文。\n"
            )

    guidance = (
        "默认第1步：先检查并预读流程文档，再执行UI动作。\n"
        + index_tip
        + intent_tip
        + platform_tip
        + "以下为本任务命中文档摘要，请优先遵循：\n"
        + "\n".join(snippets)
        + "\n若文档与页面不一致，先执行文档中的恢复/校验步骤，再继续下一步。"
    )
    loaded_docs = [p for p in used_paths if p != index_display]
    status_lines.append("加载文档: " + ("; ".join(loaded_docs) if loaded_docs else "无"))
    return guidance, used_paths, "\n".join(status_lines)


def _is_path_within(base_dir: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base_dir.resolve())
        return True
    except Exception:
        return False


def _list_manageable_flow_docs() -> list[str]:
    candidates: list[Path] = [
        FLOW_INDEX_PATH,
        INTENT_PROFILE_PATH,
        TOPIC_CATALOG_PATH,
        FLOW_GUIDELINE_PATH,
    ]
    if FLOW_DOCS_DIR.exists():
        for path_obj in sorted(FLOW_DOCS_DIR.glob("*.md"), key=lambda p: p.name.lower()):
            candidates.append(path_obj)

    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path_obj in candidates:
        resolved = _resolve_doc_path(str(path_obj))
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if _is_path_within(DOCS_ROOT, resolved):
            unique_paths.append(resolved)
    return [_display_doc_path(path_obj) for path_obj in unique_paths]


def _resolve_manageable_doc(path_text: str) -> Path | None:
    if not str(path_text or "").strip():
        return None
    candidate = _resolve_doc_path(path_text)
    if not _is_path_within(DOCS_ROOT, candidate):
        return None
    return candidate


def _refresh_flow_doc_editor(selected_doc: str):
    doc_choices = _list_manageable_flow_docs()
    if not doc_choices:
        return gr.update(choices=[], value=None), "❌ 未找到可管理文档。"

    selected = str(selected_doc or "").strip()
    if selected not in doc_choices:
        selected = doc_choices[0]
    return gr.update(choices=doc_choices, value=selected), f"📚 可管理文档已刷新（{len(doc_choices)} 项）。"


def _load_flow_doc_editor(selected_doc: str):
    selected = str(selected_doc or "").strip()
    path_obj = _resolve_manageable_doc(selected)
    if not path_obj:
        return "", "❌ 文档路径非法或不在 docs 目录下。"

    if not path_obj.exists():
        return "", f"⚠️ 文档不存在：{_display_doc_path(path_obj)}"
    try:
        content = path_obj.read_text(encoding="utf-8")
        return content, f"✅ 已加载：{_display_doc_path(path_obj)}"
    except Exception as e:
        return "", f"❌ 加载失败：{_display_doc_path(path_obj)} | {e}"


def _save_flow_doc_editor(selected_doc: str, content: str):
    selected = str(selected_doc or "").strip()
    path_obj = _resolve_manageable_doc(selected)
    if not path_obj:
        return "❌ 保存失败：文档路径非法或不在 docs 目录下。"
    try:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(str(content or ""), encoding="utf-8")
        return f"✅ 已保存：{_display_doc_path(path_obj)}（{len(str(content or ''))} 字符）"
    except Exception as e:
        return f"❌ 保存失败：{_display_doc_path(path_obj)} | {e}"

def get_proxy_choices():
    """获取中转 provider 选项"""
    providers = get_all_providers()
    return [(v["name"], k) for k, v in providers.items()]

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
    parser.add_argument(
        "--windows_host_urls",
        type=str,
        default=os.getenv(
            "OMNITOOL_WINDOWS_HOST_URLS",
            DEFAULT_REMOTE_NODE_HOSTS,
        ),
    )
    parser.add_argument("--omniparser_server_url", type=str, default="localhost:9000")
    parser.add_argument("--local", action="store_true", help="本地模式，直接控制本机桌面")
    parser.add_argument(
        "--local_node_host",
        type=str,
        default=os.getenv("OMNITOOL_LOCAL_NODE_HOST", DEFAULT_LOCAL_NODE_HOST),
        help="多节点模式下自动接入的本机节点地址",
    )
    parser.add_argument(
        "--exclude_local_node",
        action="store_true",
        help="多节点模式下不自动接入本机节点",
    )
    parser.add_argument("--max_steps", type=int, default=int(os.getenv("OMNITOOL_MAX_STEPS", "80")))
    parser.add_argument("--max_seconds", type=int, default=int(os.getenv("OMNITOOL_MAX_SECONDS", "0")))
    return parser.parse_args()
args = parse_arguments()


def _resolve_node_hosts() -> list[str]:
    raw_hosts = args.windows_host_urls or args.windows_host_url
    return resolve_node_hosts(
        raw_hosts=str(raw_hosts or ""),
        fallback_host=args.windows_host_url,
        include_local_node=not args.exclude_local_node,
        local_node_host=args.local_node_host,
        local_mode=args.local,
    )


NODE_HOSTS = _resolve_node_hosts()
NODE_IDS = [node_id_from_host(host) for host in NODE_HOSTS]
NODE_MAP = dict(zip(NODE_IDS, NODE_HOSTS))
DEFAULT_NODE_ID = NODE_IDS[0] if NODE_IDS else "local"


def _new_task_state(node_id: str, windows_host_url: str) -> dict:
    return build_new_task_state(
        node_id=node_id,
        windows_host_url=windows_host_url,
        auto_proxy_failover_max_switches=AUTO_PROXY_FAILOVER_MAX_SWITCHES,
        auto_replan_max_rounds=AUTO_REPLAN_MAX_ROUNDS,
        auto_recover_max_cycles=AUTO_RECOVER_MAX_CYCLES,
    )


def _save_node_task_state(node_id: str):
    state = NODE_TASK_STATES[node_id]
    save_task_state(
        node_id=node_id,
        state=state,
        version=TASK_STATE_VERSION,
        serialize_messages=_serialize_messages,
        auto_proxy_failover_max_switches=AUTO_PROXY_FAILOVER_MAX_SWITCHES,
        auto_replan_max_rounds=AUTO_REPLAN_MAX_ROUNDS,
        auto_recover_max_cycles=AUTO_RECOVER_MAX_CYCLES,
    )


def _load_node_task_state(node_id: str):
    data = read_task_state(node_id)
    if data is None:
        return

    state = NODE_TASK_STATES[node_id]
    apply_loaded_task_state(
        state=state,
        data=data,
        deserialize_messages=_deserialize_messages,
        as_int=_as_int,
        as_float=_as_float,
        auto_proxy_failover_max_switches=AUTO_PROXY_FAILOVER_MAX_SWITCHES,
        auto_replan_max_rounds=AUTO_REPLAN_MAX_ROUNDS,
        auto_recover_max_cycles=AUTO_RECOVER_MAX_CYCLES,
    )


def _init_node_runtime():
    if args.local:
        node_id = DEFAULT_NODE_ID
        NODE_TASK_LOCKS[node_id] = threading.Lock()
        NODE_TASK_STATES[node_id] = _new_task_state(node_id, "")
        NODE_TASK_THREADS[node_id] = None
        return
    for node_id, host in NODE_MAP.items():
        NODE_TASK_LOCKS[node_id] = threading.Lock()
        NODE_TASK_STATES[node_id] = _new_task_state(node_id, host)
        NODE_TASK_THREADS[node_id] = None
        _load_node_task_state(node_id)


_init_node_runtime()


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
        state["zhipu_api_key"] = os.getenv("ZHIPU_API_KEY", "")

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

def _api_response_callback(response: APIResponse[BetaMessage], response_state: dict):
    response_id = datetime.now().isoformat()
    response_state[response_id] = response

def _tool_output_callback(tool_output: ToolResult, tool_id: str, tool_state: dict):
    tool_state[tool_id] = tool_output

_ANALYSIS_NOISE_PREFIXES = (
    "Next I will perform the following action:",
    "Box ID:",
    "From Box ID:",
    "To Box ID:",
    "box_centroid_coordinate:",
    "value:",
)

_TOOL_RESULT_HIDE_PREFIXES = (
    "Moved mouse to",
)

def _compact_line(text: str, max_len: int = 180) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."

def _normalize_chat_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "")).lower()

def _dedupe_append(chatbot_state: list[tuple], sender: str, message: str):
    if not message:
        return
    message = message.strip()
    if not message:
        return

    if sender == "bot":
        if chatbot_state:
            prev_user, prev_bot = chatbot_state[-1]
            if prev_bot and _normalize_chat_text(prev_bot) == _normalize_chat_text(message):
                return
        chatbot_state.append((None, message))
    else:
        if chatbot_state:
            prev_user, prev_bot = chatbot_state[-1]
            if prev_user and prev_bot is None and _normalize_chat_text(prev_user) == _normalize_chat_text(message):
                return
        chatbot_state.append((message, None))

def _sanitize_analysis_text(text: str) -> str | None:
    if not text:
        return None

    lines = [line.strip() for line in text.replace("\r", "").split("\n")]
    lines = [line for line in lines if line]
    if not lines:
        return None

    thought_line = None
    action_line = None
    warning_lines: list[str] = []
    fallback_lines: list[str] = []

    for line in lines:
        if line.startswith("<img"):
            continue
        if line.startswith("💭"):
            if thought_line is None:
                thought_line = line
            continue
        if line.startswith("📋 动作"):
            if action_line is None:
                action_line = line
            continue
        if line.startswith("Analysis:"):
            if thought_line is None:
                thought = line.split("Analysis:", 1)[1].strip()
                if thought:
                    thought_line = f"💭 {thought}"
            continue
        if line.startswith("Next Action:"):
            if action_line is None:
                next_action = line.split("Next Action:", 1)[1].strip()
                if next_action:
                    action_line = f"📋 动作: {next_action}"
            continue
        if line.startswith(_ANALYSIS_NOISE_PREFIXES):
            continue
        if line.startswith("⚠️"):
            warning_lines.append(line)
            continue
        if line.startswith("🔄 Step"):
            fallback_lines.append(line)
            continue
        fallback_lines.append(line)

    rendered_lines: list[str] = []
    if thought_line:
        rendered_lines.append(_compact_line(thought_line))
    if action_line:
        rendered_lines.append(_compact_line(action_line))
    rendered_lines.extend([_compact_line(line) for line in warning_lines])

    if not rendered_lines:
        rendered_lines = [_compact_line(line) for line in fallback_lines[:3]]
    if not rendered_lines:
        return None

    text_out = "\n".join(rendered_lines).strip()
    if len(text_out) > 1200:
        text_out = text_out[:1200].rstrip() + "..."
    return text_out

def _sanitize_tool_result(message: ToolResult, hide_images=False) -> str | None:
    if message.output:
        output_text = str(message.output).strip()
        if not output_text:
            return None
        if output_text.startswith(_TOOL_RESULT_HIDE_PREFIXES):
            return None
        if output_text.startswith("Next I will perform the following action:"):
            return None

        # 大段输入回显会淹没聊天区，只保留摘要。
        if "\n" in output_text and len(output_text) > 240:
            return f"⌨️ 已输入文本（{len(output_text)} 字符）"
        if len(output_text) > 320:
            return output_text[:320] + "..."
        return output_text
    if message.error:
        return f"❌ {message.error}"
    if message.base64_image:
        return None
    return None

def chatbot_output_callback(message, chatbot_state, hide_images=False, sender="bot"):
    def _render_message(message: str | BetaTextBlock | BetaToolUseBlock | ToolResult, hide_images=False):
        _debug_print(f"_render_message: {str(message)[:100]}")

        if isinstance(message, str):
            return _sanitize_analysis_text(message)

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
            return None
        # render tool result
        if is_tool_result:
            message = cast(ToolResult, message)
            return _sanitize_tool_result(message, hide_images=hide_images)

        elif isinstance(message, BetaTextBlock) or isinstance(message, TextBlock):
            return _sanitize_analysis_text(message.text) or f"💭 {message.text}"
        elif isinstance(message, BetaToolUseBlock) or isinstance(message, ToolUseBlock):
            # 工具调用详情较噪音，聊天区不再重复展示（动作通常已在分析中给出）。
            return None
        else:
            return message

    def _truncate_string(s, max_length=500):
        """Truncate long strings for concise printing."""
        if isinstance(s, str) and len(s) > max_length:
            return s[:max_length] + "..."
        return s
    # processing Anthropic messages
    message = _render_message(message, hide_images)

    if not message:
        return
    if not isinstance(message, str):
        message = str(message)
    _dedupe_append(chatbot_state, sender=sender, message=message)

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
PHASE_LABELS = {
    "idle": "空闲",
    "preprocessing": "预处理中",
    "planning": "规划中",
    "executing": "执行中",
    "recovering": "恢复中",
    "stopping": "停止中",
    "completed": "已完成",
    "failed": "失败",
}
RECOVERY_PHASE_PATTERNS = (
    "动作确认门",
    "恢复动作",
    "未检测到有效变化",
    "请输入标题",
    "focus probe failed",
    "retry",
    "重试",
    "回滚",
)
PHASE_STEP_PATTERNS = ("🔄 step",)

ACTIVE_NODE_IDS = NODE_IDS if NODE_IDS else [DEFAULT_NODE_ID]
ALL_NODES_ID = "__all__"
TRANSIENT_ERROR_KEYWORDS = (
    "connection aborted",
    "connection reset",
    "connection refused",
    "failed to establish a new connection",
    "max retries exceeded",
    "httpsconnectionpool",
    "name or service not known",
    "temporary failure in name resolution",
    "timed out",
    "read timed out",
    "connect timeout",
    "你的主机中的软件中止了一个已建立的连接",
    "中止了一个已建立的连接",
    "连接中止",
    "连接超时",
    "连接被拒绝",
)
ERROR_AUTO_RESET_COOLDOWN_SEC = float(os.getenv("OMNITOOL_ERROR_AUTO_RESET_COOLDOWN_SEC", "8"))
HEALTH_ICONS = {
    "healthy": "🟢",
    "degraded": "🟡",
    "offline": "🔴",
    "unknown": "⚪",
}

def _build_replan_prompt_locked(state: dict, reason: str, cycle: int, error_text: str = "") -> str:
    goal = str(state.get("goal_task") or "").strip()
    if not goal:
        goal = str(state.get("profile_topic") or "当前任务目标").strip()
    profile = str(state.get("profile_id") or "").strip()
    plan_steps = state.get("plan_steps") or []
    plan_idx = int(state.get("plan_step_index") or 0)
    remaining = 0
    if isinstance(plan_steps, list) and plan_steps:
        remaining = max(0, len(plan_steps) - plan_idx)

    details = f"触发原因：{reason}。当前轮次：{cycle}。"
    if remaining > 0:
        details += f" 旧计划剩余约 {remaining} 步。"
    if error_text:
        details += f" 最近错误：{error_text[:220]}"
    profile_hint = f"任务画像：{profile}。" if profile else ""

    return (
        f"{goal}\n"
        f"{profile_hint}{details}\n"
        "请重新输出一个更稳健的 3-7 步执行计划：\n"
        "1) 优先恢复到正确上下文（不要重复点击同一点）；\n"
        "2) 在搜索结果/推荐流里，连续点击无效时必须先滚动（scroll_down 或 PageDown）再选新卡片；\n"
        "3) 每步必须可观察成功条件（屏幕文字/控件）；\n"
        "4) 若离终点较远，先执行路径收敛动作再继续目标；\n"
        "5) 输出格式必须为：Step N: 动作描述 | Success: 关键词1, 关键词2"
    )


def _auto_replan(node_id: str, reason: str, error_text: str = "") -> bool:
    lock = NODE_TASK_LOCKS[node_id]
    state = NODE_TASK_STATES[node_id]

    with lock:
        used = int(state.get("auto_replan_used") or 0)
        budget = max(0, _as_int(state.get("auto_replan_budget", AUTO_REPLAN_MAX_ROUNDS), AUTO_REPLAN_MAX_ROUNDS))
        if used >= budget:
            return False
        cycle = int(state.get("continuous_cycle") or 1)
        prompt = _build_replan_prompt_locked(state, reason=reason, cycle=cycle, error_text=error_text)
        state["auto_replan_used"] = used + 1
        state["messages"].append({"role": Sender.USER, "content": [TextBlock(type="text", text=prompt)]})
        state["chatbot_messages"].append(
            (None, f"♻️ 自动重规划已触发（{state['auto_replan_used']}/{budget}）：{reason}")
        )
        _set_phase_locked(state, "planning", announce=True, detail="检测到偏航/失败，正在自动重规划。")
        state["status"] = "running"
        state["last_update"] = time.time()
        _save_node_task_state(node_id)

    plan_text = _generate_plan(prompt, state)
    plan_steps = _parse_plan_steps(plan_text)

    with lock:
        if plan_steps:
            state["plan"] = plan_text
            state["plan_steps"] = plan_steps
            state["plan_step_index"] = 0
            state["stagnation_rounds"] = 0
            state["chatbot_messages"].append((None, f"🗺️ 自动重规划完成：共 {len(plan_steps)} 步"))
            state["last_update"] = time.time()
            _save_node_task_state(node_id)
            return True

        # 即便未生成结构化计划，也继续执行，避免立即终止。
        state["plan"] = None
        state["plan_steps"] = None
        state["plan_step_index"] = 0
        state["chatbot_messages"].append((None, "🗺️ 自动重规划降级：未生成结构化计划，继续按恢复策略执行。"))
        state["last_update"] = time.time()
        _save_node_task_state(node_id)
        return True


def _target_node_ids(node_id: str | None) -> list[str]:
    if node_id == ALL_NODES_ID:
        return list(ACTIVE_NODE_IDS)
    if node_id in NODE_TASK_STATES:
        return [str(node_id)]
    return [DEFAULT_NODE_ID]


def _primary_node_id(node_id: str | None) -> str:
    return _target_node_ids(node_id)[0]


def _all_nodes_status_text() -> str:
    counts: dict[str, int] = {}
    health_counts: dict[str, int] = {}
    for node in ACTIVE_NODE_IDS:
        s = NODE_TASK_STATES[node].get("status", "idle")
        counts[s] = counts.get(s, 0) + 1
        h = NODE_TASK_STATES[node].get("health", "unknown")
        health_counts[h] = health_counts.get(h, 0) + 1
    parts = [f"{k}:{v}" for k, v in sorted(counts.items())]
    health_parts = [f"{k}:{v}" for k, v in sorted(health_counts.items())]
    return f"`ALL` 📡 **广播模式** | " + " | ".join(parts) + " | 🩺 " + ", ".join(health_parts)


def _phase_label(phase: str | None) -> str:
    return PHASE_LABELS.get(str(phase or "idle"), str(phase or "idle"))


def _set_phase_locked(state: dict, phase: str, announce: bool = False, detail: str | None = None):
    current = state.get("phase", "idle")
    if current == phase:
        return
    state["phase"] = phase
    state["last_update"] = time.time()
    if announce:
        msg = f"🧭 阶段：{_phase_label(phase)}"
        if detail:
            msg += f"\n{detail}"
        state["chatbot_messages"].append((None, msg))


def _task_status_text(node_id: str):
    state = NODE_TASK_STATES[node_id]
    status = state.get("status", "idle")
    phase = state.get("phase", "idle")
    health = state.get("health", "unknown")
    last_error = state.get("last_error")
    plan_steps = state.get("plan_steps") or []
    plan_index = state.get("plan_step_index", 0)
    host = state.get("windows_host_url") or "local"
    run_attempt = int(state.get("run_attempt") or 0)
    next_retry_at = state.get("next_retry_at")
    auto_replan_used = int(state.get("auto_replan_used") or 0)
    auto_replan_budget = _as_int(state.get("auto_replan_budget", AUTO_REPLAN_MAX_ROUNDS), AUTO_REPLAN_MAX_ROUNDS)
    failover_switches = _as_int(state.get("failover_switches", 0), 0)
    failover_budget = _as_int(state.get("failover_budget", AUTO_PROXY_FAILOVER_MAX_SWITCHES), AUTO_PROXY_FAILOVER_MAX_SWITCHES)
    continuous_mode = bool(state.get("continuous_mode", False))
    continuous_cycle = int(state.get("continuous_cycle") or 0)
    continuous_max_cycles = int(state.get("continuous_max_cycles") or 0)

    icon = STATUS_ICONS.get(status, "⚪")
    phase_part = f" | 🧭 {_phase_label(phase)}"
    health_icon = HEALTH_ICONS.get(str(health), "⚪")
    health_part = f" | 🩺 {health_icon}{health}"
    plan_part = ""
    if plan_steps:
        plan_part = f" | 📋 {min(plan_index + 1, len(plan_steps))}/{len(plan_steps)}"
    retry_part = f" | 🔁 A{run_attempt}/{RUN_MAX_ATTEMPTS}" if run_attempt > 0 else ""
    replan_part = ""
    if auto_replan_budget > 0:
        replan_part = f" | ♻️ RP{auto_replan_used}/{auto_replan_budget}"
    failover_part = ""
    if failover_budget > 0:
        failover_part = f" | 🔀 FO{failover_switches}/{failover_budget}"
    cycle_part = ""
    if continuous_mode:
        if continuous_max_cycles > 0:
            cycle_part = f" | ♻️ C{max(1, continuous_cycle)}/{continuous_max_cycles}"
        else:
            cycle_part = f" | ♻️ C{max(1, continuous_cycle)}/∞"
    backoff_part = ""
    if isinstance(next_retry_at, (int, float)) and next_retry_at > time.time():
        wait_s = int(max(0, next_retry_at - time.time()))
        backoff_part = f" | ⏳ {wait_s}s"

    last_update = state.get("last_update")
    updated_part = ""
    if last_update:
        updated_part = f" | 🕐 {datetime.fromtimestamp(last_update).strftime('%H:%M:%S')}"

    messages_count = len(state.get("chatbot_messages", []))
    msg_part = f" | 💬 {messages_count}" if messages_count > 0 else ""

    if status == "error" and last_error:
        err = str(last_error).replace("\n", " ").strip()
        if len(err) > 140:
            err = err[:140].rstrip() + "..."
        return f"`{host}` {icon} **错误** - {err}{health_part}{phase_part}{cycle_part}{retry_part}{replan_part}{failover_part}{plan_part}{backoff_part}{msg_part}{updated_part}"

    status_text_map = {
        "idle": "空闲",
        "starting": "启动中",
        "running": "执行中",
        "stopping": "停止中",
        "stopped": "已停止",
        "completed": "已完成",
        "interrupted": "已中断",
    }
    status_cn = status_text_map.get(status, status)
    return f"`{host}` {icon} **{status_cn}**{health_part}{phase_part}{cycle_part}{retry_part}{replan_part}{failover_part}{plan_part}{backoff_part}{msg_part}{updated_part}"


def _all_status_values() -> list[str]:
    return [_task_status_text(node_id) for node_id in ACTIVE_NODE_IDS]


def _status_outputs(selected_node_id: str) -> tuple:
    primary = _primary_node_id(selected_node_id)
    main_status = _all_nodes_status_text() if selected_node_id == ALL_NODES_ID else _task_status_text(primary)
    return (main_status, *_all_status_values())

def _all_node_chat_values() -> list[list[tuple]]:
    values: list[list[tuple]] = []
    for node_id in ACTIVE_NODE_IDS:
        lock = NODE_TASK_LOCKS[node_id]
        with lock:
            values.append(list(NODE_TASK_STATES[node_id].get("chatbot_messages", [])))
    return values

def _chat_outputs(selected_node_id: str | None) -> tuple:
    primary = _primary_node_id(selected_node_id)
    lock = NODE_TASK_LOCKS[primary]
    with lock:
        main_chat = list(NODE_TASK_STATES[primary].get("chatbot_messages", []))
    if selected_node_id == ALL_NODES_ID:
        main_chat = [*main_chat, (None, "📡 当前为广播模式，主会话显示首节点历史。")]
    return (main_chat, *_all_node_chat_values())

def _is_transient_error_text(err: str | None) -> bool:
    if not err:
        return False
    lowered = str(err).lower()
    if any(keyword in lowered for keyword in TRANSIENT_ERROR_KEYWORDS):
        return True
    if "llm request failed" in lowered and ("中转 api 请求失败" in lowered or "proxy" in lowered):
        return True
    return False


def _is_proxy_failover_error(err: str | None) -> bool:
    if not err:
        return False
    lowered = str(err).lower()
    patterns = (
        "中转 api 请求失败",
        "bad gateway",
        "502",
        "gateway",
        "httpsconnectionpool",
        "connection aborted",
        "max retries exceeded",
        "proxy",
    )
    return any(pattern in lowered for pattern in patterns)


def _next_proxy_provider(current_provider: str, tried: set[str] | None = None) -> tuple[str, dict] | None:
    providers = get_all_providers() or {}
    if not providers:
        return None
    keys = list(providers.keys())
    if not keys:
        return None

    if current_provider in keys:
        idx = keys.index(current_provider)
        ordered = keys[idx + 1 :] + keys[:idx]
    else:
        ordered = keys

    tried_set = tried or set()
    for key in ordered:
        if key in tried_set:
            continue
        cfg = providers.get(key, {}) or {}
        base_url = str(cfg.get("base_url", "") or "").strip()
        api_key = str(cfg.get("api_key", "") or "").strip()
        if not base_url or not api_key:
            continue
        return key, cfg
    return None


def _compute_retry_delay_seconds(attempt: int) -> float:
    # Exponential backoff + jitter, inspired by OpenClaw retry policy.
    base = RUN_RETRY_BASE_SEC * (2 ** max(0, attempt - 1))
    delay = min(RUN_RETRY_MAX_SEC, base)
    jitter = delay * RUN_RETRY_JITTER
    if jitter <= 0:
        return delay
    return max(1.0, delay + random.uniform(-jitter, jitter))


def _sleep_with_stop(node_id: str, seconds: float, check_interval: float = 0.5) -> bool:
    deadline = time.time() + max(0.0, seconds)
    while time.time() < deadline:
        lock = NODE_TASK_LOCKS[node_id]
        with lock:
            if NODE_TASK_STATES[node_id].get("stop"):
                return True
        time.sleep(min(check_interval, max(0.01, deadline - time.time())))
    return False


def _probe_url(url: str, timeout: float = 3.0) -> tuple[bool, str]:
    try:
        resp = requests.get(url, timeout=timeout, proxies={"http": "", "https": ""})
        if resp.status_code == 200:
            return True, "ok"
        return False, f"http {resp.status_code}"
    except Exception as exc:
        return False, str(exc)


def _check_node_health(node_id: str, timeout: float = 2.5) -> tuple[str, str]:
    host = NODE_TASK_STATES[node_id].get("windows_host_url") or ""
    omni_ok, omni_detail = _probe_url(f"http://{args.omniparser_server_url}/probe", timeout=timeout)

    host_ok = True
    host_detail = "local mode"
    if not args.local and host:
        host_ok, host_detail = _probe_url(f"http://{host}/probe", timeout=timeout)

    if omni_ok and host_ok:
        return "healthy", "omniparser+node ok"
    if omni_ok or host_ok:
        return "degraded", f"omni={omni_detail}; node={host_detail}"
    return "offline", f"omni={omni_detail}; node={host_detail}"


def _refresh_node_health_if_due(force: bool = False):
    global LAST_HEALTH_CHECK_TS
    now = time.time()
    if not force and now - LAST_HEALTH_CHECK_TS < HEALTH_CHECK_INTERVAL_SEC:
        return
    LAST_HEALTH_CHECK_TS = now

    for node_id in ACTIVE_NODE_IDS:
        health, detail = _check_node_health(node_id)
        lock = NODE_TASK_LOCKS[node_id]
        with lock:
            prev = NODE_TASK_STATES[node_id].get("health")
            NODE_TASK_STATES[node_id]["health"] = health
            NODE_TASK_STATES[node_id]["health_detail"] = detail
            NODE_TASK_STATES[node_id]["health_last_checked"] = now
            # Only announce on degradation transitions to avoid log noise.
            if prev and prev != health and health in ("degraded", "offline"):
                NODE_TASK_STATES[node_id]["chatbot_messages"].append(
                    (
                        None,
                        f"🩺 节点健康变更：{health} ({detail})",
                    )
                )
            _save_node_task_state(node_id)


def _preflight_runtime(node_id: str) -> list[str]:
    health, detail = _check_node_health(node_id, timeout=3.0)
    if health == "healthy":
        return []
    if health == "degraded":
        return [f"runtime degraded: {detail}"]
    return [f"runtime offline: {detail}"]


def _cleanup_output_dir_for_node(node_id: str) -> tuple[int, int]:
    output_dir = OUTPUT_ROOT / node_id
    if not output_dir.exists():
        return 0, 0

    now = time.time()
    cutoff = now - OUTPUT_RETENTION_HOURS * 3600
    removed_by_age = 0
    removed_by_cap = 0

    files = [p for p in output_dir.glob("*") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime)

    for file_path in files:
        try:
            if file_path.stat().st_mtime < cutoff:
                file_path.unlink(missing_ok=True)
                removed_by_age += 1
        except Exception:
            continue

    files = [p for p in output_dir.glob("*") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime)
    overflow = max(0, len(files) - OUTPUT_MAX_FILES_PER_NODE)
    for file_path in files[:overflow]:
        try:
            file_path.unlink(missing_ok=True)
            removed_by_cap += 1
        except Exception:
            continue

    return removed_by_age, removed_by_cap


def _run_maintenance_if_due(force: bool = False):
    global LAST_MAINTENANCE_TS
    now = time.time()
    if not force and now - LAST_MAINTENANCE_TS < MAINTENANCE_INTERVAL_SEC:
        return
    with MAINTENANCE_LOCK:
        now = time.time()
        if not force and now - LAST_MAINTENANCE_TS < MAINTENANCE_INTERVAL_SEC:
            return
        LAST_MAINTENANCE_TS = now

        for node_id in ACTIVE_NODE_IDS:
            removed_age, removed_cap = _cleanup_output_dir_for_node(node_id)
            if removed_age + removed_cap <= 0:
                continue
            lock = NODE_TASK_LOCKS[node_id]
            with lock:
                NODE_TASK_STATES[node_id]["chatbot_messages"].append(
                    (
                        None,
                        f"🧹 维护清理：删除 {removed_age + removed_cap} 个旧文件（过期 {removed_age}，超额 {removed_cap}）",
                    )
                )
                NODE_TASK_STATES[node_id]["last_update"] = time.time()
                _save_node_task_state(node_id)

def _maybe_auto_reset_error(node_id: str):
    state = NODE_TASK_STATES[node_id]
    if state.get("status") != "error":
        return
    err = state.get("last_error")
    if not _is_transient_error_text(err):
        return
    last_update = float(state.get("last_update") or 0)
    if last_update and (time.time() - last_update) < ERROR_AUTO_RESET_COOLDOWN_SEC:
        return
    thread = NODE_TASK_THREADS.get(node_id)
    if thread is not None and thread.is_alive():
        return
    state["status"] = "idle"
    state["phase"] = "idle"
    state["run_attempt"] = 0
    state["next_retry_at"] = None
    state["last_error"] = None
    state["last_update"] = time.time()
    _save_node_task_state(node_id)


def _reset_task_state(node_id: str):
    state = NODE_TASK_STATES[node_id]
    reset_task_state_fields(
        state,
        auto_proxy_failover_max_switches=AUTO_PROXY_FAILOVER_MAX_SWITCHES,
        auto_replan_max_rounds=AUTO_REPLAN_MAX_ROUNDS,
        auto_recover_max_cycles=AUTO_RECOVER_MAX_CYCLES,
    )
    _reset_runtime_monitor(node_id)


def _supervisor_tick():
    _refresh_node_health_if_due(force=False)
    _run_maintenance_if_due(force=False)
    for target in ACTIVE_NODE_IDS:
        lock = NODE_TASK_LOCKS[target]
        with lock:
            _maybe_auto_reset_error(target)


def _supervisor_loop():
    while not SUPERVISOR_STOP_EVENT.is_set():
        try:
            _supervisor_tick()
        except Exception as exc:
            print(f"[SUPERVISOR] tick error: {exc}")
        SUPERVISOR_STOP_EVENT.wait(SUPERVISOR_INTERVAL_SEC)


def _start_supervisor_once():
    global SUPERVISOR_THREAD
    if SUPERVISOR_THREAD is not None and SUPERVISOR_THREAD.is_alive():
        return
    SUPERVISOR_STOP_EVENT.clear()
    SUPERVISOR_THREAD = threading.Thread(target=_supervisor_loop, name="omnitool-supervisor", daemon=True)
    SUPERVISOR_THREAD.start()

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


def _compact_log_text(text: str, max_len: int = BACKEND_LOG_TEXT_MAX) -> str:
    return RUNTIME_LOG_MONITOR.compact_log_text(text, max_len=max_len)


def _should_print_backend_line(text: str) -> bool:
    return RUNTIME_LOG_MONITOR.should_print_backend_line(text)


def _reset_runtime_monitor(node_id: str):
    RUNTIME_LOG_MONITOR.reset(node_id)


def _track_runtime_signal(node_id: str, text: str):
    RUNTIME_LOG_MONITOR.emit_signal_summary(node_id, text)

def _background_output_callback(node_id: str, message, sender="bot", hide_images=False):
    llm_text = _extract_llm_output(message)
    if llm_text and not llm_text.startswith("<img"):
        if _should_print_backend_line(llm_text):
            print(f"[LOG {node_id}] {_compact_log_text(llm_text)}")
        _track_runtime_signal(node_id, llm_text)

    lock = NODE_TASK_LOCKS[node_id]
    state = NODE_TASK_STATES[node_id]
    with lock:
        phase_text = llm_text.lower() if isinstance(llm_text, str) else ""
        if any(pattern in phase_text for pattern in RECOVERY_PHASE_PATTERNS):
            _set_phase_locked(state, "recovering", announce=True)
        elif state.get("phase") == "recovering" and any(pattern in phase_text for pattern in PHASE_STEP_PATTERNS):
            _set_phase_locked(state, "executing", announce=True)

        chatbot_output_callback(
            message, state["chatbot_messages"], hide_images=True, sender=sender
        )
        state["last_update"] = time.time()
        _save_node_task_state(node_id)


def _plan_update_callback(node_id: str, plan_state: dict):
    lock = NODE_TASK_LOCKS[node_id]
    state = NODE_TASK_STATES[node_id]
    with lock:
        state["plan_step_index"] = plan_state.get("current_index", 0)
        state["last_update"] = time.time()
        _save_node_task_state(node_id)


def _topic_anchor_callback(node_id: str, evidence: dict):
    lock = NODE_TASK_LOCKS[node_id]
    state = NODE_TASK_STATES[node_id]
    with lock:
        if not state.get("topic_anchor_required"):
            return

        input_hit = bool(evidence.get("input_hit", False))
        results_hit = bool(evidence.get("results_hit", False))
        prev_input_hit = bool(state.get("topic_anchor_input_hit", False))
        prev_results_hit = bool(state.get("topic_anchor_results_hit", False))
        prev_done = bool(state.get("topic_anchor_done", False))

        state["topic_anchor_input_hit"] = input_hit
        state["topic_anchor_results_hit"] = results_hit
        done = input_hit and results_hit
        if done and (not prev_done):
            state["topic_anchor_done"] = True
            topic = str(state.get("profile_topic") or evidence.get("topic", "") or "主题")
            state["chatbot_messages"].append(
                (None, f"🎯 主题锚定完成（页面证据）：搜索框包含“{topic}”且结果区命中该主题。")
            )

        if (prev_input_hit != input_hit) or (prev_results_hit != results_hit) or (prev_done != state.get("topic_anchor_done", False)):
            state["last_update"] = time.time()
            _save_node_task_state(node_id)


def _run_task(node_id: str):
    lock = NODE_TASK_LOCKS[node_id]
    task_state = NODE_TASK_STATES[node_id]
    total_steps = 0
    with lock:
        messages = task_state["messages"]
        api_key = task_state["api_key"]
        provider = task_state["provider"]
        model = task_state["model"]
        only_n_images = task_state["only_n_most_recent_images"]
        proxy_base_url = task_state.get("proxy_base_url")
        proxy_model = task_state.get("proxy_model")
        max_steps = task_state.get("max_steps")
        max_seconds = normalize_max_seconds(task_state.get("max_seconds"))
        if CONTINUOUS_IGNORE_MAX_SECONDS and bool(task_state.get("continuous_mode", False)):
            max_seconds = None
        plan_steps = task_state.get("plan_steps") or []
        plan_state = {"current_index": task_state.get("plan_step_index", 0)} if plan_steps else None
        continuous_mode = bool(task_state.get("continuous_mode", False))
        continuous_max_cycles = int(task_state.get("continuous_max_cycles") or 0)
        continuous_interval_sec = float(task_state.get("continuous_interval_sec") or 0.0)
        topic_anchor_required = bool(task_state.get("topic_anchor_required", False))
        topic_anchor_term = str(task_state.get("profile_topic") or "").strip() if topic_anchor_required else ""
        publish_every_cycles = max(1, int(task_state.get("publish_every_cycles") or 1))
        publish_cooldown_sec = max(0.0, float(task_state.get("publish_cooldown_sec") or 0.0))
        max_publish_per_session = max(0, int(task_state.get("max_publish_per_session") or 0))
        profile_id = str(task_state.get("profile_id") or "")
        failover_budget = max(0, _as_int(task_state.get("failover_budget", AUTO_PROXY_FAILOVER_MAX_SWITCHES), AUTO_PROXY_FAILOVER_MAX_SWITCHES))

    plan_total = len(plan_steps) if plan_steps else 0
    print(
        f"[TASK {node_id}] 🚀 开始 | {model} | 计划:{plan_total}步 | max_attempts={RUN_MAX_ATTEMPTS} "
        f"| continuous={continuous_mode} | profile={profile_id or 'none'} | publish_every={publish_every_cycles} "
        f"| cooldown={publish_cooldown_sec:.0f}s | max_publish={max_publish_per_session} | failover={failover_budget}"
    )
    proxy_tried_providers: set[str] = {str(provider or "")}

    while True:
        cycle_chat_start_idx = 0
        cycle_plan_start_idx = 0
        cycle_step_count = 0
        cycle_last_error = ""
        cycle_last_error_retriable = False
        with lock:
            if task_state.get("stop"):
                task_state["status"] = "stopped"
                task_state["next_retry_at"] = None
                _set_phase_locked(task_state, "stopping", announce=True)
                _save_node_task_state(node_id)
                return

            task_state["continuous_cycle"] = int(task_state.get("continuous_cycle") or 0) + 1
            cycle = task_state["continuous_cycle"]
            cycle_chat_start_idx = len(task_state.get("chatbot_messages") or [])
            cycle_plan_start_idx = int(task_state.get("plan_step_index") or 0)
            task_state["run_attempt"] = 0
            task_state["next_retry_at"] = None
            task_state["status"] = "running"
            _set_phase_locked(
                task_state,
                "executing",
                announce=True,
                detail=f"开始第 {cycle} 轮任务执行。",
            )
            task_state["last_update"] = time.time()
            _save_node_task_state(node_id)

        cycle_completed = False
        for attempt in range(1, RUN_MAX_ATTEMPTS + 1):
            with lock:
                if task_state.get("stop"):
                    task_state["status"] = "stopped"
                    task_state["next_retry_at"] = None
                    _set_phase_locked(task_state, "stopping", announce=True)
                    _save_node_task_state(node_id)
                    return

                task_state["run_attempt"] = attempt
                task_state["next_retry_at"] = None
                if attempt > 1:
                    _set_phase_locked(
                        task_state,
                        "recovering",
                        announce=True,
                        detail=f"第 {cycle} 轮，第 {attempt} 次重试准备中（最多 {RUN_MAX_ATTEMPTS} 次）。",
                    )
                task_state["last_update"] = time.time()
                _save_node_task_state(node_id)

            preflight_errors = _preflight_runtime(node_id)
            if preflight_errors:
                err_text = "; ".join(preflight_errors)
                _track_runtime_signal(node_id, err_text)
                print(f"[TASK {node_id}] ⚠️ 预检失败 cycle={cycle} attempt={attempt}: {err_text}")
                if attempt >= RUN_MAX_ATTEMPTS:
                    with lock:
                        task_state["last_error"] = err_text
                        task_state["consecutive_failures"] = int(task_state.get("consecutive_failures") or 0) + 1
                        _save_node_task_state(node_id)
                    cycle_last_error = err_text
                    cycle_last_error_retriable = _is_transient_error_text(err_text)
                    break

                delay = _compute_retry_delay_seconds(attempt)
                with lock:
                    task_state["next_retry_at"] = time.time() + delay
                    task_state["chatbot_messages"].append(
                        (None, f"⚠️ 预检失败，将在 {delay:.1f}s 后重试：{err_text}")
                    )
                    _save_node_task_state(node_id)
                if _sleep_with_stop(node_id, delay):
                    with lock:
                        task_state["status"] = "stopped"
                        task_state["next_retry_at"] = None
                        _set_phase_locked(task_state, "stopping", announce=True)
                        _save_node_task_state(node_id)
                    return
                continue

            try:
                with lock:
                    _set_phase_locked(
                        task_state,
                        "executing",
                        announce=True,
                        detail=f"第 {cycle} 轮，第 {attempt}/{RUN_MAX_ATTEMPTS} 次执行尝试。",
                    )
                    task_state["last_update"] = time.time()
                    _save_node_task_state(node_id)

                step_counter = 0
                for loop_msg in sampling_loop_sync(
                    model=model,
                    provider=provider,
                    messages=messages,
                    output_callback=partial(_background_output_callback, node_id, hide_images=False),
                    tool_output_callback=partial(_tool_output_callback, tool_state=task_state["tools"]),
                    api_response_callback=partial(_api_response_callback, response_state=task_state["responses"]),
                    api_key=api_key,
                    only_n_most_recent_images=only_n_images,
                    max_tokens=16384,
                    omniparser_url=args.omniparser_server_url,
                    windows_host_url=task_state.get("windows_host_url"),
                    capture_output_dir=f"./tmp/outputs/{node_id}",
                    proxy_base_url=proxy_base_url,
                    proxy_model=proxy_model,
                    max_steps=max_steps,
                    max_seconds=max_seconds,
                    plan_steps=plan_steps,
                    plan_state=plan_state,
                    plan_update_callback=partial(_plan_update_callback, node_id) if plan_state else None,
                    topic_anchor_term=topic_anchor_term or None,
                    topic_anchor_callback=partial(_topic_anchor_callback, node_id) if topic_anchor_term else None,
                ):
                    step_counter += 1
                    total_steps += 1
                    if BACKEND_LOG_STEP_HEARTBEAT_EVERY > 0 and (
                        step_counter == 1 or step_counter % BACKEND_LOG_STEP_HEARTBEAT_EVERY == 0
                    ):
                        print(
                            f"[TASK {node_id}] ▶ cycle={cycle} attempt={attempt} step={step_counter} total={total_steps}"
                        )

                    with lock:
                        if task_state.get("stop"):
                            print(f"[TASK {node_id}] ⏹ 停止")
                            task_state["status"] = "stopped"
                            task_state["next_retry_at"] = None
                            _set_phase_locked(task_state, "stopping", announce=True)
                            _save_node_task_state(node_id)
                            return
                    if loop_msg is None:
                        break

                print(
                    f"[TASK {node_id}] ✅ 轮次完成 | cycle={cycle} attempt={attempt} | 本次步数={step_counter} | 总步数={total_steps}"
                )
                cycle_step_count = step_counter
                with lock:
                    task_state["next_retry_at"] = None
                    task_state["consecutive_failures"] = 0
                    task_state["auto_recover_used"] = 0
                    _save_node_task_state(node_id)
                cycle_completed = True
                break

            except Exception as e:
                err_text = str(e)
                retriable = _is_transient_error_text(err_text)
                _track_runtime_signal(node_id, err_text)
                cycle_last_error = err_text
                cycle_last_error_retriable = retriable
                print(
                    f"[TASK {node_id}] ❌ 尝试失败 cycle={cycle} attempt={attempt} retriable={retriable}: {err_text}"
                )
                with lock:
                    task_state["last_error"] = err_text
                    task_state["consecutive_failures"] = int(task_state.get("consecutive_failures") or 0) + 1
                    task_state["last_update"] = time.time()
                    _save_node_task_state(node_id)

                if (
                    retriable
                    and AUTO_PROXY_FAILOVER_ENABLED
                    and model == "omniparser + proxy"
                    and _is_proxy_failover_error(err_text)
                ):
                    switched = False
                    with lock:
                        used_switches = _as_int(task_state.get("failover_switches", 0), 0)
                        budget_switches = max(0, _as_int(task_state.get("failover_budget", AUTO_PROXY_FAILOVER_MAX_SWITCHES), AUTO_PROXY_FAILOVER_MAX_SWITCHES))
                        current_provider = str(task_state.get("provider") or provider or "")
                        if used_switches < budget_switches:
                            next_provider = _next_proxy_provider(current_provider, tried=proxy_tried_providers)
                            if next_provider:
                                new_provider, new_cfg = next_provider
                                new_base = str(new_cfg.get("base_url", "") or "").strip()
                                new_key = str(new_cfg.get("api_key", "") or "").strip()
                                new_model = str(new_cfg.get("default_model", "") or "").strip() or str(proxy_model or "")
                                task_state["provider"] = new_provider
                                task_state["proxy_base_url"] = new_base
                                task_state["api_key"] = new_key
                                task_state["proxy_model"] = new_model
                                task_state["failover_switches"] = used_switches + 1
                                task_state["next_retry_at"] = None
                                task_state["chatbot_messages"].append(
                                    (
                                        None,
                                        f"🔀 通道故障自动切换：{current_provider} -> {new_provider} "
                                        f"({task_state['failover_switches']}/{budget_switches})",
                                    )
                                )
                                _set_phase_locked(task_state, "recovering", announce=True, detail="检测到中转异常，已切换备用通道。")
                                task_state["last_update"] = time.time()
                                _save_node_task_state(node_id)
                                provider = new_provider
                                proxy_base_url = new_base
                                api_key = new_key
                                proxy_model = new_model
                                proxy_tried_providers.add(new_provider)
                                switched = True
                    if switched:
                        delay = 1.0
                        if _sleep_with_stop(node_id, delay):
                            with lock:
                                task_state["status"] = "stopped"
                                task_state["next_retry_at"] = None
                                _set_phase_locked(task_state, "stopping", announce=True)
                                _save_node_task_state(node_id)
                            return
                        continue

                if (not retriable) or attempt >= RUN_MAX_ATTEMPTS:
                    break

                delay = _compute_retry_delay_seconds(attempt)
                with lock:
                    task_state["next_retry_at"] = time.time() + delay
                    _set_phase_locked(
                        task_state,
                        "recovering",
                        announce=True,
                        detail=f"检测到可恢复错误，将在 {delay:.1f}s 后自动重试。",
                    )
                    task_state["chatbot_messages"].append(
                        (None, f"♻️ 可恢复错误：{err_text[:160]}{'...' if len(err_text) > 160 else ''}")
                    )
                    _save_node_task_state(node_id)
                if _sleep_with_stop(node_id, delay):
                    with lock:
                        task_state["status"] = "stopped"
                        task_state["next_retry_at"] = None
                        _set_phase_locked(task_state, "stopping", announce=True)
                        _save_node_task_state(node_id)
                    return

        if not cycle_completed:
            error_text = str(cycle_last_error or "").strip() or str(task_state.get("last_error") or "执行未完成")
            error_retriable = bool(cycle_last_error_retriable or _is_transient_error_text(error_text))
            if _auto_replan(node_id, reason="本轮执行失败，自动修正执行路径", error_text=error_text):
                with lock:
                    plan_steps = task_state.get("plan_steps") or []
                    plan_state = {"current_index": task_state.get("plan_step_index", 0)} if plan_steps else None
                    task_state["status"] = "running"
                    task_state["next_retry_at"] = None
                    task_state["last_update"] = time.time()
                    _save_node_task_state(node_id)
                continue

            recover_delay = 0.0
            should_recover = False
            with lock:
                recover_used = int(task_state.get("auto_recover_used") or 0)
                recover_budget = max(0, _as_int(task_state.get("auto_recover_budget", AUTO_RECOVER_MAX_CYCLES), AUTO_RECOVER_MAX_CYCLES))
                if error_retriable and recover_used < recover_budget:
                    task_state["auto_recover_used"] = recover_used + 1
                    recover_delay = _compute_retry_delay_seconds(RUN_MAX_ATTEMPTS + recover_used + 1)
                    task_state["next_retry_at"] = time.time() + recover_delay
                    _set_phase_locked(
                        task_state,
                        "recovering",
                        announce=True,
                        detail=f"距离目标较远且本轮失败，{recover_delay:.1f}s 后自动重试并修正路径。",
                    )
                    task_state["chatbot_messages"].append(
                        (None, f"🩹 自动修正重试（{task_state['auto_recover_used']}/{recover_budget}）：{error_text[:180]}")
                    )
                    task_state["status"] = "running"
                    task_state["last_update"] = time.time()
                    _save_node_task_state(node_id)
                    should_recover = True
                else:
                    task_state["status"] = "error"
                    task_state["next_retry_at"] = None
                    _set_phase_locked(task_state, "failed", announce=True)
                    _save_node_task_state(node_id)

            if should_recover:
                if _sleep_with_stop(node_id, recover_delay):
                    with lock:
                        task_state["status"] = "stopped"
                        task_state["next_retry_at"] = None
                        _set_phase_locked(task_state, "stopping", announce=True)
                        _save_node_task_state(node_id)
                    return
                continue
            return

        next_cycle_prompt = ""
        should_plan_next_cycle = False
        should_force_replan = False
        force_replan_reason = ""
        with lock:
            if task_state.get("stop"):
                task_state["status"] = "stopped"
                task_state["next_retry_at"] = None
                _set_phase_locked(task_state, "stopping", announce=True)
                _save_node_task_state(node_id)
                return
            cycle = int(task_state.get("continuous_cycle") or 1)

            publish_success_this_cycle = _detect_publish_success_locked(task_state, cycle_chat_start_idx)
            if publish_success_this_cycle:
                task_state["publish_count"] = int(task_state.get("publish_count") or 0) + 1
                task_state["last_publish_ts"] = time.time()
                task_state["chatbot_messages"].append(
                    (None, f"📌 本轮检测到发布成功，当前会话累计发布 {task_state['publish_count']} 条。")
                )
                # 对“发布类任务”采用硬终止：命中成功证据后直接完成，不再走偏航重规划。
                if (not continuous_mode) and _is_publish_task_locked(task_state):
                    plan_steps_done = task_state.get("plan_steps") or []
                    if isinstance(plan_steps_done, list) and plan_steps_done:
                        task_state["plan_step_index"] = len(plan_steps_done)
                    task_state["stagnation_rounds"] = 0
                    task_state["status"] = "completed"
                    task_state["next_retry_at"] = None
                    _set_phase_locked(task_state, "completed", announce=True, detail="检测到发布成功，任务结束。")
                    task_state["chatbot_messages"].append(
                        (None, "✅ 发布任务已达成（命中发布成功证据），停止自动重规划。")
                    )
                    task_state["last_update"] = time.time()
                    _save_node_task_state(node_id)
                    print(f"[TASK {node_id}] 🎯 发布任务命中成功证据，已结束执行。")
                    return

            anchor_required = bool(task_state.get("topic_anchor_required", False))
            anchor_done = bool(task_state.get("topic_anchor_done", False))
            topic_text = str(task_state.get("profile_topic") or "").strip()
            if anchor_required and (not anchor_done) and topic_text:
                input_hit = bool(task_state.get("topic_anchor_input_hit", False))
                results_hit = bool(task_state.get("topic_anchor_results_hit", False))
                if (not (input_hit and results_hit)) and cycle_step_count >= 2:
                    should_force_replan = True
                    force_replan_reason = f"本轮未完成主题锚定（{topic_text}），回到搜索锚定流程"
                    task_state["chatbot_messages"].append(
                        (
                            None,
                            f"⚠️ 本轮未完成主题锚定（{topic_text}）。"
                            f"页面证据：input_hit={int(input_hit)}, results_hit={int(results_hit)}。"
                            "触发自动重规划回到搜索步骤。",
                        )
                    )

            current_plan_idx = int(task_state.get("plan_step_index") or 0)
            has_progress = current_plan_idx > cycle_plan_start_idx
            if _plan_is_incomplete_locked(task_state) and (not has_progress) and cycle_step_count >= AUTO_REPLAN_MIN_STEP_COUNT:
                task_state["stagnation_rounds"] = int(task_state.get("stagnation_rounds") or 0) + 1
                task_state["chatbot_messages"].append(
                    (
                        None,
                        f"🧭 偏航监测：本轮计划进度未推进（{current_plan_idx}/{len(task_state.get('plan_steps') or [])}），"
                        f"连续停滞 {task_state['stagnation_rounds']} 轮。",
                    )
                )
            elif has_progress:
                task_state["stagnation_rounds"] = 0

            if (not should_force_replan) and (not continuous_mode):
                if _plan_is_incomplete_locked(task_state):
                    should_force_replan = True
                    force_replan_reason = "任务尚未达成且计划未完成，自动重规划继续执行"
                    task_state["chatbot_messages"].append(
                        (None, "♻️ 任务尚未达成，触发自动重规划继续执行（不直接结束）。")
                    )
                    task_state["last_update"] = time.time()
                    _save_node_task_state(node_id)
                else:
                    task_state["status"] = "completed"
                    _set_phase_locked(task_state, "completed", announce=True)
                    _save_node_task_state(node_id)
                    return
            elif (not should_force_replan) and int(task_state.get("stagnation_rounds") or 0) >= AUTO_REPLAN_STAGNATION_ROUNDS:
                should_force_replan = True
                force_replan_reason = (
                    f"连续 {int(task_state.get('stagnation_rounds') or 0)} 轮计划无推进，自动收敛路径"
                )
                task_state["chatbot_messages"].append(
                    (None, f"♻️ 连续停滞达到阈值（{AUTO_REPLAN_STAGNATION_ROUNDS}），触发自动重规划。")
                )
                task_state["last_update"] = time.time()
                _save_node_task_state(node_id)
            elif not should_force_replan:
                if continuous_max_cycles > 0 and cycle >= continuous_max_cycles:
                    task_state["status"] = "completed"
                    _set_phase_locked(task_state, "completed", announce=True, detail="达到持续模式轮次上限。")
                    _save_node_task_state(node_id)
                    return

                next_cycle = cycle + 1
                strategy = _decide_next_cycle_strategy_locked(task_state, next_cycle=next_cycle, now_ts=time.time())
                policy_label = "允许发布（最多 1 条）" if strategy.get("allow_publish") else "仅互动（禁止发布）"
                blocked_reasons = strategy.get("blocked_reasons") or []
                blocked_suffix = ""
                if blocked_reasons:
                    blocked_suffix = " | 限制：" + "；".join(str(item) for item in blocked_reasons if str(item).strip())
                focus = str(strategy.get("interaction_task") or "").strip()
                focus_suffix = f" | 重点：{focus}" if focus else ""
                task_state["chatbot_messages"].append(
                    (None, f"🧭 本轮策略：{policy_label}{blocked_suffix}{focus_suffix}")
                )

                next_cycle_prompt = _build_next_cycle_prompt_locked(task_state, next_cycle=next_cycle, strategy=strategy)
                pause_text = f"{continuous_interval_sec:.0f}s" if continuous_interval_sec > 0 else "0s"
                task_state["chatbot_messages"].append(
                    (None, f"🔁 第 {cycle} 轮完成，{pause_text} 后进入下一轮养号。")
                )
                messages.append(
                    {
                        "role": Sender.USER,
                        "content": [TextBlock(type="text", text=next_cycle_prompt)],
                    }
                )
                _set_phase_locked(task_state, "planning", announce=True, detail=f"第 {next_cycle} 轮准备中。")
                task_state["status"] = "running"
                task_state["last_update"] = time.time()
                _save_node_task_state(node_id)
                should_plan_next_cycle = True

        if should_force_replan:
            if _auto_replan(node_id, reason=force_replan_reason):
                with lock:
                    plan_steps = task_state.get("plan_steps") or []
                    plan_state = {"current_index": task_state.get("plan_step_index", 0)} if plan_steps else None
                    task_state["status"] = "running"
                    task_state["next_retry_at"] = None
                    task_state["last_update"] = time.time()
                    _save_node_task_state(node_id)
                continue
            with lock:
                task_state["status"] = "error"
                task_state["next_retry_at"] = None
                _set_phase_locked(task_state, "failed", announce=True, detail="自动重规划预算耗尽或失败。")
                _save_node_task_state(node_id)
            return

        if should_plan_next_cycle:
            next_plan_text = _generate_plan(next_cycle_prompt, task_state)
            next_plan_steps = _parse_plan_steps(next_plan_text)
            with lock:
                if task_state.get("stop"):
                    task_state["status"] = "stopped"
                    task_state["next_retry_at"] = None
                    _set_phase_locked(task_state, "stopping", announce=True)
                    _save_node_task_state(node_id)
                    return

                if next_plan_steps:
                    task_state["plan"] = next_plan_text
                    task_state["plan_steps"] = next_plan_steps
                    task_state["plan_step_index"] = 0
                    plan_steps = next_plan_steps
                    plan_state = {"current_index": 0}
                    task_state["chatbot_messages"].append((None, f"🗺️ 新一轮规划：共 {len(next_plan_steps)} 步"))
                else:
                    task_state["plan"] = None
                    task_state["plan_steps"] = None
                    task_state["plan_step_index"] = 0
                    plan_steps = []
                    plan_state = None
                    task_state["chatbot_messages"].append((None, "🗺️ 新一轮规划降级：按策略直接执行。"))
                task_state["last_update"] = time.time()
                _save_node_task_state(node_id)

        if continuous_interval_sec > 0 and _sleep_with_stop(node_id, continuous_interval_sec):
            with lock:
                task_state["status"] = "stopped"
                task_state["next_retry_at"] = None
                _set_phase_locked(task_state, "stopping", announce=True)
                _save_node_task_state(node_id)
            return


def valid_params(user_input, state, node_id: str):
    """Validate all requirements and return a list of error messages."""
    errors = []
    host_url = NODE_MAP.get(node_id, "")

    try:
        url = f'http://{args.omniparser_server_url}/probe'
        response = requests.get(url, timeout=3, proxies={"http": "", "https": ""})
        if response.status_code != 200:
            errors.append(f"OmniParser Server is not responding")
    except RequestException:
        errors.append(f"OmniParser Server is not responding. Please start it first: cd omnitool/omniparserserver && python -m omniparserserver")

    if not args.local and host_url:
        try:
            probe_resp = requests.get(
                f"http://{host_url}/probe",
                timeout=3,
                proxies={"http": "", "https": ""},
            )
            if probe_resp.status_code != 200:
                errors.append(f"Windows Host {host_url} is not responding")
        except RequestException:
            errors.append(f"Windows Host {host_url} is not responding")

    if state.get("model") != "omniparser-only" and not state["api_key"].strip():
        errors.append("LLM API Key is not set")

    if not user_input:
        errors.append("no computer use request provided")
    
    return errors


def _normalize_topic_text(topic: str) -> str:
    raw = str(topic or "").strip()
    if not raw:
        return ""
    cleaned = re.sub(r"\s+", " ", raw).strip("，。,.；;:：")
    return cleaned


def _compose_task_input(
    user_input: str,
    task_type: str,
    topic_source: str,
    preset_topic: str,
    manual_topic: str,
) -> tuple[str, str]:
    raw = (user_input or "").strip()
    topic = ""
    topic_from = "无"
    if topic_source == "手动输入":
        topic = _normalize_topic_text(manual_topic)
        topic_from = "手动输入" if topic else "无"
    elif topic_source == "预设主题":
        topic = _normalize_topic_text(preset_topic)
        topic_from = "预设主题" if topic else "无"

    detected_platform = _detect_platform_target(raw) if raw else None
    platform_label = str(PLATFORM_TARGETS.get(detected_platform, {}).get("label", "未识别")) if detected_platform else "未识别"

    # 手动输入任务优先，其次按预制模板自动生成任务。
    if raw:
        if task_type == "自动识别" and detected_platform in {"douyin", "kuaishou"}:
            platform_url = str(PLATFORM_TARGETS.get(detected_platform, {}).get("url", "") or "")
            if platform_url and platform_url.lower() not in raw.lower():
                raw = (
                    raw
                    + f"\n硬约束：目标平台={platform_label}。"
                      f"第一步必须使用 Ctrl+L 打开 {platform_url} 并回车。"
                      "禁止进入小红书相关页面（xiaohongshu.com / creator.xiaohongshu.com）。"
                )
        if topic and topic.lower() not in raw.lower():
            raw = raw + f" | 方向: {topic} | topic:{topic}"
        return raw, f"任务类型={task_type} | 平台={platform_label} | 主题={topic or '未指定'} | 来源={topic_from}"

    topic_segment = f" topic:{topic}" if topic else ""
    if task_type == "小红书养号":
        topic_text = topic or "通用"
        composed = (
            f"小红书养号 {topic_text} 相关{topic_segment}。"
            f"先搜索“{topic_text}”并进入相关内容后再互动。"
        )
        return composed, f"任务类型=小红书养号 | 平台=小红书 | 主题={topic_text} | 来源={topic_from}"
    if task_type == "小红书发布":
        topic_text = topic or "项目进度"
        composed = (
            f"小红书发布纯文本笔记 {topic_text} 相关{topic_segment}。"
            "标题与正文非空后再发布。"
        )
        return composed, f"任务类型=小红书发布 | 平台=小红书 | 主题={topic_text} | 来源={topic_from}"
    if task_type == "浏览互动":
        topic_text = topic or "通用"
        composed = (
            f"使用浏览器执行浏览互动任务，主题为 {topic_text}{topic_segment}。"
            "先搜索主题，再执行自然浏览、点赞、收藏、评论。"
        )
        return composed, f"任务类型=浏览互动 | 平台={platform_label} | 主题={topic_text} | 来源={topic_from}"
    if task_type in {"抖音发布内容", "抖音发布学习"}:
        topic_text = topic or "通用发布内容"
        composed = (
            f"打开抖音执行通用发布内容任务（{topic_text}）{topic_segment}。"
            "第一步必须进入 https://www.douyin.com 并定位“投稿/创作者中心/发布”入口；"
            "进入编辑页后完成发布前校验：标题与正文非空，话题/封面按需设置；"
            "校验通过后执行发布，并确认“发布成功”提示或发布后状态变化。"
        )
        return composed, f"任务类型=抖音发布内容 | 平台=抖音 | 主题={topic_text} | 来源={topic_from}"
    if task_type == "总结":
        topic_text = topic or "当前任务"
        composed = f"总结 {topic_text} 相关的今日进展与风险{topic_segment}。"
        return composed, f"任务类型=总结 | 平台={platform_label} | 主题={topic_text} | 来源={topic_from}"
    return raw, f"任务类型=自动识别 | 平台={platform_label} | 主题={topic or '未指定'} | 来源={topic_from}"


def _start_background_task_single(user_input, state, node_id: str) -> tuple[bool, str]:
    errors = valid_params(user_input, state, node_id)
    if errors:
        return False, "; ".join(errors)
    lock = NODE_TASK_LOCKS[node_id]
    task_state = NODE_TASK_STATES[node_id]
    with lock:
        current_status = task_state.get("status", "idle")
        if current_status == "running":
            return False, "任务正在运行"

        print(f"[START] 新任务 node={node_id}")

        _reset_task_state(node_id)

        task_state["model"] = state.get("model")
        task_state["provider"] = state.get("provider")
        task_state["api_key"] = state.get("api_key")
        task_state["task_type"] = str(state.get("task_type") or "自动识别")
        task_state["proxy_base_url"] = state.get("proxy_base_url")
        task_state["proxy_model"] = state.get("proxy_model")
        task_state["only_n_most_recent_images"] = state.get("only_n_most_recent_images", 2)
        task_state["max_steps"] = state.get("max_steps")
        normalized_max_seconds = normalize_max_seconds(state.get("max_seconds"))
        task_state["max_seconds"] = 0 if normalized_max_seconds is None else normalized_max_seconds
        task_state["auto_replan_budget"] = AUTO_REPLAN_MAX_ROUNDS
        task_state["auto_replan_used"] = 0
        task_state["stagnation_rounds"] = 0
        task_state["auto_recover_budget"] = AUTO_RECOVER_MAX_CYCLES
        task_state["auto_recover_used"] = 0
        task_state["failover_budget"] = AUTO_PROXY_FAILOVER_MAX_SWITCHES
        task_state["failover_switches"] = 0

        effective_user_input, profile_status, profile_meta = _expand_task_from_profiles(user_input)
        task_state["goal_task"] = effective_user_input
        if profile_status:
            task_state["chatbot_messages"].append((None, profile_status))
        if effective_user_input != user_input:
            task_state["chatbot_messages"].append(
                (None, "🧾 任务扩展后: " + _compact_line(effective_user_input, max_len=260))
            )
        task_state["profile_id"] = str(profile_meta.get("profile_id", "") or "")
        task_state["profile_topic"] = str(profile_meta.get("topic", "通用") or "通用")
        task_state["continuous_mode"] = bool(profile_meta.get("continuous_mode", False))
        task_state["continuous_cycle"] = 0
        task_state["continuous_max_cycles"] = max(0, _as_int(profile_meta.get("continuous_max_cycles", 1), 1))
        task_state["continuous_interval_sec"] = max(0.0, _as_float(profile_meta.get("continuous_interval_sec", 0.0), 0.0))
        task_state["publish_every_cycles"] = max(1, _as_int(profile_meta.get("publish_every_cycles", 4), 4))
        task_state["publish_cooldown_sec"] = max(0.0, _as_float(profile_meta.get("publish_cooldown_sec", 1800.0), 1800.0))
        task_state["max_publish_per_session"] = max(0, _as_int(profile_meta.get("max_publish_per_session", 2), 2))
        task_state["publish_count"] = 0
        task_state["last_publish_ts"] = None
        raw_interaction_tasks = profile_meta.get("interaction_tasks", [])
        task_state["interaction_tasks"] = [
            str(item).strip() for item in raw_interaction_tasks if str(item).strip()
        ] if isinstance(raw_interaction_tasks, list) else []
        task_state["interaction_task_index"] = 0
        task_state["topic_anchor_required"] = bool(profile_meta.get("topic_anchor_required", False))
        task_state["topic_anchor_done"] = False
        continuous_prompt_template = str(profile_meta.get("continuous_prompt_template", "") or "").strip()
        if continuous_prompt_template:
            topic = str(profile_meta.get("topic", "通用"))
            task_state["continuous_prompt"] = continuous_prompt_template.format(topic=topic).strip()
        else:
            task_state["continuous_prompt"] = ""
        if task_state["topic_anchor_required"] and task_state["profile_topic"] and task_state["profile_topic"] != "通用":
            task_state["chatbot_messages"].append(
                (None, f"🎯 主题锚定要求：本轮起先搜索“{task_state['profile_topic']}”并进入相关结果页，再进行互动。")
            )
        if task_state["continuous_mode"]:
            max_cycles = task_state["continuous_max_cycles"]
            cycle_text = "无限" if max_cycles <= 0 else str(max_cycles)
            task_state["chatbot_messages"].append(
                (
                    None,
                    f"♻️ 持续养号模式已开启：最大轮次={cycle_text}，轮次间隔={task_state['continuous_interval_sec']:.0f}s，"
                    f"发布频率=每 {task_state['publish_every_cycles']} 轮，发布冷却={task_state['publish_cooldown_sec']:.0f}s，"
                    f"会话发布上限={task_state['max_publish_per_session']}。",
                )
            )
            if CONTINUOUS_IGNORE_MAX_SECONDS and normalize_max_seconds(task_state.get("max_seconds")) is not None:
                task_state["chatbot_messages"].append(
                    (
                        None,
                        f"⏱️ 持续模式已忽略单轮 max_seconds={task_state.get('max_seconds')}，仅受手动停止/策略轮次限制。",
                    )
                )
            else:
                task_state["chatbot_messages"].append(
                    (
                        None,
                        f"⏱️ 单轮最大运行时间：{max_seconds_label(task_state.get('max_seconds'))}。",
                    )
                )
            first_cycle_strategy = _decide_next_cycle_strategy_locked(task_state, next_cycle=1, now_ts=time.time())
            first_policy_label = "允许发布（最多 1 条）" if first_cycle_strategy.get("allow_publish") else "仅互动（禁止发布）"
            first_blocked = first_cycle_strategy.get("blocked_reasons") or []
            first_blocked_suffix = ""
            if first_blocked:
                first_blocked_suffix = " | 限制：" + "；".join(str(item) for item in first_blocked if str(item).strip())
            first_focus = str(first_cycle_strategy.get("interaction_task") or "").strip()
            first_focus_suffix = f" | 重点：{first_focus}" if first_focus else ""
            task_state["chatbot_messages"].append(
                (None, f"🧭 首轮策略：{first_policy_label}{first_blocked_suffix}{first_focus_suffix}")
            )
            first_cycle_prompt = _build_next_cycle_prompt_locked(task_state, next_cycle=1, strategy=first_cycle_strategy)
        else:
            task_state["chatbot_messages"].append(
                (
                    None,
                    f"⏱️ 单轮最大运行时间：{max_seconds_label(task_state.get('max_seconds'))}。",
                )
            )
            first_cycle_prompt = ""

        task_state["messages"].append(
            {
                "role": Sender.USER,
                "content": [TextBlock(type="text", text=effective_user_input)],
            }
        )
        _set_phase_locked(
            task_state,
            "preprocessing",
            announce=True,
            detail="正在执行索引命中、流程文档预读与环境检查。",
        )
        flow_msg, flow_docs, flow_status = _build_flow_reference_message(effective_user_input)
        task_state["chatbot_messages"].append((None, flow_status))
        if flow_msg:
            task_state["messages"].append(
                {
                    "role": Sender.USER,
                    "content": [TextBlock(type="text", text=flow_msg)],
                }
            )
        if first_cycle_prompt:
            task_state["messages"].append(
                {
                    "role": Sender.USER,
                    "content": [TextBlock(type="text", text=first_cycle_prompt)],
                }
            )

        _set_phase_locked(
            task_state,
            "planning",
            announce=True,
            detail="正在生成可执行计划（带成功条件）。",
        )
        planning_seed_text = effective_user_input
        if first_cycle_prompt:
            planning_seed_text = effective_user_input + "\n" + first_cycle_prompt
        plan_text = _generate_plan(planning_seed_text, task_state)
        plan_steps = _parse_plan_steps(plan_text)
        if plan_steps:
            task_state["plan"] = plan_text
            task_state["plan_steps"] = plan_steps
            task_state["plan_step_index"] = 0
            preview = "\n".join(
                f"{idx+1}. {step.get('action', '')} | Success: {step.get('success', '')}"
                for idx, step in enumerate(plan_steps[:3])
            )
            task_state["chatbot_messages"].append(
                (
                    None,
                    f"🗺️ 规划完成：共 {len(plan_steps)} 步\n{preview}",
                )
            )
        else:
            task_state["chatbot_messages"].append(
                (
                    None,
                    "🗺️ 规划降级：未生成结构化计划，将按流程文档与通用策略执行。",
                )
            )

        _set_phase_locked(task_state, "executing", announce=True, detail="准备进入执行循环。")
        task_state["chatbot_messages"].append((user_input, None))

        task_state["status"] = "starting"
        task_state["last_update"] = time.time()
        _save_node_task_state(node_id)

        NODE_TASK_THREADS[node_id] = threading.Thread(target=_run_task, args=(node_id,), daemon=True)
        NODE_TASK_THREADS[node_id].start()

        return True, ""


def start_background_task(
    user_input,
    task_type,
    topic_source,
    preset_topic,
    manual_topic,
    proxy_provider_value,
    proxy_model_value,
    proxy_base_url_value,
    proxy_api_key_value,
    state,
    node_id,
):
    # 以界面当前值为准，避免仅靠 change 回调导致的状态不同步。
    state["task_type"] = str(task_type or "自动识别")
    state["model"] = "omniparser + proxy"
    state["proxy_provider"] = str(proxy_provider_value or "").strip()
    state["provider"] = state["proxy_provider"] or state.get("provider")
    state["proxy_model"] = str(proxy_model_value or "").strip()
    state["proxy_base_url"] = str(proxy_base_url_value or "").strip()
    state["api_key"] = str(proxy_api_key_value or "").strip()
    if state.get("provider"):
        state[f"{state['provider']}_api_key"] = state["api_key"]

    effective_input, compose_summary = _compose_task_input(
        user_input=user_input,
        task_type=task_type,
        topic_source=topic_source,
        preset_topic=preset_topic,
        manual_topic=manual_topic,
    )
    compose_summary = compose_summary + f" | 单轮时长={max_seconds_label(state.get('max_seconds'))}"
    target_ids = _target_node_ids(node_id)
    started_nodes: list[str] = []
    failed_nodes: list[str] = []
    failed_reasons: list[str] = []
    for target in target_ids:
        ok, reason = _start_background_task_single(effective_input, state, target)
        if ok:
            started_nodes.append(target)
            lock = NODE_TASK_LOCKS[target]
            with lock:
                NODE_TASK_STATES[target]["chatbot_messages"].append((None, f"🧩 启动参数：{compose_summary}"))
                _save_node_task_state(target)
        else:
            failed_nodes.append(target)
            failed_reasons.append(f"{target}: {reason}")

    if not started_nodes:
        raise gr.Error("启动失败: " + " | ".join(failed_reasons))

    primary = _primary_node_id(node_id)
    if node_id == ALL_NODES_ID:
        lock = NODE_TASK_LOCKS[primary]
        with lock:
            NODE_TASK_STATES[primary]["chatbot_messages"].append(
                (None, f"📡 广播已启动: {len(started_nodes)} 节点")
            )
            if failed_nodes:
                NODE_TASK_STATES[primary]["chatbot_messages"].append(
                    (None, "⚠️ 部分节点未启动: " + ", ".join(failed_nodes))
                )
            _save_node_task_state(primary)

    return *_chat_outputs(node_id), *_status_outputs(node_id), ""


def resume_task(node_id):
    target_ids = _target_node_ids(node_id)
    resumed = 0
    for target in target_ids:
        lock = NODE_TASK_LOCKS[target]
        task_state = NODE_TASK_STATES[target]
        with lock:
            if task_state.get("status") == "running":
                continue
            if not task_state.get("messages"):
                continue
            task_state["stop"] = False
            task_state["last_error"] = None
            task_state["status"] = "starting"
            task_state["next_retry_at"] = None
            _set_phase_locked(task_state, "executing", announce=True, detail="手动恢复任务执行。")
            task_state["last_update"] = time.time()
            _save_node_task_state(target)
            NODE_TASK_THREADS[target] = threading.Thread(target=_run_task, args=(target,), daemon=True)
            NODE_TASK_THREADS[target].start()
            resumed += 1
    if resumed == 0:
        raise gr.Error("没有可恢复的任务")
    return *_chat_outputs(node_id), *_status_outputs(node_id)


def refresh_task(node_id):
    _supervisor_tick()
    return *_chat_outputs(node_id), *_status_outputs(node_id)


def stop_task(node_id):
    for target in _target_node_ids(node_id):
        lock = NODE_TASK_LOCKS[target]
        task_state = NODE_TASK_STATES[target]
        with lock:
            task_state["stop"] = True
            if task_state.get("status") == "running":
                task_state["status"] = "stopping"
            _set_phase_locked(task_state, "stopping", announce=True)
            _save_node_task_state(target)
    return _status_outputs(node_id)


def clear_node_chat(node_id):
    for target in _target_node_ids(node_id):
        lock = NODE_TASK_LOCKS[target]
        with lock:
            _reset_task_state(target)
            _save_node_task_state(target)
    return *_chat_outputs(node_id), *_status_outputs(node_id)

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

        function activateMonitorTabFromUrl() {
            const params = new URLSearchParams(window.location.search);
            const view = (params.get('view') || '').toLowerCase();
            if (view !== 'monitor') return;

            let tries = 0;
            const maxTries = 30;
            const timer = setInterval(() => {
                tries += 1;
                const tabButtons = Array.from(document.querySelectorAll('button[role="tab"]'));
                const target = tabButtons.find(btn => (btn.textContent || '').trim() === '同时监控');
                if (target) {
                    target.click();
                    clearInterval(timer);
                    return;
                }
                if (tries >= maxTries) {
                    clearInterval(timer);
                }
            }, 250);
        }

        document.addEventListener('DOMContentLoaded', () => {
            activateMonitorTabFromUrl();
        });
        </script>
    """)
    state = gr.State({})
    
    setup_state(state.value)
    
    llm_config = load_config()
    default_provider_key = llm_config.get("default_provider", "codex_proxy")
    default_provider = llm_config.get("providers", {}).get(default_provider_key, {})

    with gr.Accordion("中转 API 设置", open=False):
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
                proxy_api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    value=default_provider.get("api_key", ""),
                    placeholder="输入 API Key",
                    interactive=True,
                )

    node_choices = []
    for idx, node_id in enumerate(ACTIVE_NODE_IDS, start=1):
        host = NODE_TASK_STATES[node_id].get("windows_host_url") or "local"
        node_choices.append((node_label(host, idx), node_id))
    if len(ACTIVE_NODE_IDS) > 1:
        node_choices.insert(0, ("All Nodes (Broadcast)", ALL_NODES_ID))

    topic_catalog = _load_topic_catalog(TOPIC_CATALOG_PATH)
    topic_choices = [str(item.get("label", "")).strip() for item in topic_catalog if str(item.get("label", "")).strip()]
    default_topic = topic_choices[0] if topic_choices else "openclaw"
    flow_doc_choices = _list_manageable_flow_docs()
    default_flow_doc = flow_doc_choices[0] if flow_doc_choices else ""

    with gr.Row():
        with gr.Column(scale=2):
            task_type = gr.Dropdown(
                label="任务类型",
                choices=["自动识别", "小红书养号", "小红书发布", "抖音发布内容", "浏览互动", "总结"],
                value="自动识别",
                interactive=True,
            )
        with gr.Column(scale=2):
            topic_source = gr.Radio(
                label="主题来源",
                choices=["不指定", "预设主题", "手动输入"],
                value="不指定",
                interactive=True,
            )
        with gr.Column(scale=2):
            preset_topic = gr.Dropdown(
                label="预设主题",
                choices=topic_choices,
                value=default_topic,
                interactive=True,
                visible=False,
            )
        with gr.Column(scale=2):
            manual_topic = gr.Textbox(
                label="手动主题",
                placeholder="例如: openclaw agent",
                interactive=True,
                visible=False,
            )
        with gr.Column(scale=2):
            active_node = gr.Dropdown(
                label="当前控制节点",
                choices=node_choices,
                value=DEFAULT_NODE_ID,
                interactive=True,
            )

    with gr.Row():
        with gr.Column(scale=8):
            chat_input = gr.Textbox(
                show_label=False,
                placeholder="输入任务（可留空，系统按任务类型+主题自动生成）",
                container=False,
            )
        with gr.Column(scale=1, min_width=60):
            submit_button = gr.Button(value="Send", variant="primary")
        with gr.Column(scale=1, min_width=60):
            stop_button = gr.Button(value="Stop", variant="secondary")
        with gr.Column(scale=1, min_width=60):
            resume_button = gr.Button(value="Resume", variant="secondary")

    status_bar = gr.Markdown(_task_status_text(DEFAULT_NODE_ID))
    gr.Markdown("⏱️ 运行时长说明：`max_seconds=0` 表示不限时，当前默认不限时。")
    gr.HTML(
        '<a href="?view=monitor" target="_blank" '
        'style="display:inline-block;margin:4px 0 10px 0;text-decoration:none;color:#2563eb;">'
        '🖥️ 新标签页打开“同时监控”</a>'
    )
    monitor_status_components = []
    monitor_view_components = []
    monitor_grid_view_components = []
    node_chat_components = []

    with gr.Tabs():
        with gr.Tab("控制台"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    chatbot = gr.Chatbot(
                        label="当前节点会话",
                        autoscroll=True,
                        height=520,
                        type="tuples",
                        allow_tags=False,
                    )
                    with gr.Accordion("各节点会话（可折叠）", open=False):
                        for idx, node_id in enumerate(ACTIVE_NODE_IDS, start=1):
                            host = NODE_TASK_STATES[node_id].get("windows_host_url") or "local"
                            with gr.Accordion(node_label(host, idx), open=False):
                                node_chat = gr.Chatbot(
                                    label=f"{host}",
                                    autoscroll=True,
                                    height=220,
                                    type="tuples",
                                    allow_tags=False,
                                )
                                node_chat_components.append(node_chat)

                with gr.Column(scale=5):
                    gr.Markdown("### 单节点大画面")
                    with gr.Tabs():
                        for idx, node_id in enumerate(ACTIVE_NODE_IDS, start=1):
                            host = NODE_TASK_STATES[node_id].get("windows_host_url") or "local"
                            with gr.Tab(node_label(host, idx)):
                                monitor_status = gr.Markdown(_task_status_text(node_id))
                                monitor_view = gr.HTML(value="", container=False, elem_classes="no-padding")
                                monitor_status_components.append(monitor_status)
                                monitor_view_components.append(monitor_view)

        with gr.Tab("同时监控"):
            gr.Markdown("### 全节点实时监控")
            with gr.Row():
                for idx, node_id in enumerate(ACTIVE_NODE_IDS, start=1):
                    host = NODE_TASK_STATES[node_id].get("windows_host_url") or "local"
                    with gr.Column():
                        gr.Markdown(f"**{node_label(host, idx).split(' | ')[0]}** `{host}`")
                        monitor_grid_view = gr.HTML(value="", container=False, elem_classes="no-padding")
                        monitor_grid_view_components.append(monitor_grid_view)

        with gr.Tab("流程内容管理"):
            gr.Markdown("### 前台管理流程索引与指导文档")
            with gr.Row():
                flow_doc_selector = gr.Dropdown(
                    label="文档选择",
                    choices=flow_doc_choices,
                    value=default_flow_doc or None,
                    interactive=True,
                )
                flow_doc_refresh_btn = gr.Button("刷新文档列表", variant="secondary")
            flow_doc_editor = gr.Textbox(
                label="文档内容",
                lines=20,
                max_lines=28,
                placeholder="选择文档后点击“加载文档内容”",
                interactive=True,
            )
            with gr.Row():
                flow_doc_load_btn = gr.Button("加载文档内容", variant="secondary")
                flow_doc_save_btn = gr.Button("保存文档内容", variant="primary")
            flow_doc_status = gr.Markdown("ℹ️ 可在前台维护 docs/index 与 docs/flows 下的流程文档。")

    def update_proxy_provider(proxy_provider_value, state):
        """切换中转 provider 时更新配置"""
        provider_config = get_provider_config(proxy_provider_value)
        if provider_config:
            # 中转模式固定：模型侧统一走 proxy，不在 UI 暴露 Model 切换。
            state["model"] = "omniparser + proxy"
            state["proxy_provider"] = proxy_provider_value
            state["proxy_base_url"] = provider_config.get("base_url", "")
            state["api_key"] = provider_config.get("api_key", "")
            state["proxy_model"] = provider_config.get("default_model", "gpt-4o")
            state["provider"] = proxy_provider_value
            return (
                gr.update(
                    choices=provider_config.get("available_models", []),
                    value=state["proxy_model"],
                ),
                gr.update(value=state["proxy_base_url"]),
                gr.update(value=state["api_key"]),
            )
        return (gr.update(), gr.update(), gr.update())

    def update_proxy_base_url(base_url_value, state):
        state["proxy_base_url"] = str(base_url_value or "").strip()

    def update_proxy_api_key(api_key_value, state):
        state["api_key"] = str(api_key_value or "").strip()
        provider_key = str(state.get("proxy_provider") or state.get("provider") or "").strip()
        if provider_key:
            state[f"{provider_key}_api_key"] = state["api_key"]

    def update_proxy_model(proxy_model_value, state):
        state["proxy_model"] = str(proxy_model_value or "").strip()

    def switch_node(node_id):
        return *_chat_outputs(node_id), *_status_outputs(node_id)

    def update_topic_source(source):
        if source == "预设主题":
            return gr.update(visible=True), gr.update(visible=False)
        if source == "手动输入":
            return gr.update(visible=False), gr.update(visible=True)
        return gr.update(visible=False), gr.update(visible=False)

    def _monitor_cache_get(node_id: str) -> dict:
        with MONITOR_CACHE_LOCK:
            if node_id not in MONITOR_CACHE:
                MONITOR_CACHE[node_id] = {
                    "frame_hash": "",
                    "html_large": "",
                    "html_grid": "",
                }
            return MONITOR_CACHE[node_id]

    def _render_monitor_html(host: str, img_b64: str, height: int = 580) -> str:
        return (
            f'<div style="height:{height}px;background:#111;border-radius:8px;overflow:hidden;position:relative;">'
            f'<div style="position:absolute;top:8px;left:10px;z-index:3;color:#fff;font-size:12px;'
            f'background:rgba(0,0,0,.55);padding:4px 8px;border-radius:6px;">'
            f'{host}</div>'
            f'<img src="data:image/png;base64,{img_b64}" '
            f'style="width:100%;height:100%;object-fit:contain;background:#000;" /></div>'
        )

    def _render_monitor_placeholder(text: str, color: str = "#86efac", height: int = 580) -> str:
        return (
            f'<div style="height:{height}px;display:flex;align-items:center;justify-content:center;'
            f'background:#111;color:{color};border-radius:8px;">{text}</div>'
        )

    def refresh_vm_monitor_for_node(node_id: str):
        cache = _monitor_cache_get(node_id)
        host = NODE_TASK_STATES[node_id].get("windows_host_url") or ""
        if args.local or not host:
            local_large = _render_monitor_placeholder("Local monitor mode", color="#86efac", height=580)
            local_grid = _render_monitor_placeholder("Local monitor mode", color="#86efac", height=240)
            if cache["html_large"] != local_large or cache["html_grid"] != local_grid:
                cache["html_large"] = local_large
                cache["html_grid"] = local_grid
                cache["frame_hash"] = "local"
                return local_large, local_grid
            return gr.update(), gr.update()
        url = f"http://{host}/screenshot?t={int(time.time()*1000)}"
        try:
            response = requests.get(
                url,
                timeout=MONITOR_REQUEST_TIMEOUT_SEC,
                proxies={"http": "", "https": ""},
            )
            if response.status_code != 200:
                if cache["html_large"] and cache["html_grid"]:
                    return gr.update(), gr.update()
                err_large = _render_monitor_placeholder(f"{host} HTTP {response.status_code}", color="#fca5a5", height=580)
                err_grid = _render_monitor_placeholder(f"{host} HTTP {response.status_code}", color="#fca5a5", height=240)
                cache["html_large"] = err_large
                cache["html_grid"] = err_grid
                return err_large, err_grid
            frame_hash = hashlib.blake2s(response.content).hexdigest()
            if frame_hash == cache["frame_hash"] and cache["html_large"] and cache["html_grid"]:
                return gr.update(), gr.update()
            img_b64 = base64.b64encode(response.content).decode("utf-8")
            html_large = _render_monitor_html(host, img_b64, height=580)
            html_grid = _render_monitor_html(host, img_b64, height=240)
            cache["frame_hash"] = frame_hash
            cache["html_large"] = html_large
            cache["html_grid"] = html_grid
            return html_large, html_grid
        except Exception as e:
            if cache["html_large"] and cache["html_grid"]:
                return gr.update(), gr.update()
            err_large = _render_monitor_placeholder(f"{host} error: {str(e)}", color="#fca5a5", height=580)
            err_grid = _render_monitor_placeholder(f"{host} error: {str(e)}", color="#fca5a5", height=240)
            cache["html_large"] = err_large
            cache["html_grid"] = err_grid
            return err_large, err_grid

    def refresh_all_vm_monitors():
        large_updates = []
        grid_updates = []
        for node_id in ACTIVE_NODE_IDS:
            large_html, grid_html = refresh_vm_monitor_for_node(node_id)
            large_updates.append(large_html)
            grid_updates.append(grid_html)
        return [*large_updates, *grid_updates]

    proxy_provider.change(
        fn=update_proxy_provider,
        inputs=[proxy_provider, state],
        outputs=[proxy_model, proxy_base_url, proxy_api_key],
    )
    proxy_model.change(fn=update_proxy_model, inputs=[proxy_model, state], outputs=None)
    proxy_base_url.change(fn=update_proxy_base_url, inputs=[proxy_base_url, state], outputs=None)
    proxy_api_key.change(fn=update_proxy_api_key, inputs=[proxy_api_key, state], outputs=None)
    status_outputs = [status_bar, *monitor_status_components]
    chat_outputs = [chatbot, *node_chat_components]
    chat_and_status_outputs = [*chat_outputs, *status_outputs]
    task_start_outputs = [*chat_outputs, *status_outputs, chat_input]

    active_node.change(
        fn=switch_node,
        inputs=[active_node],
        outputs=chat_and_status_outputs,
    )
    topic_source.change(
        fn=update_topic_source,
        inputs=[topic_source],
        outputs=[preset_topic, manual_topic],
    )
    flow_doc_refresh_btn.click(
        fn=_refresh_flow_doc_editor,
        inputs=[flow_doc_selector],
        outputs=[flow_doc_selector, flow_doc_status],
    )
    flow_doc_load_btn.click(
        fn=_load_flow_doc_editor,
        inputs=[flow_doc_selector],
        outputs=[flow_doc_editor, flow_doc_status],
    )
    flow_doc_selector.change(
        fn=_load_flow_doc_editor,
        inputs=[flow_doc_selector],
        outputs=[flow_doc_editor, flow_doc_status],
    )
    flow_doc_save_btn.click(
        fn=_save_flow_doc_editor,
        inputs=[flow_doc_selector, flow_doc_editor],
        outputs=[flow_doc_status],
    )
    chatbot.clear(
        fn=clear_node_chat,
        inputs=[active_node],
        outputs=chat_and_status_outputs,
    )

    submit_button.click(
        start_background_task,
        [
            chat_input,
            task_type,
            topic_source,
            preset_topic,
            manual_topic,
            proxy_provider,
            proxy_model,
            proxy_base_url,
            proxy_api_key,
            state,
            active_node,
        ],
        task_start_outputs,
    )
    stop_button.click(
        stop_task,
        [active_node],
        status_outputs,
    )
    resume_button.click(
        resume_task,
        [active_node],
        chat_and_status_outputs,
    )

    auto_refresh_timer = gr.Timer(value=1.5)
    auto_refresh_timer.tick(
        refresh_task,
        [active_node],
        chat_and_status_outputs,
    )
    vm_refresh_timer = gr.Timer(value=MONITOR_REFRESH_INTERVAL_SEC)
    vm_refresh_timer.tick(
        refresh_all_vm_monitors,
        [],
        [*monitor_view_components, *monitor_grid_view_components],
    )
    
if __name__ == "__main__":
    _refresh_node_health_if_due(force=True)
    _run_maintenance_if_due(force=True)
    _start_supervisor_once()
    demo.launch(server_name="0.0.0.0", server_port=7888)



