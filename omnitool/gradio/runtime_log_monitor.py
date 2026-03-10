"""Runtime log monitoring helpers.

Keeps backend log-noise filtering and signal aggregation out of app.py.
"""

from __future__ import annotations

import threading
import time


DEFAULT_NOISY_PREFIXES = (
    "📋 动作:",
    "analysis:",
    "next action:",
    "[mouse_move]",
    "[left_click]",
    "[double_click]",
    "[right_click]",
    "[scroll_down]",
    "[scroll_up]",
    "[key]",
    "[type]",
    "[wait]",
    "[result]",
    "box id:",
    "from box id:",
    "to box id:",
    "box_centroid_coordinate:",
    "moved mouse to",
    "performed left_click",
    "performed double_click",
    "performed right_click",
    "performed scroll_down",
    "performed scroll_up",
    "pressed keys:",
    "⌨️ 已输入文本",
)

DEFAULT_IMPORTANT_MARKERS = (
    "❌",
    "⚠️",
    "♻️",
    "🔀",
    "✅",
    "⏹",
    "阶段：",
    "错误",
    "失败",
    "exception",
    "timeout",
    "403",
    "404",
    "429",
    "500",
    "502",
    "503",
)

DEFAULT_SIGNAL_PATTERNS = {
    "action_gate": ("动作确认门",),
    "repeat_click": ("次点击相同位置", "repeated clicking on nearly the same spot"),
    "focus_probe": ("focus probe failed", "输入动作未生效"),
    "recoverable_error": ("可恢复错误",),
    "proxy_403": ("403 client error", "403", "余额不足"),
    "proxy_404": ("404 client error", "404"),
    "proxy_502": ("502 server error", "bad gateway"),
    "failover": ("通道故障自动切换",),
}


class RuntimeLogMonitor:
    def __init__(
        self,
        *,
        debug_logs: bool,
        backend_log_mode: str,
        text_max: int,
        summary_interval_sec: float,
        noisy_prefixes: tuple[str, ...] = DEFAULT_NOISY_PREFIXES,
        important_markers: tuple[str, ...] = DEFAULT_IMPORTANT_MARKERS,
        signal_patterns: dict[str, tuple[str, ...]] = DEFAULT_SIGNAL_PATTERNS,
    ) -> None:
        self.debug_logs = bool(debug_logs)
        self.backend_log_mode = str(backend_log_mode or "compact").strip().lower()
        self.text_max = max(80, int(text_max))
        self.summary_interval_sec = max(8.0, float(summary_interval_sec))
        self.noisy_prefixes = noisy_prefixes
        self.important_markers = important_markers
        self.signal_patterns = signal_patterns
        self._lock = threading.Lock()
        self._watchers: dict[str, dict] = {}

    def compact_log_text(self, text: str, max_len: int | None = None) -> str:
        line = " ".join(str(text or "").split())
        target_max = self.text_max if max_len is None else max(1, int(max_len))
        if len(line) <= target_max:
            return line
        return line[:target_max] + "..."

    def should_print_backend_line(self, text: str) -> bool:
        if self.debug_logs or self.backend_log_mode == "verbose":
            return True
        if not text:
            return False
        lowered = str(text).strip().lower()
        if not lowered:
            return False
        if any(marker.lower() in lowered for marker in self.important_markers):
            return True
        return not any(lowered.startswith(prefix) for prefix in self.noisy_prefixes)

    def reset(self, node_id: str) -> None:
        with self._lock:
            self._watchers[node_id] = {
                "last_summary_ts": time.time(),
                "counts": {},
                "samples": {},
            }

    def track_signal(self, node_id: str, text: str) -> str | None:
        if not text:
            return None
        lowered = str(text).lower()
        matched: list[str] = []
        for signal, patterns in self.signal_patterns.items():
            if any(pattern in lowered for pattern in patterns):
                matched.append(signal)
        if not matched:
            return None

        now_ts = time.time()
        with self._lock:
            watcher = self._watchers.setdefault(
                node_id,
                {"last_summary_ts": now_ts, "counts": {}, "samples": {}},
            )
            counts = watcher.setdefault("counts", {})
            samples = watcher.setdefault("samples", {})
            for signal in matched:
                counts[signal] = int(counts.get(signal, 0)) + 1
                samples[signal] = self.compact_log_text(text, max_len=140)

            if now_ts - float(watcher.get("last_summary_ts", 0.0)) < self.summary_interval_sec:
                return None

            if not counts:
                watcher["last_summary_ts"] = now_ts
                return None

            ordered = sorted(counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))
            summary = ", ".join(f"{k}={v}" for k, v in ordered[:5])
            sample_key = ordered[0][0]
            sample_text = str(samples.get(sample_key, "") or "")
            watcher["counts"] = {}
            watcher["samples"] = {}
            watcher["last_summary_ts"] = now_ts
            return f"[MONITOR {node_id}] {summary}" + (f" | sample={sample_text}" if sample_text else "")

    def emit_signal_summary(self, node_id: str, text: str) -> None:
        line = self.track_signal(node_id, text)
        if line:
            print(line)
