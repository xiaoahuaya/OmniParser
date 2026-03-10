from __future__ import annotations

import sys
from pathlib import Path


GRADIO_DIR = Path(__file__).resolve().parent
if str(GRADIO_DIR) not in sys.path:
    sys.path.insert(0, str(GRADIO_DIR))

from runtime_log_monitor import RuntimeLogMonitor


def test_should_suppress_tool_noise_by_default():
    monitor = RuntimeLogMonitor(
        debug_logs=False,
        backend_log_mode="compact",
        text_max=220,
        summary_interval_sec=25.0,
    )

    assert monitor.should_print_backend_line("[mouse_move] [745, 342]") is False
    assert monitor.should_print_backend_line("[result] Performed left_click") is False
    assert monitor.should_print_backend_line("[key] ctrl+a") is False


def test_should_keep_important_recovery_messages():
    monitor = RuntimeLogMonitor(
        debug_logs=False,
        backend_log_mode="compact",
        text_max=220,
        summary_interval_sec=25.0,
    )

    assert monitor.should_print_backend_line("⚠️ 动作确认门: 刚才动作后未检测到有效变化") is True
