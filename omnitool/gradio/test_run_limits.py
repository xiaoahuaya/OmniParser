from __future__ import annotations

import sys
from pathlib import Path


GRADIO_DIR = Path(__file__).resolve().parent
if str(GRADIO_DIR) not in sys.path:
    sys.path.insert(0, str(GRADIO_DIR))

from run_limits import max_seconds_label, normalize_max_seconds


def test_normalize_max_seconds_zero_or_negative_means_unlimited():
    assert normalize_max_seconds(0) is None
    assert normalize_max_seconds(-1) is None
    assert normalize_max_seconds(None) is None


def test_normalize_max_seconds_positive_keeps_value():
    assert normalize_max_seconds(900) == 900


def test_max_seconds_label_uses_unlimited_text():
    assert max_seconds_label(0) == "不限时"
    assert max_seconds_label(None) == "不限时"
    assert max_seconds_label(120) == "120s"
