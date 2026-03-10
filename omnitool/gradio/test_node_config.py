from __future__ import annotations

import sys
from pathlib import Path


GRADIO_DIR = Path(__file__).resolve().parent
if str(GRADIO_DIR) not in sys.path:
    sys.path.insert(0, str(GRADIO_DIR))

import node_config


def test_resolve_node_hosts_auto_includes_local_node():
    hosts = node_config.resolve_node_hosts(
        raw_hosts="192.168.31.134:5000,192.168.31.135:5000,192.168.31.136:5000",
        include_local_node=True,
        local_node_host="localhost:5000",
    )
    assert hosts == [
        "192.168.31.134:5000",
        "192.168.31.135:5000",
        "192.168.31.136:5000",
        "localhost:5000",
    ]


def test_resolve_node_hosts_dedupes_existing_local_node():
    hosts = node_config.resolve_node_hosts(
        raw_hosts="192.168.31.134:5000,localhost:5000",
        include_local_node=True,
        local_node_host="localhost:5000",
    )
    assert hosts == [
        "192.168.31.134:5000",
        "localhost:5000",
    ]


def test_node_label_marks_local_host():
    assert node_config.node_label("localhost:5000", 4) == "Local | localhost:5000"
    assert node_config.node_label("192.168.31.134:5000", 1) == "Node1 | 192.168.31.134:5000"
