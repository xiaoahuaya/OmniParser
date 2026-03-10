from __future__ import annotations

import re


DEFAULT_REMOTE_NODE_HOSTS = (
    "192.168.31.134:5000,192.168.31.135:5000,192.168.31.136:5000"
)
DEFAULT_LOCAL_NODE_HOST = "localhost:5000"


def normalize_host(host: str) -> str:
    return str(host or "").strip().replace("http://", "").replace("https://", "")


def node_id_from_host(host: str) -> str:
    normalized = normalize_host(host).lower().replace(".", "_").replace(":", "_")
    return re.sub(r"[^a-z0-9_]+", "_", normalized).strip("_")


def is_local_host(host: str) -> bool:
    normalized = normalize_host(host).lower()
    return normalized.startswith("localhost:") or normalized.startswith("127.0.0.1:")


def resolve_node_hosts(
    *,
    raw_hosts: str,
    fallback_host: str = "",
    include_local_node: bool = True,
    local_node_host: str = DEFAULT_LOCAL_NODE_HOST,
    local_mode: bool = False,
) -> list[str]:
    if local_mode:
        return []

    parts = [normalize_host(item) for item in str(raw_hosts or "").split(",") if normalize_host(item)]
    if not parts and fallback_host:
        parts = [normalize_host(fallback_host)]

    if include_local_node:
        local_host = normalize_host(local_node_host)
        if local_host:
            parts.append(local_host)

    deduped: list[str] = []
    seen: set[str] = set()
    for host in parts:
        if not host or host in seen:
            continue
        seen.add(host)
        deduped.append(host)
    return deduped


def node_label(host: str, index: int) -> str:
    normalized = normalize_host(host) or "local"
    prefix = "Local" if is_local_host(normalized) else f"Node{index}"
    return f"{prefix} | {normalized}"
