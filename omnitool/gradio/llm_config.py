"""
LLM 中转 API 配置管理模块
支持 Codex 和 Claude 中转 API 的配置读取和保存
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

CONFIG_FILE_NAME = "llm_config.json"

def get_config_path() -> Path:
    """获取配置文件路径"""
    current_dir = Path(__file__).parent
    return current_dir / CONFIG_FILE_NAME

def load_config() -> Dict[str, Any]:
    """加载配置文件"""
    config_path = get_config_path()

    if not config_path.exists():
        return get_default_config()

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"加载配置文件失败: {e}，使用默认配置")
        return get_default_config()

def save_config(config: Dict[str, Any]) -> bool:
    """保存配置文件"""
    config_path = get_config_path()

    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except IOError as e:
        print(f"保存配置文件失败: {e}")
        return False

def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        "providers": {
            "codex_proxy": {
                "name": "Codex 中转",
                "base_url": "https://right.codes/codex/v1",
                "api_key": "",
                "default_model": "gpt-5.2-codex",
                "available_models": [
                    "gpt-5.4",
                    "gpt-5.3",
                    "gpt-5.2-codex",
                    "gpt-5.2",
                    "gpt-5.1-codex-max",
                    "gpt-5.1-codex",
                    "gpt-5.1",
                    "gpt-5-codex",
                ]
            },
            "claude_proxy": {
                "name": "Claude 中转",
                "base_url": "https://api.anthropic.com",
                "api_key": "",
                "default_model": "claude-3-5-sonnet-20241022",
                "available_models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
            }
        },
        "default_provider": "codex_proxy"
    }

def get_provider_config(provider_key: str) -> Optional[Dict[str, Any]]:
    """获取指定 provider 的配置"""
    config = load_config()
    return config.get("providers", {}).get(provider_key)

def get_all_providers() -> Dict[str, Dict[str, Any]]:
    """获取所有 provider 配置"""
    config = load_config()
    return config.get("providers", {})

def get_provider_choices() -> list:
    """获取 provider 选择列表，用于 Gradio Dropdown"""
    providers = get_all_providers()
    return [(v["name"], k) for k, v in providers.items()]

def update_provider_config(provider_key: str, **kwargs) -> bool:
    """更新指定 provider 的配置"""
    config = load_config()

    if provider_key not in config.get("providers", {}):
        return False

    for key, value in kwargs.items():
        if value is not None:
            config["providers"][provider_key][key] = value

    return save_config(config)

def get_default_provider() -> str:
    """获取默认 provider"""
    config = load_config()
    return config.get("default_provider", "codex_proxy")

def set_default_provider(provider_key: str) -> bool:
    """设置默认 provider"""
    config = load_config()
    config["default_provider"] = provider_key
    return save_config(config)
