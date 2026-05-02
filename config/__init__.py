"""Configuration module"""

import yaml
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_dir = Path(__file__).parent
        config_path = config_dir / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path) -> None:
    """Lưu config dict trở lại file YAML"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


__all__ = ['load_config', 'save_config']
