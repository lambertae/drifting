"""Miscellaneous utilities."""
from __future__ import annotations
import os
import yaml


class EasyDict(dict):
    """Dictionary subclass that allows attribute-style access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def _dict_to_easydict(d):
    if not isinstance(d, dict):
        return d
    out = EasyDict()
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _dict_to_easydict(v)
        elif isinstance(v, list):
            out[k] = [_dict_to_easydict(i) for i in v]
        else:
            out[k] = v
    return out


def load_config(config_path: str) -> EasyDict:
    """Load a YAML config file and return an EasyDict."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return _dict_to_easydict(yaml.safe_load(f))
