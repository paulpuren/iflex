"""Merge parser defaults, YAML configuration, and explicit CLI options."""

import argparse
from pathlib import Path

import yaml


def parse_args_with_yaml(parser, argv=None, *, ignore_unknown=False, aliases=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="")
    config_args, remaining = pre_parser.parse_known_args(argv)
    config = {}
    if config_args.config:
        with Path(config_args.config).open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise ValueError("YAML config must be a key-value mapping.")
        for source, target in (aliases or {}).items():
            if source in config and target not in config:
                config[target] = config[source]
        valid_keys = {action.dest for action in parser._actions}
        unknown = set(config) - valid_keys
        if unknown and not ignore_unknown:
            raise KeyError(f"Unknown keys in config {config_args.config}: {sorted(unknown, key=str)}")
        if ignore_unknown:
            config = {key: value for key, value in config.items() if key in valid_keys}
    # argparse fills only missing namespace attributes with defaults. Explicit CLI
    # options therefore override YAML even when their value equals a default.
    args = parser.parse_args(remaining, namespace=argparse.Namespace(**config))
    args.config = config_args.config
    return args
