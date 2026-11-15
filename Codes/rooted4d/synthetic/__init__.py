"""Known-truth geometry-domain benchmark."""

from .benchmark import SyntheticCase, generate_case, save_case
from .config import SemisyntheticConfig, config_from_mapping, load_config
from .dataset_builder import build_semisynthetic_dataset

__all__ = [
    "SemisyntheticConfig",
    "SyntheticCase",
    "build_semisynthetic_dataset",
    "config_from_mapping",
    "generate_case",
    "load_config",
    "save_case",
]
