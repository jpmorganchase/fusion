"""Top-level package for fusion."""

# ruff: noqa: F405, F403
__author__ = """Fusion Devs"""
__email__ = "fusion_developers@jpmorgan.com"
__version__ = "1.0.21"

from _fusion import *

from fusion.fs_sync import fsync
from fusion.fusion import Fusion, FusionCredentials

__all__ = ["Fusion", "FusionCredentials", "fsync", "rust_ok"]
