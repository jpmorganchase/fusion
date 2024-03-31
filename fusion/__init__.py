"""Top-level package for fusion."""

# ruff: noqa: F405, F403
__author__ = """Fusion Devs"""
__email__ = "fusion_developers@jpmorgan.com"
__version__ = "1.0.21"

from fusion.fs_sync import fsync
from fusion.fusion import Fusion, FusionCredentials
from _fusion import *

__all__ = ["Fusion", "FusionCredentials", "fsync", "rust_ok"]
