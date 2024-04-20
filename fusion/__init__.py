"""Top-level package for fusion."""

__author__ = """Fusion Devs"""
__email__ = "fusion_developers@jpmorgan.com"
__version__ = "1.0.23"

from _fusion import *  # noqa: F403

from fusion.fs_sync import fsync
from fusion.fusion import Fusion, FusionCredentials

__all__ = ["Fusion", "FusionCredentials", "fsync", "rust_ok"]  # noqa: F405
