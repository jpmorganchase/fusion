"""Top-level package for fusion."""

__author__ = """Fusion Devs"""
__email__ = "fusion_developers@jpmorgan.com"
__version__ = "1.4.0"

from fusion._fusion import FusionCredentials
from fusion.fs_sync import fsync
from fusion.fusion import Fusion

from ._fusion import *  # noqa: F403

__all__ = ["Fusion", "FusionCredentials", "fsync", "rust_ok"]  # noqa: F405
