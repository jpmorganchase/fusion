"""Top-level package for fusion."""

__author__ = """Fusion Devs"""
__email__ = "fusion_developers@jpmorgan.com"
__version__ = "1.1.0-dev1"

from _fusion import *  # noqa: F403

from fusion.authentication import FusionCredentials
from fusion.fs_sync import fsync
from fusion.fusion import Fusion

__all__ = ["Fusion", "FusionCredentials", "fsync", "rust_ok"]  # noqa: F405
