"""Top-level package for fusion."""

__author__ = """Fusion Devs"""
__email__ = "fusion_developers@jpmorgan.com"
__version__ = "3.0.0"

from fusion.credentials import FusionCredentials
from fusion.fs_sync import fsync
from fusion.fusion import Fusion

__all__ = ["Fusion", "FusionCredentials", "fsync"]  # noqa: F405
