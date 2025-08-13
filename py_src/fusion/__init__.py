"""Top-level package for fusion."""

__author__ = """Fusion Devs"""
__email__ = "fusion_developers@jpmorgan.com"
__version__ = "2.0.15"

from .credentials import FusionCredentials
from .fs_sync import fsync
from .fusion import Fusion

__all__ = ["Fusion", "FusionCredentials", "fsync"]
