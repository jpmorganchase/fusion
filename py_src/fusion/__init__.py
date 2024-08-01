"""Top-level package for fusion."""

__author__ = """Fusion Devs"""
__email__ = "fusion_developers@jpmorgan.com"
<<<<<<< HEAD
__version__ = "1.2.2"
=======
__version__ = "1.2.1"
>>>>>>> origin/master

from fusion._fusion import FusionCredentials
from fusion.fs_sync import fsync
from fusion.fusion import Fusion

from ._fusion import *  # noqa: F403

__all__ = ["Fusion", "FusionCredentials", "fsync", "rust_ok"]  # noqa: F405
