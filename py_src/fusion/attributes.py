"""Fusion Product class and functions."""

from __future__ import annotations

import json as js
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any

import pandas as pd

from fusion.utils import _is_json, convert_date_format, make_bool, make_list, tidy_string

if TYPE_CHECKING:
    import requests

    from fusion import Fusion


@dataclass
class Attribute:
    """Attribute class."""

    title: str
    identifier: str
    index: int
    dataType: str
    description: str = ""
    isDatatsetKey: bool | None = None
    source: str | None = None
    sourceFieldId: str | None = None
    isInternalDatasetKey: bool | None = None
    isExternallyVisible: bool | None = None
    unit: Any | None = None
    multiplier: float = 1.0
    isPropogationEligible: bool | None = None
    isMetric: bool | None = None
    availableFrom: str | None = None
    deprecatedFrom: str | None = None
    term: str = "bizterm1"
    dataset: int | None = None
    attributeType: str | None = None

    def __str__(self: Attribute) -> str:
        """Format object representation."""
        attrs = {k: v for k, v in self.__dict__.items()}
        return f"Attribute(\n" + ",\n ".join(f"{k}={v!r}" for k, v in attrs.items()) + "\n)"
    
    def __repr__(self: Attribute) -> str:
        """Format object representation."""
        s = ", ".join(f"{getattr(self, f.name)!r}" for f in fields(self))
        return "(" + s + ")"

