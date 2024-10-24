"""Fusion types definitions."""

from enum import Enum


class Types(Enum):
    """Fusion types.

    Args:
        Enum (class: `enum.Enum`): Enum inheritance.
    """

    String = 1
    Boolean = 2
    Decimal = 3
    Float = 4
    Double = 5
    Timestamp = 6
    Date = 8
    Binary = 9
    Long = 11
    Integer = 12
    Short = 13
    Byte = 14
    Datetime = 6  # noqa: PIE796
