from enum import Enum


class ApplicationType(Enum):
    """
    Enum representing different types of applications.

    Attributes:
        SEAL (str): Represents a seal application.
        INTELLIGENT_SOLUTION (str): Represents an intelligent solution application.
        USER_TOOL (str): Represents a user tool application.

    Args:
        Enum (class): Inherits from the Enum class to create enumerations.
    """

    SEAL = "Application (SEAL)"
    INTELLIGENT_SOLUTION = "Intelligent Solution"
    USER_TOOL = "User Tool"
