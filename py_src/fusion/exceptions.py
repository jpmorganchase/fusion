"""Bespoke exceptions and errors."""

from typing import Optional


class APIResponseError(Exception):
    """APIResponseError exception wrapper to handle API response errors.

    Args:
        Exception : Exception to wrap.
    """

    def __init__(self, original_exception: Exception, message: str = "", status_code: Optional[int] = None) -> None:
        self.original_exception = original_exception
        self.status_code = status_code
        if message:
            full_message = f"APIResponseError: Status {status_code}, {message}, Error: {str(original_exception)}"
        else:
            full_message = f"APIResponseError: Status {status_code}, Error: {str(original_exception)}"
        super().__init__(full_message)

        # Optionally, copy original exception attributes
        self.__dict__.update(getattr(original_exception, "__dict__", {}))


class APIRequestError(Exception):
    """APIRequestError exception wrapper to handle API request erorrs.

    Args:
        Exception : Exception to wrap.
    """


class APIConnectError(Exception):
    """APIConnectError exception wrapper to handle API connection errors.

    Args:
        Exception : Exception to wrap.
    """


class UnrecognizedFormatError(Exception):
    """UnrecognizedFormatError exception wrapper to handle format errors.

    Args:
        Exception : Exception to wrap.
    """


class CredentialError(Exception):
    """CredentialError exception wrapper to handle errors in credentials provided for authentication.

    Args:
        Exception : Exception to wrap.
    """

    def __init__(self, original_exception: Exception, message: str = "", status_code: Optional[int] = None) -> None:
        self.original_exception = original_exception
        self.status_code = status_code
        full_message = f"APIResponseError: Status {status_code}, Error: {str(original_exception)}"
        if message:
            full_message = f"{message} | {full_message}"
        super().__init__(full_message)

        # Optionally, copy original exception attributes
        self.__dict__.update(getattr(original_exception, "__dict__", {}))


class FileFormatError(Exception):
    """FileFormatRequiredError exception wrapper to handle errors in download when file format is not accepted or
        cannot be determined.

    Args:
        Exception : Exception to wrap.
    """
