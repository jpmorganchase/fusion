"""Bespoke exceptions and errors."""


class APIResponseError(Exception):
    """APIResponseError exception wrapper to handle API response errors.

    Args:
        Exception : Exception to wrap.
    """

    pass


class APIRequestError(Exception):
    """APIRequestError exception wrapper to handle API request erorrs.

    Args:
        Exception : Exception to wrap.
    """

    pass


class APIConnectError(Exception):
    """APIConnectError exception wrapper to handle API connection errors.

    Args:
        Exception : Exception to wrap.
    """

    pass


class UnrecognizedFormatError(Exception):
    """UnrecognizedFormatError exception wrapper to handle format errors.

    Args:
        Exception : Exception to wrap.
    """

    pass


class CredentialError(Exception):
    """CredentialError exception wrapper to handle errors in credentials provided for authentication.

    Args:
        Exception : Exception to wrap.
    """

    pass
