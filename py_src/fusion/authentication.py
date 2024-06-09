"""Fusion authentication module."""

import logging
from typing import Any, Optional, Union
from urllib.parse import urlparse

import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from fusion._fusion import FusionCredentials

logger = logging.getLogger(__name__)
VERBOSE_LVL = 25


def _res_plural(ref_int: int, pluraliser: str = "s") -> str:
    """Private function to return the plural form when the number is more than one.

    Args:
        ref_int (int): The reference integer that determines whether to return a plural suffix.
        pluraliser (str, optional): The plural suffix. Defaults to "s".

    Returns:
        str: The plural suffix to append to a string.
    """
    return "" if abs(ref_int) == 1 else pluraliser


def _is_url(url: str) -> bool:
    """Test whether the content of a string is a valid URL.

    Args:
        url (str): The content to evaluate.

    Returns:
        bool: True if the content of data is a URL, False otherwise.
    """
    try:
        parsed = urlparse(url)
        return all([parsed.scheme, parsed.netloc])
    except (ValueError, AttributeError):
        return False


class FusionOAuthAdapter(HTTPAdapter):
    """An OAuth adapter to manage authentication and session tokens."""

    def __init__(
        self,
        credentials: FusionCredentials,
        proxies: Optional[dict[str, str]] = None,
        refresh_within_seconds: int = 5,
        auth_retries: Optional[Union[int, Retry]] = None,
        mount_url: str = "",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Class constructor to create a FusionOAuthAdapter object.

        Args:
            credentials (FusionCredentials): Valid user credentials to authenticate.
            proxies (dict, optional): Specify a proxy if required to access the authentication server.
                Defaults to {}.
            refresh_within_seconds (int, optional): When an API call is made with less than the specified
                number of seconds until the access token expires, or after expiry, it will refresh the token.
                Defaults to 5.
            auth_retries (Union[int, Retry]): Number of times to attempt to authenticate to handle connection problems.
                Defaults to None.
            mount_url (str, optional): Mount url.
        """
        if proxies is None:
            proxies = {}
        super().__init__(*args, **kwargs)

        self.credentials = credentials

        if proxies:
            self.proxies = proxies
        else:
            self.proxies = self.credentials.proxies
        self.token = None
        self.number_token_refreshes = 0
        self.refresh_within_seconds = refresh_within_seconds
        self.mount_url = mount_url

        if not auth_retries:
            self.auth_retries = Retry(total=20, backoff_factor=0.2)
        else:
            self.auth_retries = Retry.from_int(auth_retries)

    def send(  # noqa: PLR0915
        self,
        request: requests.PreparedRequest,
        *args: Any,  # noqa: ARG002
        **kwargs: Any,
    ) -> requests.Response:
        """Function to send a request to the authentication server.

        Args:
            request (requests.PreparedRequest): A HTTP Session.

        Returns:
            requests.Response: The response from the server.
        """
        # We'll always have a url but requests makes this field optional
        if request.url:
            request.headers.update(self.credentials.get_fusion_token_headers(request.url))

        response = super().send(request, **kwargs)
        return response


class FusionAiohttpSession(aiohttp.ClientSession):
    """Bespoke aiohttp session."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Class constructor to create a FusionOAuthAdapter object.

        Args:
            args (optional): List of argument from aiohttp session.
            kwargs (dict, optional): Named list or arguments for aiohttp session.
                Defaults to {}.
            refresh_within_seconds (int, optional): When an API call is made with less than the specified
                number of seconds until the access token expires, or after expiry, it will refresh the token.
                Defaults to 5.
        """
        self.token = None
        self.refresh_within_seconds: Optional[int] = None
        self.number_token_refreshes: Optional[int] = None
        self.credentials: Optional[FusionCredentials] = None
        super().__init__(*args, **kwargs)

    def post_init(self, credentials: Optional[FusionCredentials] = None, refresh_within_seconds: int = 5) -> None:
        """Set member variables."""
        self.token = None
        self.refresh_within_seconds = refresh_within_seconds
        self.number_token_refreshes = 0
        self.credentials = credentials
        self.fusion_token_dict: dict[str, str] = {}
        self.fusion_token_expiry_dict: dict[str, Any] = {}
