"""Fusion authentication module."""

import logging
from typing import Any, Optional, Union
from urllib.parse import urlparse

import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from fusion._fusion import FusionCredentials

from . import __version__ as version
from .exceptions import CredentialError

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

    def _refresh_token_data(self) -> tuple[str, str]:
        payload = (
            {
                "grant_type": "client_credentials",
                "client_id": self.credentials.client_id,
                "client_secret": self.credentials.client_secret,
                "aud": self.credentials.resource,
            }
            if self.credentials.grant_type == "client_credentials"
            else {
                "grant_type": "password",
                "client_id": self.credentials.client_id,
                "username": self.credentials.username,
                "password": self.credentials.password,
                "resource": self.credentials.resource,
            }
        )

        try:
            s = requests.Session()
            if self.proxies:
                # mypy does not recognise session.proxies as a dict so fails this line, we'll ignore this chk
                s.proxies.update(self.proxies)
            s.mount("http://", HTTPAdapter(max_retries=self.auth_retries))
            if not self.credentials.auth_url:
                raise CredentialError("A valid auth_url is required")
            response = s.post(self.credentials.auth_url, data=payload)
            response.raise_for_status()
            response_data = response.json()
            access_token = response_data["access_token"]
            expiry = response_data["expires_in"]
            return access_token, expiry
        except CredentialError as ex:
            raise CredentialError(f"Failed to authenticate against OAuth server") from ex

    def _refresh_fusion_token_data(self, request: requests.PreparedRequest, **kwargs: Any) -> tuple[str, str]:
        if not request.url:
            raise CredentialError("A valid request URL is required")  # pragma: no cover
        full_url_lst = request.url.split("/")
        url = "/".join(full_url_lst[: full_url_lst.index("datasets") + 2]) + "/authorize/token"
        session = requests.Session()
        response = session.get(url, headers=request.headers, **kwargs)
        response_data = response.json()
        access_token = response_data["access_token"]
        expiry = response_data["expires_in"]
        return access_token, expiry

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

        url_lst = request.path_url.split("/")
        fusion_auth_req = "distributions" in url_lst

        if not self.credentials.bearer_token:
            # First time through, get a token
            token, expiry = self._refresh_token_data()
            self.credentials.put_bearer_token(token, int(expiry))
            self.number_token_refreshes += 1
            logger.log(
                VERBOSE_LVL,
                f"Got init bearer token {self.number_token_refreshes} time{_res_plural(self.number_token_refreshes)}",
            )

        if not self.credentials.bearer_token:
            raise CredentialError("Failed to authenticate against OAuth server")

        exp = self.credentials.bearer_token.expires_in_secs()
        if exp and exp < self.refresh_within_seconds:
            # Expired or about to expire, get a new token
            token, expiry = self._refresh_token_data()
            self.credentials.put_bearer_token(token, int(expiry))
            self.number_token_refreshes += 1
            logger.log(
                VERBOSE_LVL,
                f"Refreshed token {self.number_token_refreshes} time{_res_plural(self.number_token_refreshes)}",
            )
        request.headers.update(self.credentials.get_bearer_token_header())
        request.headers.update({"User-Agent": f"fusion-python-sdk {version}"})

        if fusion_auth_req:
            catalog = url_lst[url_lst.index("catalogs") + 1]
            dataset = url_lst[url_lst.index("datasets") + 1]
            fusion_token_key = catalog + "_" + dataset

            if fusion_token_key not in self.credentials.fusion_token:
                fusion_token, fusion_token_expiry = self._refresh_fusion_token_data(request, **kwargs)
                self.credentials.put_fusion_token(fusion_token_key, fusion_token, int(fusion_token_expiry))
                logger.log(VERBOSE_LVL, "Refreshed fusion token")
            else:
                fusion_token_expires_in = self.credentials.get_fusion_token_expires_in(fusion_token_key)
                if fusion_token_expires_in and fusion_token_expires_in < self.refresh_within_seconds:
                    fusion_token, fusion_token_expiry = self._refresh_fusion_token_data(request)
                    self.credentials.put_fusion_token(fusion_token_key, fusion_token, int(fusion_token_expiry))
                    logger.log(VERBOSE_LVL, "Refreshed fusion token")

            request.headers.update(self.credentials.get_fusion_token_header(fusion_token_key))

        if self.credentials.fusion_e2e is not None:
            request.headers.update({"fusion-e2e": self.credentials.fusion_e2e})

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
