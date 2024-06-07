"""Fusion authentication module."""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlparse

import aiohttp
import fsspec
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import __version__ as version
from .exceptions import CredentialError

logger = logging.getLogger(__name__)
VERBOSE_LVL = 25


def try_get_env_var(var_name: str, default: Optional[str] = None) -> Optional[str]:
    """Get the value of an environment variable or return a default value.

    Args:
        var_name (str): The name of the environment variable.
        default (str, optional): The default value to return if the environment variable is not set. Defaults to None.

    Returns:
        Optional[str]: The value of the environment variable or the default value.
    """
    return os.environ.get(var_name, default)


def try_get_client_id(client_id: Optional[str]) -> Optional[str]:
    """Get the client ID from the environment variable or return None.

    Returns:
        Optional[str]: The client ID or None.
    """
    if client_id:
        return client_id
    return try_get_env_var("FUSION_CLIENT_ID")


def try_get_client_secret(client_secret: Optional[str]) -> Optional[str]:
    """Get the client secret from the environment variable or return None.

    Returns:
        Optional[str]: The client secret or None.
    """
    if client_secret:
        return client_secret
    return try_get_env_var("FUSION_CLIENT_SECRET")


def try_get_fusion_e2e(fusion_e2e: Optional[str]) -> Optional[str]:
    """Get the Fusion E2E token from the environment variable or return None.

    Returns:
        Optional[str]: The client secret or None.
    """
    if fusion_e2e:
        return fusion_e2e
    return try_get_env_var("FUSION_E2E")


def _res_plural(ref_int: int, pluraliser: str = "s") -> str:
    """Private function to return the plural form when the number is more than one.

    Args:
        ref_int (int): The reference integer that determines whether to return a plural suffix.
        pluraliser (str, optional): The plural suffix. Defaults to "s".

    Returns:
        str: The plural suffix to append to a string.
    """
    return "" if abs(ref_int) == 1 else pluraliser


def _is_json(data: str) -> bool:
    """Test whether the content of a string is a JSON object.

    Args:
        data (str): The content to evaluate.

    Returns:
        bool: True if the content of data is JSON, False otherwise.
    """
    try:
        json.loads(data)
    except ValueError:
        return False
    return True


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


def get_default_fs() -> fsspec.filesystem:
    """Retrieve default filesystem.

    Returns: filesystem

    """
    protocol = os.environ.get("FS_PROTOCOL", "file")
    if "S3_ENDPOINT" in os.environ and "AWS_ACCESS_KEY_ID" in os.environ and "AWS_SECRET_ACCESS_KEY" in os.environ:
        endpoint = os.environ["S3_ENDPOINT"]
        fs = fsspec.filesystem(
            "s3",
            client_kwargs={"endpoint_url": f"https://{endpoint}"},
            key=os.environ["AWS_ACCESS_KEY_ID"],
            secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
    else:
        fs = fsspec.filesystem(protocol)
    return fs


class FusionCredentials:
    """Utility functions to manage credentials."""

    def __init__(  # noqa: PLR0913
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        resource: Optional[str] = None,
        auth_url: Optional[str] = None,
        bearer_token: Optional[str] = None,
        bearer_token_expiry: Optional[datetime] = None,
        is_bearer_token_expirable: Optional[bool] = None,
        proxies: Optional[dict[str, str]] = None,
        grant_type: str = "client_credentials",
        fusion_e2e: Optional[str] = None,
    ) -> None:
        """Constructor for the FusionCredentials authentication management class.

        Args:
            client_id (str, optional): A valid OAuth client identifier. Defaults to None.
            client_secret (str, optional): A valid OAuth client secret. Defaults to None.
            username (str, optional): A valid username. Defaults to None.
            password (str, optional): A valid password for the username. Defaults to None.
            resource (str, optional): The OAuth audience. Defaults to None.
            auth_url (str, optional): URL for the OAuth authentication server. Defaults to None.
            bearer_token (str, optional): Bearer token. Defaults to None.
            bearer_token_expiry (datetime, optional): Bearer token expiry. Defaults to None.
            is_bearer_token_expirable (bool, optional): Is bearer token expirable. Defaults to None.
            proxies (dict, optional): Any proxy servers required to route HTTP and HTTPS requests to the internet.
            grant_type (str, optional): Allows the grant type to be changed to support different credential types.
                Defaults to client_credentials.
            fusion_e2e (str, Optional): Fusion E2E token. Defaults to None.
        """
        if proxies is None:
            proxies = {}
        self.client_id = try_get_client_id(client_id)
        self.client_secret = try_get_client_secret(client_secret)
        self.username = username
        self.password = password
        self.resource = resource
        self.auth_url = auth_url
        self.proxies = proxies
        self.grant_type = grant_type
        self.bearer_token = bearer_token
        self.bearer_token_expiry = bearer_token_expiry
        self.is_bearer_token_expirable = is_bearer_token_expirable
        self.fusion_e2e = try_get_fusion_e2e(fusion_e2e)

    @staticmethod
    def add_proxies(
        http_proxy: str,
        https_proxy: Optional[str] = None,
        credentials_file: str = "config/client_credentials.json",
    ) -> None:
        """A function to add proxies to an existing credentials files.
        This function can be called to add proxy addresses to a credential file downloaded from the Fusion website.

        Args:
            http_proxy (str): The HTTP proxy address.
            https_proxy (str): The HTTPS proxy address. If not specified then this will be the
                copied form the HTTP proxy.
            credentials_file (str, optional): The path and filename to store the credentials under.
                Path may be absolute or relative to current working directory.
                Defaults to 'config/client_credentials.json'.

        Returns:
            None
        """

        credentials = FusionCredentials.from_file(credentials_file)
        credentials.proxies["http"] = http_proxy
        if https_proxy is None:
            https_proxy = http_proxy

        credentials.proxies["https"] = https_proxy

        data: dict[str, Any] = {
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "resource": credentials.resource,
            "auth_url": credentials.auth_url,
            "proxies": credentials.proxies,
        }
        json_data = json.dumps(data, indent=4)
        with Path(credentials_file).open("w") as credentialsfile:
            credentialsfile.write(json_data)

    @staticmethod
    def generate_credentials_file(
        credentials_file: str = "config/client_credentials.json",
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        resource: str = "JPMC:URI:RS-93742-Fusion-PROD",
        auth_url: str = "https://authe.jpmorgan.com/as/token.oauth2",
        proxies: Optional[Union[str, dict[str, str]]] = None,
    ) -> "FusionCredentials":
        """Utility function to generate credentials file that can be used for authentication.

        Args:
            credentials_file (str, optional): The path and filename to store the credentials under.
                Path may be absolute or relative to current working directory.
                Defaults to 'config/client_credentials.json'.
            client_id (str, optional): A valid OAuth client identifier. Defaults to None.
            client_secret (str, optional): A valid OAuth client secret. Defaults to None.
            resource (str, optional): The OAuth audience. Defaults to None.
            auth_url (str, optional): URL for the OAuth authentication server. Defaults to None.
            proxies (Union[str, dict], optional): Any proxy servers required to route HTTP and HTTPS
                requests to the internet. Defaults to {}. Keys are http and https. Or specify a single
                URL to set both http and https

        Raises:
            CredentialError: Exception to handle missing values required for authentication.

        Returns:
           FusionCredentials: a credentials object that can be used for authentication.
        """
        if not client_id:
            raise CredentialError("A valid client_id is required")
        if not client_secret:
            raise CredentialError("A valid client secret is required")

        data: dict[str, Any] = {
            "client_id": client_id,
            "client_secret": client_secret,
            "resource": resource,
            "auth_url": auth_url,
        }

        proxies_resolved = {}
        if proxies:
            if isinstance(proxies, dict):
                raw_proxies_dict = proxies
            elif isinstance(proxies, str):
                if _is_url(proxies):
                    raw_proxies_dict = {"http": proxies, "https": proxies}
                elif _is_json(proxies):
                    raw_proxies_dict = json.loads(proxies)
                else:
                    raise CredentialError(f"A valid proxies param is required, [{proxies}] is not supported.")
            else:
                raise CredentialError(f"A valid proxies param is required, [{proxies}] is not supported.")

            # Now validate and conform proxies dict
            valid_pxy_keys = ["http", "https", "http_proxy", "https_proxy"]
            pxy_key_map = {
                "http": "http",
                "https": "https",
                "http_proxy": "http",
                "https_proxy": "https",
            }
            lcase_dict = {k.lower(): v for k, v in raw_proxies_dict.items()}

            if set(lcase_dict.keys()).intersection(set(valid_pxy_keys)) != set(lcase_dict.keys()):
                raise CredentialError(
                    f"Invalid proxies keys in dict {raw_proxies_dict.keys()}."
                    f"Only {pxy_key_map.keys()} are accepted and will be mapped as necessary."
                )
            proxies_resolved = {pxy_key_map[k]: v for k, v in lcase_dict.items()}

        data["proxies"] = proxies_resolved
        json_data = json.dumps(data, indent=4)
        Path(credentials_file).parent.mkdir(parents=True, exist_ok=True)
        with Path(credentials_file).open("w") as credentialsfile:
            credentialsfile.write(json_data)

        credentials = FusionCredentials.from_file(file_path=credentials_file)
        return credentials

    @staticmethod
    def from_dict(credentials: dict[str, Any]) -> "FusionCredentials":
        """Create a credentials object from a dictionary.

            This is the only FusionCredentials creation method that supports the password grant type
            since the username and password should be provided by the user.

        Args:
            credentials (dict): A dictionary containing the requried keys: client_id, client_secret,
                resource, auth_url, and optionally proxies and an OAuth grant type.

        Returns:
            FusionCredentials: a credentials object that can be used for authentication.
        """
        if not credentials or not isinstance(credentials, dict):
            raise CredentialError("A valid credentials dictionary is required")

        grant_type = credentials.get("grant_type", "client_credentials")
        try:
            if grant_type == "client_credentials":
                client_id = try_get_client_id(credentials.get("client_id"))
                client_secret = try_get_client_secret(credentials.get("client_secret"))
                username = None
                password = None
                bearer_token = None
                bearer_token_expiry = datetime.now(tz=timezone.utc)
                is_bearer_token_expirable = True
                resource = credentials["resource"]
                auth_url = credentials["auth_url"]
            elif grant_type == "bearer":
                client_id = None
                client_secret = None
                username = None
                password = None
                bearer_token = credentials["bearer_token"]
                bearer_token_expiry = (
                    pd.to_datetime(credentials.get("bearer_token_expiry"), utc=True)  # type: ignore
                    if credentials.get("bearer_token_expiry")
                    else datetime.now(tz=timezone.utc)
                )
                bearer_token_expirable = credentials.get("bearer_token_expirable")
                if isinstance(bearer_token_expirable, str):
                    bearer_token_expirable = bearer_token_expirable.lower()

                is_bearer_token_expirable = bearer_token_expirable not in ["false"] if bearer_token_expirable else True
                resource = None
                auth_url = None
            elif grant_type == "password":
                client_id = try_get_client_id(credentials.get("client_id"))
                client_secret = None
                username = credentials["username"]
                password = credentials["password"]
                bearer_token = None
                bearer_token_expiry = datetime.now(tz=timezone.utc)
                is_bearer_token_expirable = True
                resource = credentials["resource"]
                auth_url = credentials["auth_url"]
            else:
                raise CredentialError(f"Unrecognised grant type {grant_type}")
        except KeyError as e:
            raise CredentialError(f"Missing required key in credentials dictionary: {e}") from e
        fusion_e2e = credentials.get("fusion_e2e")
        proxies = credentials.get("proxies")
        creds = FusionCredentials(
            client_id,
            client_secret,
            username,
            password,
            resource,
            auth_url,
            bearer_token,
            bearer_token_expiry,
            is_bearer_token_expirable,
            proxies,
            grant_type=grant_type,
            fusion_e2e=fusion_e2e,
        )
        return creds

    @staticmethod
    def from_file(
        file_path: str = "config/client.credentials.json",
        fs: Optional[fsspec.filesystem] = None,
        _walk_up_dirs: bool = True,
    ) -> "FusionCredentials":
        """Create a credentials object from a file.

        Args:
            file_path (str, optional): Path (absolute or relative) and filename
                to load credentials from. Defaults to 'config/client.credentials.json'.
            fs (fsspec.filesystem): Filesystem to use.
            walk_up_dirs (bool): if true it walks up the directories in search of a config folder
        Returns:
            FusionCredentials: a credentials object that can be used for authentication.
        """
        fs = fs if fs else get_default_fs()
        p_path = Path(file_path)
        to_use_file_path: Optional[Path] = None

        if p_path.exists():  # exact match path case
            logger.log(VERBOSE_LVL, f"Found credentials file at {file_path}")
            to_use_file_path = Path(file_path)
        else:
            for p in p_path.absolute().parents:
                if (p / p_path).exists():
                    to_use_file_path = p / p_path
                    logger.log(VERBOSE_LVL, f"Found credentials file at {to_use_file_path}")
                    break
        if not to_use_file_path:
            raise FileNotFoundError(f"Credentials file not found at {p_path} or in any parent directory.")
        if fs.size(to_use_file_path) > 0:
            try:
                with fs.open(to_use_file_path, "r") as creds_f:
                    data = json.load(creds_f)
                    credentials = FusionCredentials.from_dict(data)  # noqa: PLW2901
                    return credentials
            except CredentialError as e:
                logger.error(e)
                raise e
        else:
            msg = f"{to_use_file_path} is an empty file, make sure to add your credentials to it."
            logger.error(msg)
            raise OSError(msg)

    @staticmethod
    def from_object(credentials_source: Union[str, dict[str, Any], "FusionCredentials"]) -> "FusionCredentials":
        """Utility function that will determine how to create a credentials object based on data passed.

        Args:
            credentials_source (Union[str, dict]): A string which could be a filename or a JSON object, or a dictionary.

        Raises:
            CredentialError: Exception raised when the provided credentials is not one of the supported types

        Returns:
            FusionCredentials: a credentials object that can be used for authentication.
        """
        if isinstance(credentials_source, FusionCredentials):
            return credentials_source
        elif isinstance(credentials_source, dict):
            return FusionCredentials.from_dict(credentials_source)
        elif isinstance(credentials_source, str):
            if _is_json(credentials_source):
                return FusionCredentials.from_dict(json.loads(credentials_source))
            return FusionCredentials.from_file(credentials_source)
        else:
            raise CredentialError(f"Could not resolve the credentials provided: {credentials_source}")


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
        credentials: Union[FusionCredentials, str, dict[str, Any]],
        proxies: Optional[dict[str, str]] = None,
        refresh_within_seconds: int = 5,
        auth_retries: Optional[Union[int, Retry]] = None,
        mount_url: str = "",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Class constructor to create a FusionOAuthAdapter object.

        Args:
            credentials (Union[FusionCredentials, Union[str, dict]): Valid user credentials to authenticate.
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

        if isinstance(credentials, FusionCredentials):
            self.credentials = credentials
        else:
            self.credentials = FusionCredentials.from_object(credentials)

        if proxies:
            self.proxies = proxies
        else:
            self.proxies = self.credentials.proxies
        self.token = None
        self.bearer_token_expiry = datetime.now(tz=timezone.utc)
        self.number_token_refreshes = 0
        self.refresh_within_seconds = refresh_within_seconds
        self.fusion_token_dict: dict[str, str] = {}
        self.fusion_token_expiry_dict: dict[str, Any] = {}
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

        if not self.credentials.bearer_token_expiry:
            raise CredentialError("A valid bearer token is required")
        token_expires_in = (self.credentials.bearer_token_expiry - datetime.now(tz=timezone.utc)).total_seconds()

        url_lst = request.path_url.split("/")
        fusion_auth_req = "distributions" in url_lst

        if self.credentials.is_bearer_token_expirable and token_expires_in < self.refresh_within_seconds:
            token, expiry = self._refresh_token_data()
            self.credentials.bearer_token = token
            self.credentials.bearer_token_expiry = datetime.now(tz=timezone.utc) + timedelta(seconds=int(expiry))
            self.number_token_refreshes += 1
            logger.log(
                VERBOSE_LVL,
                f"Refreshed token {self.number_token_refreshes} time{_res_plural(self.number_token_refreshes)}",
            )
        request.headers.update(
            {
                "Authorization": f"Bearer {self.credentials.bearer_token}",
                "User-Agent": f"fusion-python-sdk {version}",
            }
        )
        if fusion_auth_req:
            catalog = url_lst[url_lst.index("catalogs") + 1]
            dataset = url_lst[url_lst.index("datasets") + 1]
            fusion_token_key = catalog + "_" + dataset

            if fusion_token_key not in self.fusion_token_dict:
                fusion_token, fusion_token_expiry = self._refresh_fusion_token_data(request, **kwargs)
                self.fusion_token_dict[fusion_token_key] = fusion_token
                self.fusion_token_expiry_dict[fusion_token_key] = datetime.now(tz=timezone.utc) + timedelta(
                    seconds=int(fusion_token_expiry)
                )
                logger.log(VERBOSE_LVL, "Refreshed fusion token")
            else:
                fusion_token_expires_in = (
                    self.fusion_token_expiry_dict[fusion_token_key] - datetime.now(tz=timezone.utc)
                ).total_seconds()
                if fusion_token_expires_in < self.refresh_within_seconds:
                    fusion_token, fusion_token_expiry = self._refresh_fusion_token_data(request)
                    self.fusion_token_dict[fusion_token_key] = fusion_token
                    self.fusion_token_expiry_dict[fusion_token_key] = datetime.now(tz=timezone.utc) + timedelta(
                        seconds=int(fusion_token_expiry)
                    )
                    logger.log(VERBOSE_LVL, "Refreshed fusion token")

            request.headers.update({"Fusion-Authorization": f"Bearer {self.fusion_token_dict[fusion_token_key]}"})

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
