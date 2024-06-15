import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import fsspec
import pandas as pd

from fusion.exceptions import CredentialError
from fusion.utils import get_default_fs

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
                    credentials = FusionCredentials._internal_load(data)
                    return credentials
            except CredentialError as e:
                logger.error(e)
                raise e
        else:
            msg = f"{to_use_file_path} is an empty file, make sure to add your credentials to it."
            logger.error(msg)
            raise OSError(msg)

    @staticmethod
    def _internal_load(credentials: dict[str, Any]) -> "FusionCredentials":
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
