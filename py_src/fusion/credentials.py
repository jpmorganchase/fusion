"""Credentials and token management for Fusion API."""

import datetime
import json
import logging
import os
from datetime import datetime as dt
from typing import Any, Optional
from urllib.parse import urlparse

import jwt
import requests

from . import __version__
from .exceptions import CredentialError

logger = logging.getLogger(__name__)
DEFAULT_GRANT_TYPE = "client_credentials"
DEFAULT_AUTH_URL = "https://authe.jpmorgan.com/as/token.oauth2"
VERSION = __version__


class ProxyType:
    HTTP = "http"
    HTTPS = "https"

    @staticmethod
    def from_str(s: str) -> str:
        if s == "http":
            return ProxyType.HTTP
        elif s == "https":
            return ProxyType.HTTPS
        else:
            raise ValueError("Unrecognized proxy type")


def find_cfg_file(file_path: str) -> str:
    # Attempts to find the config file starting from file_path or its parents
    # until it finds "config/client_credentials.json"
    file_path = os.path.abspath(file_path)  # noqa: PTH100

    if os.path.isfile(file_path):  # noqa: PTH113
        logger.debug(f"Found file at the provided path: {file_path}")
        return file_path

    cwd = os.getcwd()  # noqa: PTH109
    cfg_file_name = "client_credentials.json"
    cfg_folder_name = "config"

    start_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else cwd  # noqa: PTH120
    start_dir_init = os.path.abspath(start_dir)  # noqa: PTH100

    start_dir_abs = os.path.abspath(start_dir)  # noqa: PTH100
    while True:
        full_path = os.path.join(start_dir_abs, cfg_folder_name, cfg_file_name)  # noqa: PTH118
        if os.path.isfile(full_path):  # noqa: PTH113
            logger.debug(f"Found file at: {full_path}")
            return full_path
        parent = os.path.dirname(start_dir_abs)  # noqa: PTH120
        if parent == start_dir_abs:
            # Reached root directory
            error_message = (
                f"File {cfg_file_name} not found in {start_dir_init} or any of its parents. Current parent: {start_dir}"
            )
            raise FileNotFoundError(error_message)
        start_dir_abs = parent


def fusion_url_to_auth_url(url: str) -> Optional[tuple[str, str, str]]:
    logger.debug(f"Trying to form fusion auth url from: {url}")
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL")
    except ValueError as err:
        raise CredentialError(err, f"Could not parse URL: {url}, status code: 400") from err

    segments = [seg for seg in parsed.path.split("/") if seg]

    # Not a distribution request. No need to authorize
    if "distributions" not in segments:
        logger.debug("Not a distribution request. No need to authorize")
        return None

    # Extract catalog_name
    try:
        cat_idx = segments.index("catalogs")
        catalog_name = segments[cat_idx + 1]
    except (ValueError, IndexError) as err:
        status_code = getattr(err, "status_code", 400)
        raise CredentialError(
            err, f"'catalogs' segment not found or catalog name missing in the path, status code: {status_code}"
        ) from err

    # Extract dataset_name
    try:
        ds_idx = segments.index("datasets")
        dataset_name = segments[ds_idx + 1]
    except (ValueError, IndexError) as err:
        status_code = getattr(err, "status_code", 400)
        raise CredentialError(
            err, f"'datasets' segment not found or dataset name missing in the path (status code: {status_code})"
        ) from err

    logger.debug(f"Found Catalog: {catalog_name}, Dataset: {dataset_name}")

    # Reconstruct path until datasets + 1 segment: /catalogs/.../datasets/...
    # Then append /authorize/token
    # segments[:ds_idx+2] gives us everything including the dataset name
    new_path_segments = segments[: ds_idx + 2] + ["authorize", "token"]
    new_path = "/" + "/".join(new_path_segments)

    # Construct the new URL
    # Reuse scheme, netloc from parsed
    fusion_tk_url = f"{parsed.scheme}://{parsed.netloc}{new_path}"
    logger.debug(f"Fusion token URL: {fusion_tk_url}")
    return (fusion_tk_url, catalog_name, dataset_name)


class AuthToken:
    def __init__(self, token: str, expires_in_secs: Optional[int] = None) -> None:
        current_time = int(dt.now(datetime.timezone.utc).timestamp())
        self.token = token
        if expires_in_secs is not None:
            self.expiry = current_time + expires_in_secs
        else:
            self.expiry = None  # type: ignore[assignment]

    def is_expirable(self) -> bool:
        return self.expiry is not None

    def expires_in_secs(self) -> Optional[int]:
        if self.expiry is not None:
            return self.expiry - int(dt.now(datetime.timezone.utc).timestamp())
        return None

    @staticmethod
    def from_token(token: str, expires_in_secs: Optional[int] = None) -> "AuthToken":
        return AuthToken(token, expires_in_secs)

    def as_bearer_header(self) -> tuple[str, str]:
        return ("Authorization", f"Bearer {self.token}")

    def as_fusion_header(self) -> tuple[str, str]:
        return ("Fusion-Authorization", f"Bearer {self.token}")


class FusionCredsPersistent:
    def __init__(  # noqa: PLR0913
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        resource: Optional[str] = None,
        auth_url: Optional[str] = None,
        root_url: Optional[str] = None,
        proxies: Optional[dict[str, str]] = None,
        grant_type: Optional[str] = None,
        fusion_e2e: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        kid: Optional[str] = None,
        private_key: Optional[str] = None,
        bearer_token: Optional[str] = None,
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password
        self.resource = resource
        self.auth_url = auth_url
        self.root_url = root_url
        self.proxies = proxies or {}
        self.grant_type = grant_type or DEFAULT_GRANT_TYPE
        self.fusion_e2e = fusion_e2e
        self.headers = headers or {}
        self.kid = kid
        self.private_key = private_key
        self.bearer_token = bearer_token


class FusionCredentials:
    def __init__(  # noqa: PLR0913
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        resource: Optional[str] = None,
        auth_url: Optional[str] = None,
        bearer_token: Optional[AuthToken] = None,
        proxies: Optional[dict[str, str]] = None,
        grant_type: Optional[str] = None,
        fusion_e2e: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        kid: Optional[str] = None,
        private_key: Optional[str] = None,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password
        self.resource = resource
        self.auth_url = auth_url or DEFAULT_AUTH_URL
        self.bearer_token = bearer_token
        self.fusion_token = {}  # type: ignore[var-annotated]
        self.proxies = proxies or {}
        self.grant_type = grant_type or DEFAULT_GRANT_TYPE
        self.fusion_e2e = fusion_e2e
        self.headers = headers or {}
        self.kid = kid
        self.private_key = private_key
        self.http_proxies = self.proxies

    @classmethod
    def from_client_id(  # noqa: PLR0913
        cls,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        resource: Optional[str] = None,
        auth_url: Optional[str] = None,
        proxies: Optional[dict[str, str]] = None,  # Added type annotation for proxies
        fusion_e2e: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        kid: Optional[str] = None,
        private_key: Optional[str] = None,
    ) -> "FusionCredentials":
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            resource=resource,
            auth_url=auth_url or DEFAULT_AUTH_URL,
            proxies=proxies,
            grant_type="client_credentials",
            fusion_e2e=fusion_e2e,
            headers=headers,
            kid=kid,
            private_key=private_key,
        )

    @classmethod
    def from_user_id(  # noqa: PLR0913
        cls,
        client_id: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        resource: Optional[str] = None,
        auth_url: Optional[str] = None,
        proxies: Optional[dict[str, str]] = None,
        fusion_e2e: Optional[str] = None,
        headers: Optional[dict[Any, Any]] = None,
        kid: Optional[str] = None,
        private_key: Optional[str] = None,
    ) -> "FusionCredentials":
        return cls(
            client_id=client_id,
            username=username,
            password=password,
            resource=resource,
            auth_url=auth_url or DEFAULT_AUTH_URL,
            proxies=proxies,
            grant_type="password",
            fusion_e2e=fusion_e2e,
            headers=headers,
            kid=kid,
            private_key=private_key,
        )

    @classmethod
    def from_bearer_token(
        cls,
        bearer_token: Optional[str] = None,
        bearer_token_expiry: Optional[datetime.datetime] = None,
        proxies: Optional[dict[str, str]] = None,
        fusion_e2e: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> "FusionCredentials":
        if bearer_token is None:
            raise ValueError("Bearer token not provided")
        expiry_secs = None
        if bearer_token_expiry is not None:
            # bearer_token_expiry is a date, we assume midnight expiration
            expiry_dt = datetime.datetime(  # noqa: DTZ001
                bearer_token_expiry.year,
                bearer_token_expiry.month,
                bearer_token_expiry.day,
            )
            expiry_secs = int(expiry_dt.timestamp()) - int(dt.now(datetime.timezone.utc).timestamp())
        return cls(
            bearer_token=AuthToken.from_token(bearer_token, expiry_secs),
            proxies=proxies,
            grant_type="bearer",
            fusion_e2e=fusion_e2e,
            headers=headers,
        )

    @classmethod
    def from_file(cls, file_path: str) -> "FusionCredentials":
        found_path = find_cfg_file(file_path)
        with open(found_path) as f:  # noqa: PTH123
            contents = f.read()
        try:
            creds_data = json.loads(contents)
        except json.JSONDecodeError as e:
            status_code = getattr(e, "status_code", 400)
            raise CredentialError(e, f"Invalid JSON: {e}\nContents:\n{contents}, status code: {status_code}") from e

        creds = FusionCredsPersistent(**creds_data)

        client_id = creds.client_id or os.environ.get("FUSION_CLIENT_ID")
        if client_id is None:
            raise CredentialError(Exception("No client ID provided"), "Missing client ID, status code: 400")

        if creds.grant_type == "client_credentials":
            client_secret = creds.client_secret or os.environ.get("FUSION_CLIENT_SECRET")
            if client_secret is None:
                raise CredentialError(Exception("No client secret provided"), "Missing client secret, status code: 400")
            return cls.from_client_id(
                client_id=client_id,
                client_secret=client_secret,
                resource=creds.resource,
                auth_url=creds.auth_url,
                proxies=creds.proxies,
                fusion_e2e=creds.fusion_e2e,
                headers=creds.headers,
                kid=creds.kid,
                private_key=creds.private_key,
            )
        elif creds.grant_type == "bearer":
            return cls.from_bearer_token(
                bearer_token=creds.bearer_token,
                bearer_token_expiry=None,
                proxies=creds.proxies,
                fusion_e2e=creds.fusion_e2e,
                headers=creds.headers,
            )
        elif creds.grant_type == "password":
            return cls.from_user_id(
                client_id=client_id,
                username=creds.username,
                password=creds.password,
                resource=creds.resource,
                auth_url=creds.auth_url,
                proxies=creds.proxies,
                fusion_e2e=creds.fusion_e2e,
                headers=creds.headers,
                kid=creds.kid,
                private_key=creds.private_key,
            )
        else:
            raise ValueError("Unrecognized grant type")

    def put_bearer_token(self, bearer_token: str, expires_in_secs: Optional[int] = None) -> None:
        self.bearer_token = AuthToken.from_token(bearer_token, expires_in_secs)

    def put_fusion_token(self, token_key: str, token: str, expires_in_secs: Optional[int] = None) -> None:
        self.fusion_token[token_key] = AuthToken.from_token(token, expires_in_secs)

    def _refresh_bearer_token(self, force: bool = True, max_remain_secs: int = 30) -> bool:
        if not force and self.bearer_token is not None:
            # Check expiration
            if not self.bearer_token.is_expirable():
                return False
            remain = self.bearer_token.expires_in_secs()
            if remain is not None and remain > max_remain_secs:
                return False

        if self.grant_type == "bearer":
            return True  # Nothing to do

        payload = []
        if self.kid and self.private_key:
            # JWT-based client assertion
            # Construct claims
            iat = int(dt.now(datetime.timezone.utc).timestamp())
            exp = iat + 3600
            claims = {
                "iss": self.client_id or "",
                "aud": self.auth_url or "",
                "sub": self.client_id or "",
                "iat": iat,
                "exp": exp,
                "jti": "id001",
            }
            private_key_jwt = jwt.encode(claims, self.private_key, algorithm="RS256", headers={"kid": self.kid})
            payload = [
                ("grant_type", self.grant_type),
                ("client_id", self.client_id or ""),
                (
                    "client_assertion_type",
                    "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
                ),
                (
                    "client_assertion",
                    private_key_jwt if isinstance(private_key_jwt, str) else private_key_jwt.decode("utf-8"),
                ),
                ("resource", self.resource or ""),
            ]
        elif self.grant_type == "client_credentials":
            payload = [
                ("grant_type", self.grant_type),
                ("client_id", self.client_id or ""),
                ("client_secret", self.client_secret or ""),
                ("aud", self.resource or ""),
            ]
        elif self.grant_type == "password":
            payload = [
                ("grant_type", self.grant_type),
                ("client_id", self.client_id or ""),
                ("username", self.username or ""),
                ("password", self.password or ""),
                ("resource", self.resource or ""),
            ]
        else:
            raise ValueError("Unrecognized grant type")

        response = requests.post(
            self.auth_url,
            headers={"User-Agent": f"fusion-python-sdk {VERSION}"},
            data=payload,
            proxies=self.http_proxies,
        )
        if response.status_code != 200:  # noqa: PLR2004
            raise CredentialError(
                Exception(f"API returned a {response.status_code}"), f"Could not get bearer token: {response.text}"
            )
        res_json = response.json()
        logger.debug(f"Called for Bearer token. Response: {res_json}")

        token = res_json.get("access_token")
        expires_in = res_json.get("expires_in")
        self.put_bearer_token(token, expires_in)
        return True

    def _gen_fusion_token(self, url: str) -> Optional[tuple[str, Optional[int]]]:
        if not self.bearer_token:
            raise CredentialError(Exception("Bearer token is not set"), "Bearer token is missing, status code: 401")

        headers = {
            "Authorization": f"Bearer {self.bearer_token.token}",
            "User-Agent": f"fusion-python-sdk {VERSION}",
        }

        response = requests.get(url, headers=headers, proxies=self.http_proxies)
        if response.status_code != 200:  # noqa: PLR2004
            raise CredentialError(
                Exception(f"API returned a {response.status_code}"),
                f"Error from endpoint (status code: {response.status_code}): {response.text}",
            )
        res_json = response.json()
        token = res_json.get("access_token")
        expires_in = res_json.get("expires_in")
        logger.debug(f"Got Fusion token, expires in: {expires_in}")
        return token, expires_in

    def get_fusion_token_headers(self, url: str) -> dict[str, str]:
        # Refresh bearer token if needed
        self._refresh_bearer_token(force=False, max_remain_secs=900)  # 15 mins

        ret = {"User-Agent": f"fusion-python-sdk {VERSION}"}
        if self.fusion_e2e:
            ret["fusion-e2e"] = self.fusion_e2e

        if self.headers:
            for k, v in self.headers.items():
                ret[k] = v  # noqa: PERF403

        if not self.bearer_token:
            raise CredentialError(Exception("No bearer token found"), "No bearer token set (status code: 401)")
        bearer_key, bearer_value = self.bearer_token.as_bearer_header()

        fusion_info = fusion_url_to_auth_url(url)
        if fusion_info is None:
            # Not a distribution request
            ret[bearer_key] = bearer_value
            logger.debug(f"Headers are {ret}")
            return ret

        fusion_tk_url, catalog_name, dataset_name = fusion_info
        token_key = f"{catalog_name}_{dataset_name}"

        # Check existing fusion token
        token_entry = self.fusion_token.get(token_key)
        need_new_token = True
        if token_entry:
            expires_in_secs = token_entry.expires_in_secs()
            if expires_in_secs is not None and expires_in_secs > 900:  # noqa: PLR2004
                # Enough time remains, no need to refresh
                need_new_token = False

        if need_new_token:
            new_token, new_expires_in = self._gen_fusion_token(fusion_tk_url)  # type: ignore[misc]
            self.put_fusion_token(token_key, new_token, new_expires_in)
            fusion_key, fusion_value = self.fusion_token[token_key].as_fusion_header()
        else:
            fusion_key, fusion_value = token_entry.as_fusion_header()  # type: ignore[union-attr]

        ret[bearer_key] = bearer_value
        ret[fusion_key] = fusion_value

        logger.debug(f"Headers are {ret}")
        return ret

    def get_fusion_token_expires_in(self, token_key: str) -> Optional[int]:
        token = self.fusion_token.get(token_key)
        if token is None:
            return None
        return token.expires_in_secs()  # type: ignore[no-any-return]
