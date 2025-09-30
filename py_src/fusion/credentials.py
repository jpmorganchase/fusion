from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import jwt
import requests

from fusion.exceptions import CredentialError

from . import __version__

_DEFAULT_GRANT_TYPE = "client_credentials"
_DEFAULT_AUTH_URL = "https://authe.jpmorgan.com/as/token.oauth2"
VERSION = __version__


def _now_ts() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp())


def _start_of_day_utc(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


def _fusion_url_to_auth_url(url: str) -> tuple[str, str, str] | None:
    """
    If `url` is a distribution URL of the form:
      .../catalogs/{catalog}/datasets/{dataset}/distributions...
    return:
      ( .../catalogs/{catalog}/datasets/{dataset}/authorize/token, catalog, dataset )
    Otherwise return None.
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Could not parse URL (Error Code: 400)")

    segments = [s for s in parsed.path.split("/") if s]
    if "distributions" not in segments:
        return None

    try:
        cat = segments[segments.index("catalogs") + 1]
    except Exception as e:  # noqa: BLE001
        raise ValueError("'catalogs' segment not found or catalog name missing in the path (Error Code: 400)") from e

    try:
        ds_i = segments.index("datasets")
        ds = segments[ds_i + 1]
    except Exception as e:  # noqa: BLE001
        raise ValueError("'datasets' segment not found or dataset name missing in the path (Error Code: 400)") from e

    auth_path = "/".join(segments[: ds_i + 2] + ["authorize", "token"])
    return f"{parsed.scheme}://{parsed.netloc}/{auth_path}", cat, ds


def _find_cfg_file(file_path: Path) -> Path:
    """
    If file_path is a file, return it.
    Otherwise, search upwards for ./config/client_credentials.json
    starting from file_path.parent (if exists) or CWD.
    """
    p = Path(file_path)
    if p.is_file():
        return p

    start_dir = p.parent if p.parent.exists() else Path.cwd()
    start_dir = start_dir.resolve()
    anchor = start_dir

    while True:
        candidate = start_dir / "config" / "client_credentials.json"
        if candidate.is_file():
            return candidate
        if start_dir.parent == start_dir:
            raise FileNotFoundError(
                f"File client_credentials.json not found in {anchor} or any of its parents. Current parent: {p.parent}."
            )
        start_dir = start_dir.parent.resolve()


@dataclass
class AuthToken:
    token: str
    expiry: int | None  # absolute epoch seconds UTC, or None

    def is_expirable(self) -> bool:
        return self.expiry is not None

    def expires_in_secs(self) -> int | None:
        if self.expiry is None:
            return None
        return self.expiry - _now_ts()

    @staticmethod
    def from_token(token: str, expires_in_secs: int | None = None) -> AuthToken:
        expiry = None if expires_in_secs is None else _now_ts() + int(expires_in_secs)
        return AuthToken(token=token, expiry=expiry)


class FusionCredentials:
    client_id: str | None
    client_secret: str | None
    username: str | None
    password: str | None
    resource: str | None
    auth_url: str | None
    grant_type: str
    fusion_e2e: str | None
    proxies: dict[str, str]  # keep public annotation (non-breaking)
    bearer_token: AuthToken | None
    fusion_token: dict[str, AuthToken]
    headers: dict[str, str] | None
    kid: str | None
    private_key: str | None

    @classmethod
    def from_client_id(  # noqa: PLR0913
        cls,
        client_id: str | None,
        client_secret: str | None,
        resource: str | None,
        auth_url: str | None,
        proxies: dict[str, str] | None,
        fusion_e2e: str | None,
        headers: dict[str, str] | None = None,
        kid: str | None = None,
        private_key: str | None = None,
    ) -> FusionCredentials:
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            username=None,
            password=None,
            resource=resource,
            auth_url=auth_url,
            bearer_token=None,
            proxies=proxies,
            grant_type=_DEFAULT_GRANT_TYPE,
            fusion_e2e=fusion_e2e,
            headers=headers,
            kid=kid,
            private_key=private_key,
        )

    @classmethod
    def from_user_id(  # noqa: PLR0913
        cls,
        username: str | None,
        password: str | None,
        resource: str | None,
        auth_url: str | None,
        proxies: dict[str, str] | None,
        fusion_e2e: str | None,
        headers: dict[str, str] | None = None,
        kid: str | None = None,
        private_key: str | None = None,
    ) -> FusionCredentials:
        return cls(
            client_id=None,
            client_secret=None,
            username=username,
            password=password,
            resource=resource,
            auth_url=auth_url,
            bearer_token=None,
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
        resource: str | None,
        auth_url: str | None,
        bearer_token: str | None,
        bearer_token_expiry: date | None,
        proxies: dict[str, str] | None,
        fusion_e2e: str | None,
    ) -> FusionCredentials:
        if not bearer_token:
            raise ValueError("Bearer token not provided")
        expiry = _start_of_day_utc(bearer_token_expiry) if bearer_token_expiry else None
        return cls(
            client_id=None,
            client_secret=None,
            username=None,
            password=None,
            resource=resource,
            auth_url=auth_url,
            bearer_token=AuthToken(token=bearer_token, expiry=expiry),
            proxies=proxies,
            grant_type="bearer",
            fusion_e2e=fusion_e2e,
        )

    @classmethod
    def from_file(cls, file_path: Path) -> FusionCredentials:
        found = _find_cfg_file(Path(file_path))
        try:
            with found.open(encoding="utf-8") as fh:
                cfg = json.load(fh)
        except json.JSONDecodeError as e:  # noqa: TRY003
            with found.open(encoding="utf-8") as fh:
                contents = fh.read()
            raise CredentialError(ValueError(f"Invalid JSON: {e}\nContents:\n{contents}\nError Code: 400")) from e

        grant_type = cfg.get("grant_type", _DEFAULT_GRANT_TYPE)
        client_id = cfg.get("client_id") or os.environ.get("FUSION_CLIENT_ID")
        client_secret = cfg.get("client_secret") or os.environ.get("FUSION_CLIENT_SECRET")
        resource = cfg.get("resource")
        auth_url = cfg.get("auth_url")
        proxies = cfg.get("proxies") or {}
        fusion_e2e = cfg.get("fusion_e2e")
        kid = cfg.get("kid")
        private_key = cfg.get("private_key")

        if grant_type == "client_credentials":
            if not client_id:
                raise CredentialError(ValueError("Missing client ID (Error Code: 400)"))
            if client_secret is None:
                raise CredentialError(ValueError("Missing client secret (Error Code: 400)"))
            return cls.from_client_id(
                client_id=client_id,
                client_secret=client_secret,
                resource=resource,
                auth_url=auth_url,
                proxies=proxies,
                fusion_e2e=fusion_e2e,
                kid=kid,
                private_key=private_key,
            )
        if grant_type == "password":
            if not client_id:
                raise CredentialError(ValueError("Missing client ID (Error Code: 400)"))
            return cls(
                client_id=client_id,
                client_secret=None,
                username=cfg.get("username"),
                password=cfg.get("password"),
                resource=resource,
                auth_url=auth_url,
                bearer_token=None,
                proxies=proxies,
                grant_type="password",
                fusion_e2e=fusion_e2e,
                kid=kid,
                private_key=private_key,
            )
        if grant_type == "bearer":
            return cls.from_bearer_token(
                resource=resource,
                auth_url=auth_url,
                bearer_token=cfg.get("bearer_token"),
                bearer_token_expiry=None,
                proxies=proxies,
                fusion_e2e=fusion_e2e,
            )
        raise CredentialError(ValueError("Unrecognized grant type"))

    def __init__(  # noqa: PLR0913
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        username: str | None = None,
        password: str | None = None,
        resource: str | None = None,
        auth_url: str | None = None,
        bearer_token: AuthToken | None = None,
        proxies: dict[str, str] | None = None,
        grant_type: str | None = None,
        fusion_e2e: str | None = None,
        headers: dict[str, str] | None = None,
        kid: str | None = None,
        private_key: str | None = None,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password
        self.resource = resource
        self.auth_url = auth_url or _DEFAULT_AUTH_URL
        self.grant_type = grant_type or _DEFAULT_GRANT_TYPE
        self.fusion_e2e = fusion_e2e
        # private backing field; exposes `proxies` via property while keeping public annotation
        self._proxies: dict[str, str] = dict(proxies or {})
        self.bearer_token = bearer_token
        self.fusion_token: dict[str, AuthToken] = {}

        # lazy HTTP session
        self._session: requests.Session | None = None

        self.headers = headers or {}
        self.kid = kid
        self.private_key = private_key

    @property  # type: ignore[no-redef]
    def proxies(self) -> dict[str, str]:
        return self._proxies

    @proxies.setter
    def proxies(self, value: dict[str, str] | None) -> None:
        self._proxies = dict(value or {})

    def put_bearer_token(self, bearer_token: str, expires_in_secs: int | None) -> None:
        self.bearer_token = AuthToken.from_token(bearer_token, expires_in_secs)

    def put_fusion_token(self, token_key: str, token: str, expires_in_secs: int | None) -> None:
        self.fusion_token[token_key] = AuthToken.from_token(token, expires_in_secs)

    def get_bearer_token_header(self) -> tuple[tuple[str, str], ...]:
        if not self.bearer_token:
            raise ValueError("No bearer token set (Error Code: 400)")
        return (("Authorization", f"Bearer {self.bearer_token.token}"),)

    def get_fusion_token_header(self, token_key: str) -> tuple[tuple[str, str], ...]:
        tok = self.fusion_token.get(token_key)
        if not tok:
            raise ValueError(f"No fusion token for key '{token_key}' (Error Code: 400)")
        return (("Fusion-Authorization", f"Bearer {tok.token}"),)

    def get_fusion_token_expires_in(self, token_key: str) -> int | None:
        tok = self.fusion_token.get(token_key)
        return tok.expires_in_secs() if tok else None

    def _session_or_new(self) -> requests.Session:
        if self._session is None:
            s = requests.Session()
            if self._proxies:
                s.proxies.update(self._proxies)
            self._session = s
        return self._session

    def refresh_bearer_token(self) -> None:
        """
        Use OAuth2 to fetch a bearer access_token and cache it.
        - client_credentials: grant_type, client_id, client_secret, aud=resource
        - password: grant_type, client_id, username, password, resource
        - bearer: noop
        Stores expiry using 'expires_in' (seconds) if present.
        """
        if self.grant_type == "bearer":
            return

        sess = self._session_or_new()
        payload: dict[str, str] = {"grant_type": self.grant_type}

        # narrow Optional[str] to str for mypy/type-checking
        auth_url: str = self.auth_url or _DEFAULT_AUTH_URL

        if self.kid and self.private_key:
            # JWT-based client assertion
            # Construct claims
            iat = int(datetime.now(dt.timezone.utc).timestamp())
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
            payload = {
                "grant_type": self.grant_type,
                "client_id": self.client_id or "",
                "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
                "client_assertion": private_key_jwt
                if isinstance(private_key_jwt, str)
                else private_key_jwt.decode("utf-8"),  # type: ignore[attr-defined]
                "resource": self.resource or "",
            }
        elif self.grant_type == "client_credentials":
            payload.update(
                {
                    "client_id": self.client_id or "",
                    "client_secret": self.client_secret or "",
                }
            )
            if self.resource:
                # Rust used 'aud' — many OAuth servers accept this.
                payload["aud"] = self.resource
        elif self.grant_type == "password":
            payload.update(
                {
                    "client_id": self.client_id or "",
                    "username": self.username or "",
                    "password": self.password or "",
                }
            )
            if self.resource:
                payload["resource"] = self.resource
        else:
            raise ValueError("Unrecognized grant type")

        resp = sess.post(auth_url, data=payload, headers={"User-Agent": f"fusion-python-sdk {VERSION}"})
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:  # noqa: TRY003
            raise ValueError(f"Could not post request: {e} (Error Code: {resp.status_code})") from e

        try:
            data = resp.json()
        except ValueError as e:  # noqa: TRY003
            raise ValueError(f"Could not parse text to json: {e} (Error Code: 500)") from e

        token = data.get("access_token")
        if not token:
            raise ValueError("Missing access_token in generate token response (Error Code: 500)")
        expires_in = data.get("expires_in")
        self.put_bearer_token(token, int(expires_in) if expires_in is not None else None)

    def get_fusion_token_headers(self, url: str) -> dict[str, str]:  # noqa: PLR0912
        """
        Returns headers required for Fusion distribution endpoints.
        If `url` is not a distribution request, only returns Bearer Authorization.
        If it is, also fetches/caches a short-lived Fusion token for the
        (catalog, dataset) pair and includes 'Fusion-Authorization'.
        """
        # Ensure we have a fresh-enough bearer token (simple policy: if missing or expiring <= 15m, refresh)
        refresh_needed = False
        if not self.bearer_token:
            refresh_needed = True
        else:
            remains = self.bearer_token.expires_in_secs()
            if remains is not None and remains <= 15 * 60:
                refresh_needed = True
        if refresh_needed:
            self.refresh_bearer_token()
        if not self.bearer_token:
            raise ValueError("No bearer token set (Error Code: 400)")

        headers: dict[str, str] = {"User-Agent": f"fusion-python-sdk {VERSION}"}
        if self.fusion_e2e:
            headers["fusion-e2e"] = self.fusion_e2e

        # Always include standard bearer
        headers["Authorization"] = f"Bearer {self.bearer_token.token}"

        if self.headers:
            headers.update(self.headers)

        info = _fusion_url_to_auth_url(url)
        if not info:
            return headers

        fusion_auth_url, catalog, dataset = info
        token_key = f"{catalog}_{dataset}"

        # If we have a non-expiring or not-soon-expiring cached token, reuse it.
        tok = self.fusion_token.get(token_key)
        if tok:
            remain = tok.expires_in_secs()
            if remain is None or remain > 15 * 60:
                headers["Fusion-Authorization"] = f"Bearer {tok.token}"
                return headers

        # Need (re)fetch Fusion token via GET to fusion_auth_url with Bearer
        sess = self._session_or_new()
        resp = sess.get(
            fusion_auth_url,
            headers={
                "Authorization": f"Bearer {self.bearer_token.token}",
                "User-Agent": f"fusion-python-sdk {VERSION}",
            },
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:  # noqa: TRY003
            raise ValueError(f"Error from endpoint: {e} (Error Code: {resp.status_code})") from e

        try:
            data = resp.json()
        except ValueError as e:  # noqa: TRY003
            raise ValueError(f"Could not parse response to json: {e} (Error Code: 500)") from e

        fusion_tok = data.get("access_token")
        if not fusion_tok:
            raise ValueError("Missing access_token in fusion token response (Error Code: 500)")
        expires_in = data.get("expires_in")
        self.put_fusion_token(token_key, fusion_tok, int(expires_in) if expires_in is not None else None)

        headers["Fusion-Authorization"] = f"Bearer {fusion_tok}"
        return headers
