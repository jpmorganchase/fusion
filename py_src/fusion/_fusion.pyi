from __future__ import annotations  # noqa: PYI044

from datetime import date
from pathlib import Path

class AuthToken:
    token: str
    expiry: int | None

    def is_expirable(self) -> bool: ...
    def expires_in_secs(self) -> int | None: ...
    @staticmethod
    def from_token(token: str, expires_in_secs: int | None) -> AuthToken: ...

class FusionCredentials:
    client_id: str | None
    client_secret: str | None
    username: str | None
    password: str | None
    resource: str | None
    auth_url: str | None
    grant_type: str
    fusion_e2e: str | None
    proxies: dict[str, str]
    bearer_token: AuthToken | None
    fusion_token: dict[str, AuthToken]

    @classmethod
    def from_client_id(
        cls: type[FusionCredentials],
        client_id: str | None,
        client_secret: str | None,
        resource: str | None,
        auth_url: str | None,
        proxies: dict[str, str] | None,
        fusion_e2e: str | None,
    ) -> FusionCredentials: ...
    @classmethod
    def from_user_id(
        cls: type[FusionCredentials],
        username: str | None,
        password: str | None,
        resource: str | None,
        auth_url: str | None,
        proxies: dict[str, str] | None,
        fusion_e2e: str | None,
    ) -> FusionCredentials: ...
    @classmethod
    def from_bearer_token(
        cls: type[FusionCredentials],
        resource: str | None,
        auth_url: str | None,
        bearer_token: str | None,
        bearer_token_expiry: date | None,
        proxies: dict[str, str] | None,
        fusion_e2e: str | None,
    ) -> FusionCredentials: ...
    @classmethod
    def from_file(cls: type[FusionCredentials], file_path: Path) -> FusionCredentials: ...
    def __init__(  # noqa: PLR0913
        self,
        client_id: str | None,
        client_secret: str | None,
        username: str | None,
        password: str | None,
        resource: str | None,
        auth_url: str | None,
        bearer_token: AuthToken | None,
        proxies: dict[str, str] | None,
        grant_type: str | None,
        fusion_e2e: str | None,
    ) -> None: ...
    def put_bearer_token(self, bearer_token: str, expires_in_secs: int | None) -> None: ...
    def put_fusion_token(self, token_key: str, token: str, expires_in_secs: int | None) -> None: ...
    def get_bearer_token_header(self) -> tuple[tuple[str, str], ...]: ...
    def get_fusion_token_header(self, token_key: str) -> tuple[tuple[str, str], ...]: ...
    def get_fusion_token_expires_in(self, token_key: str) -> int | None: ...
    def refresh_bearer_token(self) -> None: ...
    def get_fusion_token_headers(self, url: str) -> dict[str, str]: ...

def rust_ok() -> bool: ...
