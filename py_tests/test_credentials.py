# py_tests/test_credentials.py
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from fusion.credentials import AuthToken, FusionCredentials
from fusion.exceptions import CredentialError


def test_auth_token_basic_expirable() -> None:
    tok = AuthToken.from_token("tkn", 3600)
    assert tok.token == "tkn"
    assert tok.is_expirable() is True

    remaining = tok.expires_in_secs()
    assert remaining is not None
    assert remaining <= 3600  # noqa: PLR2004

    tok2 = AuthToken.from_token("tkn2", None)
    assert tok2.is_expirable() is False
    assert tok2.expires_in_secs() is None


def test_from_client_id_sets_fields() -> None:
    creds = FusionCredentials.from_client_id(
        client_id="id",
        client_secret="secret",
        resource="res",
        auth_url="https://auth.example",
        proxies={"https": "https://proxy"},
        fusion_e2e="e2e",
        headers={"key": "value"},
        kid="kid",
        private_key="privkey",
    )
    assert creds.client_id == "id"
    assert creds.client_secret == "secret"
    assert creds.resource == "res"
    assert creds.auth_url == "https://auth.example"
    assert creds.grant_type == "client_credentials"
    assert creds.proxies.get("https") == "https://proxy"
    assert creds.fusion_e2e == "e2e"
    assert creds.headers == {"key": "value"}
    assert creds.kid == "kid"
    assert creds.private_key == "privkey"


def test_from_user_id_sets_fields() -> None:
    creds = FusionCredentials.from_user_id(
        username="u",
        password="p",
        resource="res",
        auth_url="https://auth.example",
        proxies=None,
        fusion_e2e=None,
    )
    assert creds.username == "u"
    assert creds.password == "p"
    assert creds.grant_type == "password"


def test_from_bearer_token_expiry_at_midnight() -> None:
    d = date(2024, 11, 1)
    creds = FusionCredentials.from_bearer_token(
        resource=None,
        auth_url=None,
        bearer_token="bt",
        bearer_token_expiry=d,
        proxies=None,
        fusion_e2e=None,
    )
    assert creds.grant_type == "bearer"
    assert creds.bearer_token is not None
    assert isinstance(creds.bearer_token.expiry, int) or creds.bearer_token.expiry is None


def test_put_and_get_headers() -> None:
    creds = FusionCredentials.from_client_id("id", "sec", "res", "https://auth", None, None)
    creds.put_bearer_token("abc", 10)
    hdr = creds.get_bearer_token_header()
    assert hdr == (("Authorization", "Bearer abc"),)

    creds.put_fusion_token("k", "xyz", 10)
    fhdr = creds.get_fusion_token_header("k")
    assert fhdr == (("Fusion-Authorization", "Bearer xyz"),)

    assert creds.get_fusion_token_expires_in("k") is not None


class _MockResp:
    def __init__(self, json_data: dict[str, Any], status_code: int = 200) -> None:
        self._json = json_data
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:  # noqa: PLR2004
            raise RuntimeError(f"HTTP {self.status_code}")


def test_refresh_bearer_token_client_credentials_makes_request() -> None:
    creds = FusionCredentials.from_client_id("id", "sec", "res", "https://auth", None, None)

    mock_sess = MagicMock()
    mock_sess.post.return_value = _MockResp({"access_token": "newbt", "expires_in": 3600})

    # Patch the Session constructor used in the module
    with patch("fusion.credentials.requests.Session", return_value=mock_sess):
        creds.refresh_bearer_token()

    assert creds.bearer_token is not None
    assert creds.bearer_token.token == "newbt"
    mock_sess.post.assert_called_once()
    # Ensure header helper works
    assert creds.get_bearer_token_header() == (("Authorization", "Bearer newbt"),)


def test_get_fusion_token_headers_non_distribution_uses_only_bearer() -> None:
    creds = FusionCredentials.from_client_id("id", "sec", "res", "https://auth", None, None)

    mock_sess = MagicMock()
    mock_sess.post.return_value = _MockResp({"access_token": "newbt", "expires_in": 3600})

    with patch("fusion.credentials.requests.Session", return_value=mock_sess):
        # refresh happens implicitly in get_fusion_token_headers when missing/expiring
        headers = creds.get_fusion_token_headers("https://api.example.com/catalogs/x/datasets/y")  # no distributions

    assert headers["Authorization"] == "Bearer newbt"
    assert "Fusion-Authorization" not in headers


def test_get_fusion_token_headers_distribution_fetches_and_caches() -> None:
    creds = FusionCredentials.from_client_id("id", "sec", "res", "https://auth", None, None)

    mock_sess = MagicMock()
    # 1) bearer refresh
    mock_sess.post.return_value = _MockResp({"access_token": "bt", "expires_in": 3600})
    # 2) fusion token GET
    mock_sess.get.return_value = _MockResp({"access_token": "ft", "expires_in": 1800})

    with patch("fusion.credentials.requests.Session", return_value=mock_sess):
        url = "https://host/catalogs/c1/datasets/d1/distributions"
        headers_1 = creds.get_fusion_token_headers(url)
        # second call should reuse cached fusion token if not near expiry
        headers_2 = creds.get_fusion_token_headers(url)

    assert headers_1["Authorization"] == "Bearer bt"
    assert headers_1["Fusion-Authorization"] == "Bearer ft"
    assert headers_2["Fusion-Authorization"] == "Bearer ft"

    # post called once (bearer), get called once (fusion token fetch)
    assert mock_sess.post.call_count == 1
    assert mock_sess.get.call_count == 1


def test_from_file_relative_walkup(tmp_path: Path) -> None:
    # ./level_1/level_2/level_3 ; put config two levels up at ./level_1/config/client_credentials.json
    level_3 = tmp_path / "level_1" / "level_2" / "level_3"
    level_3.mkdir(parents=True)
    (tmp_path / "level_1" / "config").mkdir()
    cfg_path = tmp_path / "level_1" / "config" / "client_credentials.json"
    cfg_path.write_text(
        json.dumps(
            {
                "grant_type": "client_credentials",
                "client_id": "cid",
                "client_secret": "csec",
                "resource": "res",
                "auth_url": "https://auth",
            }
        )
    )

    # chdir into level_3 for relative search
    cwd = Path.cwd()
    try:
        os.chdir(level_3)  # noqa: PTH100
        creds = FusionCredentials.from_file(Path("client_credentials.json"))
    finally:
        os.chdir(cwd)  # noqa: PTH100

    assert isinstance(creds, FusionCredentials)
    assert creds.client_id == "cid"
    assert creds.client_secret == "csec"


def test_from_file_invalid_json_raises_error(tmp_path: Path) -> None:
    # To hit the malformed JSON branch, write invalid JSON
    bad = tmp_path / "bad.json"
    bad.write_text('{"client_id": "x",')  # malformed
    with pytest.raises(CredentialError):
        FusionCredentials.from_file(bad)
