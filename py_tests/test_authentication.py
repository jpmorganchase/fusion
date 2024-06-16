import json
import os
import pickle
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional
from unittest.mock import MagicMock, Mock, patch

import fsspec
import pytest
import requests
import requests_mock
from freezegun import freeze_time

from fusion._fusion import FusionCredentials
from fusion._legacy.authentication import (
    try_get_client_id,
    try_get_client_secret,
)
from fusion.authentication import (
    FusionOAuthAdapter,
)
from fusion.exceptions import CredentialError
from fusion.fusion import Fusion
from fusion.utils import (
    distribution_to_url,
    get_default_fs,
)

from .conftest import change_dir


def test_pickle_fusion_credentials(tmp_path: Path) -> None:
    creds = FusionCredentials.from_client_id(
        client_id="my_client_id",
        client_secret="my_client_secret",
        resource="my_resource",
        auth_url="my_auth_url",
        proxies={},
        fusion_e2e=None,
    )
    creds.put_bearer_token("some_token", 1234)

    creds_file = tmp_path / "creds.pkl"
    with creds_file.open("wb") as file:
        pickle.dump(creds, file)

    with creds_file.open("rb") as file:
        creds_loaded = pickle.load(file)

    assert isinstance(creds_loaded, FusionCredentials)
    assert creds_loaded.client_id == "my_client_id"
    assert creds_loaded.client_secret == "my_client_secret"
    assert creds_loaded.resource == "my_resource"
    assert creds_loaded.auth_url == "my_auth_url"


def test_from_file_relative_path_walkup_exists(tmp_path: Path, good_json: str) -> None:
    # Create a temporary credentials file
    dir_down_path = Path(tmp_path / "level_1" / "level_2" / "level_3")
    dir_down_path.mkdir(parents=True)
    credentials_file = dir_down_path.parent.parent / "client_credentials.json"
    credentials_file.write_text(good_json)

    with change_dir(dir_down_path):
        # Call the from_file method with a relative path
        credentials = FusionCredentials.from_file(file_path=Path("client_credentials.json"))

        # Verify that the credentials object is created correctly
        assert isinstance(credentials, FusionCredentials)
        assert credentials.client_id
        assert credentials.client_secret


def test_from_file_file_not_found(tmp_path: Path) -> None:
    # Call the from_file method with a non-existent file
    missing_creds_file = tmp_path / "client_credentials.json"
    with pytest.raises(FileNotFoundError):
        FusionCredentials.from_file(file_path=missing_creds_file)


def test_from_file_empty_file(tmp_path: Path) -> None:
    # Create an empty credentials file
    credentials_file = tmp_path / "client_credentials.json"
    credentials_file.touch()

    # Call the from_file method with an empty file
    with pytest.raises(CredentialError):
        FusionCredentials.from_file(file_path=credentials_file)


def test_from_file_invalid_json(tmp_path: Path) -> None:
    # Create a credentials file with invalid JSON
    credentials_file = tmp_path / "client_credentials.json"
    credentials_file.write_text('{"client_id": "my_client_id"}')

    # Call the from_file method with invalid JSON
    with pytest.raises(CredentialError):
        FusionCredentials.from_file(file_path=credentials_file)


class MockResponse:
    def __init__(self, status_code: int = 200, json_data: Optional[dict[str, Any]] = None) -> None:
        self.status_code = status_code
        self.json_data = json_data

    def json(self) -> Optional[dict[str, Any]]:
        return self.json_data


@pytest.mark.skip(reason="Time travel won't work over native code")
def test_refresh_token_data_success(
    fusion_oauth_adapter: FusionOAuthAdapter, requests_mock: requests_mock.Mocker
) -> None:
    exp_win = 180
    init_token = "token123_1"
    next_token = "token123_2"

    snap_t = datetime.now(tz=timezone.utc)
    delta_before_exp = snap_t + timedelta(seconds=60)
    delta_after_exp = snap_t + timedelta(seconds=200)

    with freeze_time(snap_t) as frozen_datetime:
        # Initial auth req
        assert fusion_oauth_adapter.credentials.auth_url
        requests_mock.post(
            fusion_oauth_adapter.credentials.auth_url, json={"access_token": init_token, "expires_in": exp_win}
        )

        token, expiry = fusion_oauth_adapter._refresh_token_data()
        assert token == init_token
        assert int(expiry) == exp_win

        # Time travel to before expiry
        frozen_datetime.move_to(delta_before_exp)
        token, expiry = fusion_oauth_adapter._refresh_token_data()
        assert token == init_token
        assert int(expiry) == exp_win

        # Time travel to after expiry
        frozen_datetime.move_to(delta_after_exp)
        requests_mock.post(
            fusion_oauth_adapter.credentials.auth_url, json={"access_token": next_token, "expires_in": exp_win}
        )
        token, expiry = fusion_oauth_adapter._refresh_token_data()
        assert token == next_token
        assert int(expiry) == exp_win


def test_refresh_token_data_failure(
    fusion_oauth_adapter: FusionOAuthAdapter, requests_mock: requests_mock.Mocker
) -> None:
    assert fusion_oauth_adapter.credentials.auth_url
    requests_mock.post(fusion_oauth_adapter.credentials.auth_url, status_code=500)
    with pytest.raises(requests.exceptions.HTTPError):
        fusion_oauth_adapter._refresh_token_data()


# @pytest.mark.skip(reason="Legacy code")
def test_refresh_fusion_token_data(
    fusion_oauth_adapter: FusionOAuthAdapter,
    requests_mock: requests_mock.Mocker,
    credentials: FusionCredentials,
) -> None:
    fusion_obj = Fusion(credentials=credentials)
    catalog = "my_catalog"
    dataset = "my_dataset"
    datasetseries = "2020-04-04"
    file_format = "csv"
    url = distribution_to_url(fusion_obj.root_url, catalog, dataset, datasetseries, file_format)
    token_auth_url = f"{fusion_obj.root_url}catalogs/csv/datasets/{catalog}/authorize/token"

    # Prep the mock urls
    prep_req = requests.Request("GET", url).prepare()
    requests_mock.get(token_auth_url, json={"access_token": "ds_token123", "expires_in": 180})

    # Set a fake bearer token exp
    fusion_oauth_adapter.credentials.put_bearer_token("token123", 1800)
    fusion_oauth_adapter._refresh_fusion_token_data(prep_req)


def test_refresh_fusion_token_data_refresh(
    fusion_oauth_adapter: FusionOAuthAdapter,
    requests_mock: requests_mock.Mocker,
    credentials: FusionCredentials,
) -> None:
    fusion_obj = Fusion(credentials=credentials)
    catalog = "my_catalog"
    dataset = "my_dataset"
    datasetseries = "2020-04-04"
    file_format = "csv"
    fusion_token_key = catalog + "_" + dataset
    fusion_oauth_adapter.credentials.put_fusion_token(fusion_token_key, "prev_token", 0)

    url = distribution_to_url(fusion_obj.root_url, dataset, datasetseries, file_format, catalog)
    token_auth_url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/authorize/token"

    exp_win = 3600
    assert fusion_oauth_adapter.credentials.auth_url
    requests_mock.post(
        fusion_oauth_adapter.credentials.auth_url, json={"access_token": "token123", "expires_in": exp_win}
    )

    # Prep the mock urls
    prep_req = requests.Request("GET", url).prepare()
    requests_mock.get(prep_req)  # type: ignore
    requests_mock.get(token_auth_url, json={"access_token": "ds_token123", "expires_in": 180})

    fusion_oauth_adapter.send(prep_req)


@pytest.mark.skip(reason="Time travel won't work over native code")
def test_fusion_oauth_adapter_send(
    fusion_oauth_adapter: FusionOAuthAdapter,
    requests_mock: requests_mock.Mocker,
    credentials: FusionCredentials,
) -> None:
    fusion_obj = Fusion(credentials=credentials)
    catalog = "my_catalog"
    dataset = "my_dataset"
    datasetseries = "2020-04-04"
    file_format = "csv"
    fusion_token_key = catalog + "_" + dataset

    url = distribution_to_url(fusion_obj.root_url, dataset, datasetseries, file_format, catalog)
    token_auth_url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/authorize/token"

    exp_win = 3600
    assert fusion_oauth_adapter.credentials.auth_url
    requests_mock.post(
        fusion_oauth_adapter.credentials.auth_url, json={"access_token": "token123", "expires_in": exp_win}
    )

    # Prep the mock urls
    prep_req = requests.Request("GET", url).prepare()
    requests_mock.get(url)

    init_token = "ds_token123_1"
    next_token = "ds_token123_2"

    snap_t = datetime.now(tz=timezone.utc)
    delta_before_exp = snap_t + timedelta(seconds=60)
    delta_after_exp = snap_t + timedelta(seconds=200)

    with freeze_time(snap_t) as frozen_datetime:
        requests_mock.get(token_auth_url, json={"access_token": init_token, "expires_in": 180})
        fusion_oauth_adapter.send(prep_req)
        assert fusion_oauth_adapter.credentials.fusion_token[fusion_token_key].token == init_token

        frozen_datetime.move_to(delta_before_exp)
        fusion_oauth_adapter.send(prep_req)
        assert fusion_oauth_adapter.credentials.fusion_token[fusion_token_key].token == init_token

        requests_mock.get(token_auth_url, json={"access_token": next_token, "expires_in": 180})
        frozen_datetime.move_to(delta_after_exp)
        fusion_oauth_adapter.send(prep_req)
        assert fusion_oauth_adapter.credentials.fusion_token[fusion_token_key].token == next_token


def test_fusion_oauth_adapter_send_header(
    fusion_oauth_adapter: FusionOAuthAdapter,
    requests_mock: requests_mock.Mocker,
    credentials: FusionCredentials,
) -> None:
    fusion_obj = Fusion(credentials=credentials)
    catalog = "my_catalog"
    dataset = "my_dataset"
    datasetseries = "2020-04-04"
    file_format = "csv"
    fusion_oauth_adapter.credentials = credentials
    url = distribution_to_url(fusion_obj.root_url, dataset, datasetseries, file_format, catalog)
    token_auth_url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/authorize/token"

    exp_win = 3600
    assert fusion_oauth_adapter.credentials.auth_url
    requests_mock.post(
        fusion_oauth_adapter.credentials.auth_url, json={"access_token": "token123", "expires_in": exp_win}
    )

    # Prep the mock urls
    prep_req = requests.Request("GET", url).prepare()
    requests_mock.get(url)

    init_token = "ds_token123_1"

    requests_mock.get(token_auth_url, json={"access_token": init_token, "expires_in": 180})

    fusion_oauth_adapter.send(prep_req)
    if prep_req.headers.get("fusion-e2e"):
        assert prep_req.headers.get("fusion-e2e") == credentials.fusion_e2e


@pytest.mark.skip(reason="Legacy code")
def test_fusion_oauth_adapter_send_no_bearer_token_exp(fusion_oauth_adapter: FusionOAuthAdapter) -> None:
    fusion_oauth_adapter.credentials.bearer_token = None
    with pytest.raises(CredentialError):
        fusion_oauth_adapter.send(Mock())


@pytest.fixture()
def local_fsspec_fs() -> Generator[tuple[fsspec.filesystem, str], None, None]:
    with TemporaryDirectory() as tmp_dir, patch("fsspec.filesystem") as mock_fs:
        # Configure the mock to return a LocalFileSystem that points to our temporary directory
        local_fs = fsspec.filesystem("file", auto_mkdir=True)
        mock_fs.return_value = local_fs
        yield local_fs, tmp_dir


def test_default_filesystem() -> None:
    """Test the default filesystem is local file when no env vars are set."""
    with patch.dict(os.environ, {}, clear=True), patch("fsspec.filesystem") as mock_fs:
        mock_fs.return_value = MagicMock()
        fs = get_default_fs()
        mock_fs.assert_called_once_with("file")
        assert isinstance(fs, MagicMock), "Should return a filesystem object."


def test_s3_filesystem() -> None:
    """Test that S3 filesystem is used when S3 env vars are set."""
    env_vars = {
        "FS_PROTOCOL": "s3",
        "S3_ENDPOINT": "s3.example.com",
        "AWS_ACCESS_KEY_ID": "key",
        "AWS_SECRET_ACCESS_KEY": "secret",
    }
    with patch.dict(os.environ, env_vars), patch("fsspec.filesystem") as mock_fs:
        mock_fs.return_value = MagicMock()
        fs = get_default_fs()
        mock_fs.assert_called_once_with(
            "s3", client_kwargs={"endpoint_url": "https://s3.example.com"}, key="key", secret="secret"
        )
        assert isinstance(fs, MagicMock), "Should return a filesystem object."


def test_from_object_with_json_file(tmp_path: Path) -> None:
    credentials_file = tmp_path / "credentials.json"
    credentials = {
        "grant_type": "client_credentials",
        "client_id": "my_client_id",
        "client_secret": "my_client_secret",
        "resource": "my_resource",
        "auth_url": "my_auth_url",
    }

    with Path(credentials_file).open("w") as file:
        json.dump(credentials, file)

    creds = FusionCredentials.from_file(credentials_file)

    assert isinstance(creds, FusionCredentials)
    assert creds.client_id == "my_client_id"
    assert creds.client_secret == "my_client_secret"
    assert creds.resource == "my_resource"
    assert creds.auth_url == "my_auth_url"


@pytest.mark.parametrize(
    ("client_id", "expected", "env_var_value"),
    [
        (None, "12345", "12345"),  # Test fetching from environment variable
        ("abcde", "abcde", "12345"),  # Test returning provided client_id
        (None, None, None),  # Test with no environment variable set
    ],
)
def test_try_get_client_id(monkeypatch: pytest.MonkeyPatch, client_id: str, expected: str, env_var_value: str) -> None:
    if env_var_value is not None:
        monkeypatch.setenv("FUSION_CLIENT_ID", env_var_value)
    else:
        monkeypatch.delenv("FUSION_CLIENT_ID", raising=False)
    assert try_get_client_id(client_id) == expected


@pytest.mark.parametrize(
    ("client_secret", "expected", "env_var_value"),
    [
        (None, "secret123", "secret123"),  # Test fetching from environment variable
        ("mysecret", "mysecret", "secret123"),  # Test returning provided client_secret
        (None, None, None),  # Test with no environment variable set
    ],
)
def test_try_get_client_secret(
    monkeypatch: pytest.MonkeyPatch, client_secret: str, expected: str, env_var_value: str
) -> None:
    if env_var_value is not None:
        monkeypatch.setenv("FUSION_CLIENT_SECRET", env_var_value)
    else:
        monkeypatch.delenv("FUSION_CLIENT_SECRET", raising=False)
    assert try_get_client_secret(client_secret) == expected


def test_client_from_env_vars(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client_id = "my_client_id"
    client_secret = "my_client_secret"
    monkeypatch.setenv("FUSION_CLIENT_ID", client_id)
    monkeypatch.setenv("FUSION_CLIENT_SECRET", client_secret)

    creds_dict = {
        "resource": "my_resource",
        "auth_url": "https://auth_url.com",
    }

    creds_file = tmp_path / "creds.json"
    creds_file.write_text(json.dumps(creds_dict))

    creds = FusionCredentials.from_file(creds_file)

    assert creds.client_id == client_id
    assert creds.client_secret == client_secret
