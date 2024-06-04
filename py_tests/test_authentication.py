import json
import os
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

from fusion.authentication import (
    FusionAiohttpSession,
    FusionCredentials,
    FusionOAuthAdapter,
    get_default_fs,
    try_get_client_id,
    try_get_client_secret,
)
from fusion.exceptions import CredentialError
from fusion.fusion import Fusion
from fusion.utils import (
    distribution_to_url,
)

from .conftest import change_dir


def test_creds_from_dict() -> None:  # noqa: PLR0915
    credentials = {
        "grant_type": "client_credentials",
        "client_id": "my_client_id",
        "client_secret": "my_client_secret",
        "resource": "my_resource",
        "auth_url": "my_auth_url",
    }

    creds = FusionCredentials.from_dict(credentials)

    assert creds.client_id == "my_client_id"
    assert creds.client_secret == "my_client_secret"
    assert creds.username is None
    assert creds.password is None
    assert creds.bearer_token is None
    assert creds.is_bearer_token_expirable is True
    assert creds.resource == "my_resource"
    assert creds.auth_url == "my_auth_url"
    assert not creds.proxies
    assert creds.grant_type == "client_credentials"

    credentials = {
        "grant_type": "bearer",
        "bearer_token": "my_bearer_token",
        "bearer_token_expiry": "2022-01-01",
        "bearer_token_expirable": "true",
    }

    creds = FusionCredentials.from_dict(credentials)

    assert creds.client_id is None
    assert creds.client_secret is None
    assert creds.username is None
    assert creds.password is None
    assert creds.bearer_token == "my_bearer_token"
    assert creds.is_bearer_token_expirable is True
    assert creds.resource is None
    assert creds.auth_url is None
    assert not creds.proxies
    assert creds.grant_type == "bearer"

    credentials = {
        "grant_type": "password",
        "client_id": "my_client_id",
        "username": "my_username",
        "password": "my_password",
        "resource": "my_resource",
        "auth_url": "my_auth_url",
    }

    creds = FusionCredentials.from_dict(credentials)

    assert creds.client_id == "my_client_id"
    assert creds.client_secret is None
    assert creds.username == "my_username"
    assert creds.password == "my_password"
    assert creds.bearer_token is None
    assert creds.is_bearer_token_expirable is True
    assert creds.resource == "my_resource"
    assert creds.auth_url == "my_auth_url"
    assert not creds.proxies
    assert creds.grant_type == "password"

    credentials = {
        "grant_type": "password",
        "client_id": "my_client_id",
        "username": "my_username",
        "password": "my_password",
        "resource": "my_resource",
        "auth_url": "my_auth_url",
        "fusion_e2e": "fusion-e2e",
    }

    creds = FusionCredentials.from_dict(credentials)

    assert creds.client_id == "my_client_id"
    assert creds.client_secret is None
    assert creds.username == "my_username"
    assert creds.password == "my_password"
    assert creds.bearer_token is None
    assert creds.is_bearer_token_expirable is True
    assert creds.resource == "my_resource"
    assert creds.auth_url == "my_auth_url"
    assert not creds.proxies
    assert creds.grant_type == "password"
    assert creds.fusion_e2e == "fusion-e2e"

    credentials = {"grant_type": "unknown"}

    with pytest.raises(CredentialError):
        FusionCredentials.from_dict(credentials)


def test_add_proxies(tmp_path: Path) -> None:
    credentials_file = str(tmp_path / "client_credentials.json")
    client_id = "my_client_id"
    client_secret = "my_client_secret"
    resource = "my_resource"
    auth_url = "my_auth_url"
    proxies = {
        "http": "http://proxy.example.com",
        "https": "https://proxy.example.com",
    }

    # Call the generate_credentials_file function
    credentials = FusionCredentials.generate_credentials_file(
        credentials_file=credentials_file,
        client_id=client_id,
        client_secret=client_secret,
        resource=resource,
        auth_url=auth_url,
        proxies=proxies,
    )

    # Verify that the credentials file was created
    assert Path(credentials_file).exists()

    # Define the input parameters
    http_proxy = "http://proxy.example.com"
    https_proxy = "https://proxy.example.com"

    # Call the add_proxies function
    credentials.add_proxies(http_proxy, https_proxy, credentials_file)

    # Verify that the proxies were added to the credentials object
    assert credentials.proxies["http"] == http_proxy
    assert credentials.proxies["https"] == https_proxy

    credentials.add_proxies(http_proxy, None, credentials_file)


def test_generate_credentials_file(tmp_path: Path) -> None:
    # Define the input parameters
    credentials_file = str(tmp_path / "client_credentials.json")
    client_id = "my_client_id"
    client_secret = "my_client_secret"
    resource = "my_resource"
    auth_url = "my_auth_url"
    proxies = {
        "http": "http://proxy.example.com",
        "https": "https://proxy.example.com",
    }

    # Call the generate_credentials_file function
    credentials = FusionCredentials.generate_credentials_file(
        credentials_file=credentials_file,
        client_id=client_id,
        client_secret=client_secret,
        resource=resource,
        auth_url=auth_url,
        proxies=proxies,
    )

    # Verify that the credentials file was created
    assert Path(credentials_file).exists()

    # Verify the content of the credentials file
    with Path(credentials_file).open() as file:
        data = json.load(file)
        assert data["client_id"] == client_id
        assert data["client_secret"] == client_secret
        assert data["resource"] == resource
        assert data["auth_url"] == auth_url
        assert data["proxies"]["http"] == proxies["http"]
        assert data["proxies"]["https"] == proxies["https"]

    # Verify the returned credentials object
    assert isinstance(credentials, FusionCredentials)
    assert credentials.client_id == client_id
    assert credentials.client_secret == client_secret
    assert credentials.resource == resource
    assert credentials.auth_url == auth_url
    assert credentials.proxies["http"] == proxies["http"]
    assert credentials.proxies["https"] == proxies["https"]


def test_generate_credentials_w_json_pxy_file(tmp_path: Path) -> None:
    # Define the input parameters
    credentials_file = str(tmp_path / "client_credentials.json")
    client_id = "my_client_id"
    client_secret = "my_client_secret"
    resource = "my_resource"
    auth_url = "my_auth_url"
    proxies = """{
        "http": "http://proxy.example.com",
        "https": "https://proxy.example.com"
    }"""
    proxies_d = {"http": "http://proxy.example.com", "https": "https://proxy.example.com"}

    # Call the generate_credentials_file function
    credentials = FusionCredentials.generate_credentials_file(
        credentials_file=credentials_file,
        client_id=client_id,
        client_secret=client_secret,
        resource=resource,
        auth_url=auth_url,
        proxies=proxies,
    )

    # Verify that the credentials file was created
    assert Path(credentials_file).exists()

    # Verify the content of the credentials file
    with Path(credentials_file).open() as file:
        data = json.load(file)
        assert data["client_id"] == client_id
        assert data["client_secret"] == client_secret
        assert data["resource"] == resource
        assert data["auth_url"] == auth_url
        assert data["proxies"]["http"] == proxies_d["http"]
        assert data["proxies"]["https"] == proxies_d["https"]

    # Verify the returned credentials object
    assert isinstance(credentials, FusionCredentials)
    assert credentials.client_id == client_id
    assert credentials.client_secret == client_secret
    assert credentials.resource == resource
    assert credentials.auth_url == auth_url
    assert credentials.proxies["http"] == proxies_d["http"]
    assert credentials.proxies["https"] == proxies_d["https"]


def test_generate_credentials_w_bad_pxy_file(tmp_path: Path) -> None:
    # Define the input parameters
    credentials_file = str(tmp_path / "client_credentials.json")
    client_id = "my_client_id"
    client_secret = "my_client_secret"
    resource = "my_resource"
    auth_url = "my_auth_url"
    bad_proxies = {
        "bad_http": "http://proxy.example.com",
        "evil_https": "https://proxy.example.com",
    }

    with pytest.raises(CredentialError):
        # Call the generate_credentials_file function
        _ = FusionCredentials.generate_credentials_file(
            credentials_file=credentials_file,
            client_id=client_id,
            client_secret=client_secret,
            resource=resource,
            auth_url=auth_url,
            proxies=3.14159,  # type: ignore
        )
    with pytest.raises(CredentialError):
        _ = FusionCredentials.generate_credentials_file(
            credentials_file=credentials_file,
            client_id=client_id,
            client_secret=client_secret,
            resource=resource,
            auth_url=auth_url,
            proxies=bad_proxies,
        )


def test_generate_credentials_file_missing_client_id(tmp_path: Path) -> None:
    # Define the input parameters with missing client_id
    credentials_file = str(tmp_path / "client_credentials.json")
    client_secret = "my_client_secret"
    resource = "my_resource"
    auth_url = "my_auth_url"
    proxies = {
        "http": "http://proxy.example.com",
        "https": "https://proxy.example.com",
    }

    # Call the generate_credentials_file function and expect a CredentialError
    with pytest.raises(CredentialError):
        FusionCredentials.generate_credentials_file(
            credentials_file=credentials_file,
            client_secret=client_secret,
            resource=resource,
            auth_url=auth_url,
            proxies=proxies,
        )


def test_generate_credentials_file_missing_client_secret(tmp_path: Path) -> None:
    # Define the input parameters with missing client_secret
    credentials_file = str(tmp_path / "client_credentials.json")
    client_id = "my_client_id"
    resource = "my_resource"
    auth_url = "my_auth_url"
    proxies = {
        "http": "http://proxy.example.com",
        "https": "https://proxy.example.com",
    }

    # Call the generate_credentials_file function and expect a CredentialError
    with pytest.raises(CredentialError):
        FusionCredentials.generate_credentials_file(
            credentials_file=credentials_file,
            client_id=client_id,
            resource=resource,
            auth_url=auth_url,
            proxies=proxies,
        )


def test_generate_credentials_file_invalid_proxies(tmp_path: Path) -> None:
    # Define the input parameters with invalid proxies
    credentials_file = str(tmp_path / "client_credentials.json")
    client_id = "my_client_id"
    client_secret = "my_client_secret"
    resource = "my_resource"
    auth_url = "my_auth_url"
    proxies = "invalid_proxies"

    # Call the generate_credentials_file function and expect a CredentialError
    with pytest.raises(CredentialError):
        FusionCredentials.generate_credentials_file(
            credentials_file=credentials_file,
            client_id=client_id,
            client_secret=client_secret,
            resource=resource,
            auth_url=auth_url,
            proxies=proxies,
        )


def test_generate_credentials_file_existing_file(tmp_path: Path) -> None:
    # Create an existing credentials file
    credentials_file = tmp_path / "client_credentials.json"
    credentials_file.write_text("existing file")

    # Define the input parameters
    client_id = "my_client_id"
    client_secret = "my_client_secret"
    resource = "my_resource"
    auth_url = "my_auth_url"
    proxies = {
        "http": "http://proxy.example.com",
        "https": "https://proxy.example.com",
    }

    FusionCredentials.generate_credentials_file(
        credentials_file=str(credentials_file),
        client_id=client_id,
        client_secret=client_secret,
        resource=resource,
        auth_url=auth_url,
        proxies=proxies,
    )
    assert credentials_file.exists()


def test_from_file_absolute_path_exists(tmp_path: Path, good_json: str) -> None:
    # Create a temporary credentials file
    credentials_file = tmp_path / "client_credentials.json"
    credentials_file.write_text(good_json)

    # Call the from_file method with an absolute path
    credentials = FusionCredentials.from_file(file_path=str(credentials_file))

    # Verify that the credentials object is created correctly
    assert isinstance(credentials, FusionCredentials)
    assert credentials.client_id
    assert credentials.client_secret


def test_from_file_relative_path_exists(tmp_path: Path, good_json: str) -> None:
    # Create a temporary credentials file
    credentials_file = tmp_path / "client_credentials.json"
    credentials_file.write_text(good_json)

    with change_dir(tmp_path):
        # Call the from_file method with a relative path
        credentials = FusionCredentials.from_file(file_path="client_credentials.json")

        # Verify that the credentials object is created correctly
        assert isinstance(credentials, FusionCredentials)
        assert credentials.client_id
        assert credentials.client_secret


def test_from_file_relative_path_walkup_exists(tmp_path: Path, good_json: str) -> None:
    # Create a temporary credentials file
    dir_down_path = Path(tmp_path / "level_1" / "level_2" / "level_3")
    dir_down_path.mkdir(parents=True)
    credentials_file = dir_down_path.parent.parent / "client_credentials.json"
    credentials_file.write_text(good_json)

    with change_dir(dir_down_path):
        # Call the from_file method with a relative path
        credentials = FusionCredentials.from_file(file_path="client_credentials.json")

        # Verify that the credentials object is created correctly
        assert isinstance(credentials, FusionCredentials)
        assert credentials.client_id
        assert credentials.client_secret


def test_credentials_from_object() -> None:
    # Create a credentials object
    credentials = FusionCredentials(
        client_id="my_client_id",
        client_secret="my_client_secret",
        resource="my_resource",
        auth_url="my_auth_url",
    )

    # Call the from_object method with the credentials object
    new_credentials = FusionCredentials.from_object(credentials)

    # Verify that the new credentials object is the same as the original
    assert new_credentials.client_id == credentials.client_id
    assert new_credentials.client_secret == credentials.client_secret
    assert new_credentials.resource == credentials.resource
    assert new_credentials.auth_url == credentials.auth_url


def test_from_file_file_not_found(tmp_path: Path) -> None:
    # Call the from_file method with a non-existent file
    missing_creds_file = tmp_path / "client_credentials.json"
    with pytest.raises(FileNotFoundError):
        FusionCredentials.from_file(file_path=str(missing_creds_file))


def test_from_file_empty_file(tmp_path: Path) -> None:
    # Create an empty credentials file
    credentials_file = tmp_path / "client_credentials.json"
    credentials_file.touch()

    # Call the from_file method with an empty file
    with pytest.raises(OSError, match="is an empty file, make sure to add your credentials to it."):
        FusionCredentials.from_file(file_path=str(credentials_file))


def test_from_file_invalid_json(tmp_path: Path) -> None:
    # Create a credentials file with invalid JSON
    credentials_file = tmp_path / "client_credentials.json"
    credentials_file.write_text('{"client_id": "my_client_id"}')

    # Call the from_file method with invalid JSON
    with pytest.raises(CredentialError):
        FusionCredentials.from_file(file_path=str(credentials_file))


class MockResponse:
    def __init__(self, status_code: int = 200, json_data: Optional[dict[str, Any]] = None) -> None:
        self.status_code = status_code
        self.json_data = json_data

    def json(self) -> Optional[dict[str, Any]]:
        return self.json_data


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


def test_refresh_fusion_token_data(
    fusion_oauth_adapter: FusionOAuthAdapter,
    fusion_oauth_adapter_from_obj: FusionOAuthAdapter,
    requests_mock: requests_mock.Mocker,
    creds_dict: dict[str, Any],
) -> None:
    creds = FusionCredentials.from_dict(creds_dict)
    fusion_obj = Fusion(credentials=creds)
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
    fusion_oauth_adapter.credentials.bearer_token_expiry = datetime.now(tz=timezone.utc) + timedelta(seconds=1800)
    fusion_oauth_adapter._refresh_fusion_token_data(prep_req)

    fusion_oauth_adapter_from_obj.credentials.bearer_token_expiry = datetime.now(tz=timezone.utc) + timedelta(
        seconds=1800
    )
    fusion_oauth_adapter_from_obj._refresh_fusion_token_data(prep_req)


def test_refresh_fusion_token_data_refresh(
    fusion_oauth_adapter: FusionOAuthAdapter,
    requests_mock: requests_mock.Mocker,
    creds_dict: dict[str, Any],
) -> None:
    creds = FusionCredentials.from_dict(creds_dict)
    fusion_obj = Fusion(credentials=creds)
    catalog = "my_catalog"
    dataset = "my_dataset"
    datasetseries = "2020-04-04"
    file_format = "csv"
    fusion_token_key = catalog + "_" + dataset
    fusion_oauth_adapter.fusion_token_dict[fusion_token_key] = "prev_token"
    fusion_oauth_adapter.fusion_token_expiry_dict[fusion_token_key] = datetime.now(tz=timezone.utc)
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


def test_fusion_oauth_adapter_send(
    fusion_oauth_adapter: FusionOAuthAdapter,
    requests_mock: requests_mock.Mocker,
    creds_dict: dict[str, Any],
) -> None:
    creds = FusionCredentials.from_dict(creds_dict)
    fusion_obj = Fusion(credentials=creds)
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
        assert fusion_oauth_adapter.fusion_token_dict[fusion_token_key] == init_token

        frozen_datetime.move_to(delta_before_exp)
        fusion_oauth_adapter.send(prep_req)
        assert fusion_oauth_adapter.fusion_token_dict[fusion_token_key] == init_token

        requests_mock.get(token_auth_url, json={"access_token": next_token, "expires_in": 180})
        frozen_datetime.move_to(delta_after_exp)
        fusion_oauth_adapter.send(prep_req)
        assert fusion_oauth_adapter.fusion_token_dict[fusion_token_key] == next_token


def test_fusion_oauth_adapter_send_header(
    fusion_oauth_adapter: FusionOAuthAdapter,
    requests_mock: requests_mock.Mocker,
    creds_dict: dict[str, Any],
) -> None:
    creds = FusionCredentials.from_dict(creds_dict)
    fusion_obj = Fusion(credentials=creds)
    catalog = "my_catalog"
    dataset = "my_dataset"
    datasetseries = "2020-04-04"
    file_format = "csv"
    fusion_oauth_adapter.credentials = creds
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
        assert prep_req.headers.get("fusion-e2e") == creds_dict.get("fusion_e2e")


def test_fusion_oauth_adapter_send_no_bearer_token_exp(fusion_oauth_adapter: FusionOAuthAdapter) -> None:
    fusion_oauth_adapter.credentials.bearer_token_expiry = None
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


def test_from_object_with_dict() -> None:
    credentials = {
        "grant_type": "client_credentials",
        "client_id": "my_client_id",
        "client_secret": "my_client_secret",
        "resource": "my_resource",
        "auth_url": "my_auth_url",
    }

    creds = FusionCredentials.from_object(credentials)

    assert isinstance(creds, FusionCredentials)
    assert creds.client_id == "my_client_id"
    assert creds.client_secret == "my_client_secret"
    assert creds.resource == "my_resource"
    assert creds.auth_url == "my_auth_url"


def test_from_object_with_json_string() -> None:
    credentials = """{
        "grant_type": "client_credentials",
        "client_id": "my_client_id",
        "client_secret": "my_client_secret",
        "resource": "my_resource",
        "auth_url": "my_auth_url"
    }"""

    creds = FusionCredentials.from_object(credentials)

    assert isinstance(creds, FusionCredentials)
    assert creds.client_id == "my_client_id"
    assert creds.client_secret == "my_client_secret"
    assert creds.resource == "my_resource"
    assert creds.auth_url == "my_auth_url"


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

    creds = FusionCredentials.from_object(str(credentials_file))

    assert isinstance(creds, FusionCredentials)
    assert creds.client_id == "my_client_id"
    assert creds.client_secret == "my_client_secret"
    assert creds.resource == "my_resource"
    assert creds.auth_url == "my_auth_url"


def test_from_object_with_invalid_credentials() -> None:
    credentials = 12345

    with pytest.raises(CredentialError):
        FusionCredentials.from_object(credentials)  # type: ignore


def test_async_session() -> None:
    session = FusionAiohttpSession()
    assert session

    session.post_init()
    assert session


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
