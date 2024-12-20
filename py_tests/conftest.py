import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Union
from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest

from fusion._fusion import FusionCredentials
from fusion.authentication import FusionOAuthAdapter
from fusion.fusion import Fusion

PathLike = Union[str, Path]


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--experiments", action="store_true", default=False, help="Run tests marked as experiments")


def pytest_collection_modifyitems(config: Any, items: Any) -> None:
    if not config.getoption("--experiments"):
        skip_experiments = pytest.mark.skip(reason="need --experiments option to run")
        for item in items:
            if "experiments" in item.keywords:
                item.add_marker(skip_experiments)


@contextmanager
def change_dir(destination: PathLike) -> Generator[None, None, None]:
    try:
        # Save the current working directory
        cwd = Path.cwd()
        # Change the working directory
        os.chdir(destination)
        yield
    finally:
        # Change back to the original directory
        os.chdir(cwd)


@pytest.fixture
def example_creds_dict() -> dict[str, Any]:
    return {
        "client_id": "vf3tdjK0jdp7MdY3",
        "client_secret": "vswag2iet7Merdkdwe64YcI9gxbemjMsh5jgimrwpcghsqc2mnj4w4qQffrfhtKz0ba3u48tqJrbp1y",
        "resource": "JPMC:URI:RS-97834-Fusion-PROD",
        "auth_url": "https://authe.mysite.com/as/token.oauth2",
        "proxies": {
            "http": "http://myproxy.com:8080",
            "https": "http://myproxy.com:8081",
        },
    }


@pytest.fixture
def example_creds_dict_from_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    example_client_id = "vf3tdjK0jdp7MdY3"
    example_client_secret = "vswag2iet7Merdkdwe64YcI9gxbemjMsh5jgimrwpcghsqc2mnj4w4qQffrfhtKz0ba3u48tqJrbp1y"
    monkeypatch.setenv("FUSION_CLIENT_ID", example_client_id)
    monkeypatch.setenv("FUSION_CLIENT_SECRET", example_client_secret)

    return {
        "resource": "JPMC:URI:RS-97834-Fusion-PROD",
        "auth_url": "https://authe.mysite.com/as/token.oauth2",
        "proxies": {
            "http": "http://myproxy.com:8080",
            "https": "http://myproxy.com:8081",
        },
    }


@pytest.fixture
def example_creds_dict_https_pxy() -> dict[str, Any]:
    return {
        "client_id": "vf3tdjK0jdp7MdY3",
        "client_secret": "vswag2iet7Merdkdwe64YcI9gxbemjMsh5jgimrwpcghsqc2mnj4w4qQffrfhtKz0ba3u48tqJrbp1y",
        "resource": "JPMC:URI:RS-97834-Fusion-PROD",
        "auth_url": "https://authe.mysite.com/as/token.oauth2",
        "proxies": {
            "https": "http://myproxy.com:8081",
        },
    }


@pytest.fixture
def example_creds_dict_https_pxy_e2e() -> dict[str, Any]:
    return {
        "client_id": "vf3tdjK0jdp7MdY3",
        "client_secret": "vswag2iet7Merdkdwe64YcI9gxbemjMsh5jgimrwpcghsqc2mnj4w4qQffrfhtKz0ba3u48tqJrbp1y",
        "resource": "JPMC:URI:RS-97834-Fusion-PROD",
        "auth_url": "https://authe.mysite.com/as/token.oauth2",
        "proxies": {
            "https": "http://myproxy.com:8081",
        },
        "fusion_e2e": "fusion-e2e-token",
    }


@pytest.fixture
def example_creds_dict_no_pxy(example_creds_dict: dict[str, Any]) -> dict[str, Any]:
    example_creds_dict.pop("proxies")
    return example_creds_dict


@pytest.fixture
def example_creds_dict_empty_pxy(example_creds_dict: dict[str, Any]) -> dict[str, Any]:
    example_creds_dict["proxies"].pop("http")
    example_creds_dict["proxies"].pop("https")
    return example_creds_dict


@pytest.fixture(
    params=[
        "example_creds_dict",
        # "example_creds_dict_from_env", ## Enable this test after fixing ENV VAR reading in Rust
        "example_creds_dict_https_pxy",
        "example_creds_dict_no_pxy",
        "example_creds_dict_empty_pxy",
        "example_creds_dict_https_pxy_e2e",
    ]
)
def credentials_examples(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    """Parameterized fixture to return credentials from different sources."""
    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(request.getfixturevalue(request.param), f)
    return credentials_file


@pytest.fixture
def good_json() -> str:
    return """{
        "client_id": "vf3tdjK0jdp7MdY3",
        "client_secret": "vswag2iet7Merdkdwe64YcI9gxbemjMsh5jgimrwpcghsqc2mnj4w4qQffrfhtKz0ba3u48tqJrbp1y",
        "resource": "JPMC:URI:RS-97834-Fusion-PROD",
        "auth_url": "https://authe.mysite.com/as/token.oauth2",
        "proxies": {
            "http": "http://myproxy.com:8080",
            "https": "http://myproxy.com:8081"
            }
        }"""


@pytest.fixture
def credentials(example_creds_dict: dict[str, Any], tmp_path: Path) -> FusionCredentials:
    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)
    creds.put_bearer_token("my_token", 1800)
    return creds


@pytest.fixture
def fusion_oauth_adapter(credentials: FusionCredentials) -> FusionOAuthAdapter:
    return FusionOAuthAdapter(credentials)


@pytest.fixture
def fusion_obj(credentials: FusionCredentials) -> Fusion:
    fusion = Fusion(credentials=credentials)
    return fusion


@pytest.fixture
def data_table() -> pl.DataFrame:
    return pl.DataFrame(
        {"col_1": range(10), "col_2": [str(x) for x in range(10)], "col_3": [x / 3.14159 for x in range(10)]}
    )


@pytest.fixture
def data_table_as_csv(data_table: pl.DataFrame) -> str:
    return data_table.write_csv(None)


@pytest.fixture
def data_table_as_json(data_table: pl.DataFrame) -> str:
    return data_table.write_json(None)


@pytest.fixture
def mock_product_pd_read_csv() -> Generator[pd.DataFrame, Any, None]:
    """Mock the pd.read_csv function."""
    product_df = pd.DataFrame(
        {
            "title": "Test Product",
            "identifier": "TEST_PRODUCT",
        },
        index=[0],
    )
    with patch("fusion.fusion.pd.read_csv", return_value=product_df) as mock:
        yield mock


@pytest.fixture
def mock_dataset_pd_read_csv() -> Generator[pd.DataFrame, Any, None]:
    """Mock the pd.read_csv function."""
    dataset_df = pd.DataFrame(
        {"title": "Test Dataset", "identifier": "TEST_DATASET", "category": "Test", "product": "TEST_PRODUCT"},
        index=[0],
    )
    with patch("fusion.fusion.pd.read_csv", return_value=dataset_df) as mock:
        yield mock
