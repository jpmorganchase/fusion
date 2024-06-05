import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Union

import pytest

from fusion.authentication import FusionCredentials, FusionOAuthAdapter
from fusion.fusion import Fusion

PathLike = Union[str, Path]


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


@pytest.fixture()
def example_client_id() -> str:
    return "vf3tdjK0jdp7MdY3"


@pytest.fixture()
def example_client_secret() -> str:
    return "vswag2iet7Merdkdwe64YcI9gxbemjMsh5jgimrwpcghsqc2mnj4w4qQffrfhtKz0ba3u48tqJrbp1y"


@pytest.fixture()
def example_http_proxy() -> str:
    return "http://myproxy.com:8080"


@pytest.fixture()
def example_https_proxy() -> str:
    return "https://myproxy.com:8080"


@pytest.fixture()
def example_proxy_http_dict(example_http_proxy: str) -> dict[str, str]:
    return {"http": example_http_proxy}


@pytest.fixture()
def example_proxy_https_dict(example_https_proxy: str) -> dict[str, str]:
    return {"https": example_https_proxy}


@pytest.fixture()
def example_proxy_both_dict(example_http_proxy: str, example_https_proxy: str) -> dict[str, str]:
    return {"http": example_http_proxy, "https": example_https_proxy}


@pytest.fixture()
def example_proxy_both_alt_dict(example_http_proxy: str, example_https_proxy: str) -> dict[str, str]:
    return {"http_proxy": example_http_proxy, "https_proxy": example_https_proxy}


@pytest.fixture()
def example_proxy_str1(example_http_proxy: str) -> str:
    return example_http_proxy


@pytest.fixture()
def example_proxy_str_bad() -> str:
    return "not_a_proxy"


@pytest.fixture()
def example_creds_dict(example_client_id: str, example_client_secret: str) -> dict[str, Any]:
    # Mocked creds info
    return {
        "client_id": example_client_id,
        "client_secret": example_client_secret,
        "resource": "JPMC:URI:RS-97834-Fusion-PROD",
        "auth_url": "https://authe.mysite.com/as/token.oauth2",
        "proxies": {
            "http": "http://myproxy.com:8080",
            "https": "http://myproxy.com:8081",
        },
    }


@pytest.fixture()
def example_creds_dict_from_env(
    monkeypatch: pytest.MonkeyPatch, example_client_id: str, example_client_secret: str
) -> dict[str, Any]:
    # Mocked creds info

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


@pytest.fixture()
def example_creds_dict_https_pxy(example_client_id: str, example_client_secret: str) -> dict[str, Any]:
    # Mocked creds info
    return {
        "client_id": example_client_id,
        "client_secret": example_client_secret,
        "resource": "JPMC:URI:RS-97834-Fusion-PROD",
        "auth_url": "https://authe.mysite.com/as/token.oauth2",
        "proxies": {
            "https": "http://myproxy.com:8081",
        },
    }


@pytest.fixture()
def example_creds_dict_https_pxy_e2e(example_client_id: str, example_client_secret: str) -> dict[str, Any]:
    # Mocked creds info
    return {
        "client_id": example_client_id,
        "client_secret": example_client_secret,
        "resource": "JPMC:URI:RS-97834-Fusion-PROD",
        "auth_url": "https://authe.mysite.com/as/token.oauth2",
        "proxies": {
            "https": "http://myproxy.com:8081",
        },
        "fusion_e2e": "fusion-e2e-token",
    }


@pytest.fixture()
def example_creds_dict_no_pxy(example_creds_dict: dict[str, Any]) -> dict[str, Any]:
    example_creds_dict.pop("proxies")
    return example_creds_dict


@pytest.fixture()
def example_creds_dict_empty_pxy(example_creds_dict: dict[str, Any]) -> dict[str, Any]:
    example_creds_dict["proxies"].pop("http")
    example_creds_dict["proxies"].pop("https")
    return example_creds_dict


@pytest.fixture(
    params=[
        "example_creds_dict",
        "example_creds_dict_from_env",
        "example_creds_dict_https_pxy",
        "example_creds_dict_no_pxy",
        "example_creds_dict_empty_pxy",
        "example_creds_dict_https_pxy_e2e",
    ]
)
def creds_dict(request: pytest.FixtureRequest) -> Any:
    """Parameterized fixture to return credentials from different sources."""
    return request.getfixturevalue(request.param)


@pytest.fixture()
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


@pytest.fixture()
def bad_json1() -> str:
    return """{
        "client_id" "vf3tdjK0jdp7MdY3",
        "client_secret": "vswag2iet7Merdkdwe64YcI9gxbemjMsh5jgimrwpcghsqc2mnj4w4qQffrfhtKz0ba3u48tqJrbp1y",
        "resource": "JPMC:URI:RS-97834-Fusion-PROD",
        "auth_url": "https://authe.mysite.com/as/token.oauth2",
        "proxies": {
            "http": "http://myproxy.com:8080",
            "https": "http://myproxy.com:8081",
        }"""


@pytest.fixture()
def bad_json2() -> str:
    return """{
        "client_id", vf3tdjK0jdp7MdY3,
        "client_secret": "vswag2iet7Merdkdwe64YcI9gxbemjMsh5jgimrwpcghsqc2mnj4w4qQffrfhtKz0ba3u48tqJrbp1y",
        "resource": "JPMC:URI:RS-97834-Fusion-PROD",
        "auth_url": "https://authe.mysite.com/as/token.oauth2",
        "proxies": {
            "http": "http://myproxy.com:8080",
            "https": "http://myproxy.com:8081",
        }"""


@pytest.fixture()
def bad_json3() -> str:
    return """{
        "client_id", "vf3tdjK0jdp7MdY3",
        "client_secret": "vswag2iet7Merdkdwe64YcI9gxbemjMsh5jgimrwpcghsqc2mnj4w4qQffrfhtKz0ba3u48tqJrbp1y",
        "resource": "JPMC:URI:RS-97834-Fusion-PROD",
        "auth_url": "https://authe.mysite.com/as/token.oauth2",
        "proxies":
            "http": "http://myproxy.com:8080",
            "https": "http://myproxy.com:8081",
        }"""


@pytest.fixture()
def credentials(example_creds_dict: dict[str, Any]) -> FusionCredentials:
    return FusionCredentials.from_dict(example_creds_dict)


@pytest.fixture()
def fusion_oauth_adapter(credentials: FusionCredentials) -> FusionOAuthAdapter:
    return FusionOAuthAdapter(credentials)


@pytest.fixture()
def fusion_oauth_adapter_from_obj(example_creds_dict: dict[str, Any]) -> FusionOAuthAdapter:
    proxies = {
        "http": "http://myproxy.com:8080",
        "https": "http://myproxy.com:8081",
    }
    return FusionOAuthAdapter(example_creds_dict, auth_retries=5, proxies=proxies)


@pytest.fixture()
def fusion_obj(example_creds_dict: dict[str, Any]) -> Fusion:
    creds = FusionCredentials.from_dict(example_creds_dict)
    fusion = Fusion(credentials=creds)
    return fusion
