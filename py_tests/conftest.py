import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Union

import pytest

from fusion._fusion import FusionCredentials
from fusion.authentication import FusionOAuthAdapter
from fusion.fusion import Fusion

from .bench_rep_gen import generate_benchmark_html

PathLike = Union[str, Path]


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--benchmark", action="store_true", default=False, help="Run benchmark tests")


def pytest_collection_modifyitems(config: Any, items: Any) -> None:
    if config.getoption("--benchmark"):
        for item in items:
            if "benchmark" not in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Only benchmark tests are selected"))
        return
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="to be run with --benchmark option"))


@pytest.hookimpl()
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: ARG001
    if session.config.getoption("--benchmark"):
        json_file = sorted(Path().glob(".benchmarks/**/*.json"))[-1]
        html_file = Path(".reports/py/py_bench.html")
        generate_benchmark_html(json_file, html_file)


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


@pytest.fixture()
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


@pytest.fixture()
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


@pytest.fixture()
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
        # "example_creds_dict_from_env", ## Enable this test after fixing ENV VAR reading in Rust
        "example_creds_dict_https_pxy",
        "example_creds_dict_no_pxy",
        "example_creds_dict_empty_pxy",
        "example_creds_dict_https_pxy_e2e",
    ]
)
def credentials_examples(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    """Parameterized fixture to return credentials from different sources."""
    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(request.getfixturevalue(request.param), f)
    return credentials_file


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
def credentials(example_creds_dict: dict[str, Any], tmp_path: Path) -> FusionCredentials:
    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)
    creds.put_bearer_token("my_token", 1800)
    return creds


@pytest.fixture()
def fusion_oauth_adapter(credentials: FusionCredentials) -> FusionOAuthAdapter:
    return FusionOAuthAdapter(credentials)


@pytest.fixture()
def fusion_obj(credentials: FusionCredentials) -> Fusion:
    fusion = Fusion(credentials=credentials)
    return fusion
