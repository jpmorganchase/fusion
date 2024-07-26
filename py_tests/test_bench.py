from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.mark.benchmark(group="credentials")
def test_rust_creds(benchmark: Any, example_creds_dict: dict[str, Any], tmp_path: Path) -> None:
    from fusion._fusion import FusionCredentials

    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)

    benchmark.pedantic(
        FusionCredentials.from_file,
        args=(credentials_file,),
        iterations=10,
        rounds=500,
    )


def test_rust_creds_plain(benchmark: Any, example_creds_dict: dict[str, Any]) -> None:
    from fusion._fusion import FusionCredentials

    benchmark.pedantic(
        FusionCredentials.__init__,
        args=(
            example_creds_dict["client_id"],
            example_creds_dict["client_secret"],
            example_creds_dict["resource"],
            example_creds_dict["auth_url"],
            example_creds_dict["proxies"],
        ),
        iterations=10,
        rounds=500,
    )


@pytest.mark.benchmark(group="credentials")
def test_py_creds(benchmark: Any, example_creds_dict: dict[str, Any], tmp_path: Path) -> None:
    from fusion._legacy.authentication import FusionCredentials

    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)

    benchmark.pedantic(
        FusionCredentials.from_file,
        args=(credentials_file,),
        iterations=10,
        rounds=500,
    )


@pytest.mark.benchmark(group="dummy")
def test_rust_ok(benchmark: Any) -> None:
    from fusion._fusion import rust_ok

    benchmark.pedantic(rust_ok, iterations=200, rounds=10_000)


def py_rust_ok_equiv() -> bool:
    return True


@pytest.mark.benchmark(group="dummy")
def test_py_rust_ok(benchmark: Any) -> None:
    benchmark.pedantic(py_rust_ok_equiv, iterations=200, rounds=10_000)
