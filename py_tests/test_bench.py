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

    benchmark(FusionCredentials.from_file, credentials_file)


@pytest.mark.benchmark(group="credentials")
def test_py_creds(benchmark: Any, example_creds_dict: dict[str, Any], tmp_path: Path) -> None:
    from fusion._legacy.authentication import FusionCredentials

    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)

    benchmark(FusionCredentials.from_file, credentials_file)


@pytest.mark.benchmark(group="dummy")
def test_rust_ok(benchmark: Any) -> None:
    from fusion._fusion import rust_ok

    benchmark(rust_ok)


def py_rust_ok_equiv() -> bool:
    return True


@pytest.mark.benchmark(group="dummy")
def test_py_rust_ok(benchmark: Any) -> None:
    benchmark(py_rust_ok_equiv)
