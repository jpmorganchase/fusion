from typing import Any
import json
import pytest
from pathlib import Path


@pytest.mark.benchmark(group="credentials")
def test_rust_creds(benchmark: Any, example_creds_dict: dict[str, Any], tmp_path: Path) -> None:
    from fusion._fusion import FusionCredentials
    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)

    benchmark(FusionCredentials.from_file, str(credentials_file))


@pytest.mark.benchmark(group="credentials")
def test_py_creds(benchmark: Any, example_creds_dict: dict[str, Any], tmp_path: Path) -> None:
    from fusion.authentication import FusionCredentials

    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)

    benchmark(FusionCredentials.from_file, str(credentials_file))
