from typing import Any

import pytest

from fusion._fusion import FusionCredentials


@pytest.mark.benchmark(group="credentials")
def test_rust_creds(benchmark: Any, creds_dict: "FusionCredentials") -> None:
    from fusion._fusion import FusionCredentials

    benchmark(FusionCredentials.from_file, str(creds_dict))


@pytest.mark.benchmark(group="credentials")
def test_py_creds(benchmark: Any, creds_dict: "FusionCredentials") -> None:
    from fusion.authentication import FusionCredentials

    benchmark(FusionCredentials.from_file, str(creds_dict))
