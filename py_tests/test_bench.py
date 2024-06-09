
import pytest


@pytest.mark.benchmark(group="credentials")
def test_rust_creds(benchmark, tmp_creds_file):
    from fusion._fusion import FusionCredentials

    benchmark(FusionCredentials.from_file, str(tmp_creds_file))


@pytest.mark.benchmark(group="credentials")
def test_py_creds(benchmark, tmp_creds_file):
    from fusion.authentication import FusionCredentials

    benchmark(FusionCredentials.from_file, str(tmp_creds_file))
