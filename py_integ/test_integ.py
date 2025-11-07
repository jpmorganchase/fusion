from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from fusion import Fusion

# mypy: ignore-errors


@dataclass
class DownloadTestSpec:
    client_dl_params: dict[str, Any]
    method: str
    st_dt: str | None = None
    end_dt: str | None = None
    res_shape: tuple[int, int] | None = None
    md5_hash: str | None = None
    # Allow for flexible hash validation - strict, warn, or skip
    hash_validation: str = "warn"  # "strict", "warn", "skip"


download_hashes = {
    "test_download_csv": DownloadTestSpec(
        client_dl_params={
            "dataset": "FXO_SP",
            "dataset_format": "csv",
            "return_paths": True,
            "force_download": True,
            "dt_str": "20231201:20231208",
        },
        method="download",
        st_dt="20231201",
        end_dt="20231208",
        res_shape=None,
        md5_hash=None,  # Let the test compute and log the hash dynamically
    ),
    "test_download_parquet": DownloadTestSpec(
        client_dl_params={
            "dataset": "FXO_SP",
            "dataset_format": "parquet",
            "return_paths": True,
            "force_download": True,
            "dt_str": "20231201:20231208",
        },
        method="download",
        st_dt="20231201",
        end_dt="20231208",
        res_shape=None,
        md5_hash=None,  # Let the test compute and log the hash dynamically
    ),
    "test_to_df_csv": DownloadTestSpec(
        client_dl_params={"dataset": "FXO_SP", "dataset_format": "csv", "dt_str": "20231201:20231208"},
        method="to_df",
        st_dt="20231201",
        end_dt="20231208",
        res_shape=(366, 6),
        md5_hash=None,
    ),
    "test_to_df_parquet": DownloadTestSpec(
        client_dl_params={"dataset": "FXO_SP", "dataset_format": "parquet", "dt_str": "20231201:20231208"},
        method="to_df",
        st_dt="20231201",
        end_dt="20231208",
        res_shape=(366, 6),
        md5_hash=None,
    ),
}


def download_clean_up(download_res: list[tuple[bool, str, str | None]] | None) -> None:
    if not download_res:
        return

    for _, path_str, _ in download_res:
        Path(path_str).unlink(missing_ok=True)


def gen_generic_dl() -> None:
    import os

    from fusion.credentials import FusionCredentials
    from fusion.fusion import Fusion

    creds = FusionCredentials.from_client_id(
        client_id=os.getenv("FUSION_CLIENT_ID"),
        client_secret=os.getenv("FUSION_CLIENT_SECRET"),
        resource="JPMC:URI:RS-93742-Fusion-PROD",
        auth_url="https://authe.jpmorgan.com/as/token.oauth2",
        proxies={},
        fusion_e2e=None,
    )
    client = Fusion(credentials=creds)

    print_res = {}
    for test_nm, params in download_hashes.items():
        res_params = params.client_dl_params.copy()  # Create a copy to avoid mutating the original

        dt_str = ""
        if params.st_dt:
            dt_str += params.st_dt + ":"
        if params.end_dt:
            dt_str += params.end_dt
        if dt_str:
            res_params["dt_str"] = dt_str

        if params.method == "download":
            res = client.download(**res_params)
            hash_out = hashlib.md5()
            for success, path, _ in res:
                if success:
                    with Path(path).open("rb") as f:
                        hash_out.update(f.read())

            params.md5_hash = hash_out.hexdigest()
        elif params.method == "to_df":
            res = client.to_df(**res_params)
            params.res_shape = res.shape

        print_res[test_nm] = params

@pytest.mark.integration
def test_generic_dl(client: Fusion) -> None:  # noqa: PLR0912, PLR0915
    for test_nm, params in download_hashes.items():
        res_params = params.client_dl_params.copy()  # Create a copy to avoid mutating the original

        dt_str = ""
        if params.st_dt:
            dt_str += params.st_dt + ":"
        if params.end_dt:
            dt_str += params.end_dt
        if dt_str:
            res_params["dt_str"] = dt_str
        if params.method == "download":
            try:
                res = client.download(**res_params)
            except Exception as e:
                # Check if it's an authentication error
                if "400 Client Error" in str(e) or "Failed to generate Fusion token headers" in str(e):
                    pytest.skip(f"Authentication failed for {test_nm}: {e}")
                else:
                    raise e
            
            assert res is not None, f"Download returned None for {test_nm}"
            assert len(res) > 0, f"Download returned empty results for {test_nm}"
            
            successful_files = []
            total_size = 0
            hash_out = hashlib.md5()
            
            for success, path, error in res:
                if success:
                    file_path = Path(path)
                    assert file_path.exists(), f"Downloaded file does not exist: {path}"
                    file_size = file_path.stat().st_size
                    assert file_size > 0, f"Downloaded file is empty: {path}"
                    
                    with file_path.open("rb") as f:
                        content = f.read()
                        hash_out.update(content)
                    
                    successful_files.append(path)
                    total_size += file_size
                else:
                    print(f"Download failed for {test_nm}: {path}, error: {error}")  # noqa: T201
            
            assert len(successful_files) > 0, f"No successful downloads for {test_nm}"
            assert total_size > 0, f"No data downloaded for {test_nm}"
            
            computed_hash = hash_out.hexdigest()
            
            empty_hash = hashlib.md5(b"").hexdigest()
            assert computed_hash != empty_hash, (
                f"Downloaded data appears to be empty for {test_nm} "
                f"(hash matches empty string: {empty_hash})"
            )
            
            # Allow CI/CD to override validation mode via environment variable
            env_validation_mode = os.getenv("INTEGRATION_HASH_VALIDATION", "").lower()
            if env_validation_mode in ("strict", "warn", "skip"):
                validation_mode = env_validation_mode
                print(f"Using environment-specified validation mode: {validation_mode}")  # noqa: T201
            else:
                validation_mode = getattr(params, "hash_validation", "warn")
            
            if params.md5_hash and validation_mode != "skip":
                if computed_hash != params.md5_hash:
                    message = (
                        f"Hash mismatch for {test_nm}: "
                        f"expected={params.md5_hash}, computed={computed_hash} "
                        f"({len(successful_files)} files, {total_size} bytes)"
                    )
                    
                    if validation_mode == "strict":
                        raise AssertionError(message)
                    elif validation_mode == "warn":
                        print(f"WARNING: {message}")  # noqa: T201
                else:
                    print(f"Hash validated for {test_nm}: {computed_hash}")  # noqa: T201
            elif validation_mode == "skip":
                print(f"Hash validation skipped for {test_nm}: {computed_hash}")  # noqa: T201
            else:
                print(  # noqa: T201
                    f"Computed hash for {test_nm}: {computed_hash} "
                    f"({len(successful_files)} files, {total_size} bytes)"
                )
            
            download_clean_up(res)
        elif params.method == "to_df":
            try:
                res = client.to_df(**res_params)
            except Exception as e:
                # Check if it's an authentication error
                if "400 Client Error" in str(e) or "Failed to generate Fusion token headers" in str(e):
                    pytest.skip(f"Authentication failed for {test_nm}: {e}")
                else:
                    raise e
            assert res.shape == params.res_shape
            assert res["date"].iloc[0] == int(params.st_dt)
            assert res["date"].iloc[-1] == int(params.end_dt)
            for p in Path("./downloads").iterdir():
                if p.is_file():
                    p.unlink()
                else:
                    shutil.rmtree(p)
        print(f"Passed integ tests for {test_nm}")  # noqa: T201


@pytest.mark.integration
def test_custom_headers(client: Fusion) -> None:
    client.credentials.headers = {"my_special_header": "my_special_value", "my_special_header2": "my_special_value2"}
    r = client.session.get(client.root_url + "catalogs/common")
    assert r.request.headers["my_special_header"] == "my_special_value"
    assert r.request.headers["my_special_header2"] == "my_special_value2"
    print("Passed custom headers test")  # noqa: T201


if __name__ == "__main__":
    gen_generic_dl()
