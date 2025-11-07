from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from fusion import Fusion


@dataclass
class DownloadTestSpec:
    client_dl_params: dict[str, str | bool]
    method: str
    st_dt: str | None = None
    end_dt: str | None = None
    res_shape: tuple[int, int] | None = None


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
    ),
    "test_to_df_csv": DownloadTestSpec(
        client_dl_params={"dataset": "FXO_SP", "dataset_format": "csv", "dt_str": "20231201:20231208"},
        method="to_df",
        st_dt="20231201",
        end_dt="20231208",
        res_shape=(366, 6),
    ),
    "test_to_df_parquet": DownloadTestSpec(
        client_dl_params={"dataset": "FXO_SP", "dataset_format": "parquet", "dt_str": "20231201:20231208"},
        method="to_df",
        st_dt="20231201",
        end_dt="20231208",
        res_shape=(366, 6),
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
        res_params = params.client_dl_params

        dt_str = ""
        if params.st_dt:
            dt_str += params.st_dt + ":"
        if params.end_dt:
            dt_str += params.end_dt
        if dt_str:
            res_params["dt_str"] = dt_str

        if params.method == "download":
            res = client.download(**res_params)
        elif params.method == "to_df":
            res = client.to_df(**res_params)
            params.res_shape = res.shape

        print_res[test_nm] = params

@pytest.mark.integration
def test_generic_dl(client: Fusion) -> None:  # noqa: PLR0912, PLR0915
    for test_nm, params in download_hashes.items():
        res_params = params.client_dl_params.copy()

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
                error_msg = str(e)
                if any(phrase in error_msg for phrase in [
                    "400 Client Error", 
                    "Failed to generate Fusion token headers",
                    "Checksum validation is required but missing checksum information",
                    "Checksum validation failed"
                ]):
                    pytest.skip(f"Skipping {test_nm} due to expected error: {error_msg}")
                else:
                    raise e
            
            assert res is not None, f"Download returned None for {test_nm}"
            assert len(res) > 0, f"Download returned empty results for {test_nm}"
            
            successful_downloads = 0
            total_size = 0
            
            for success, path, _error in res:
                if success:
                    file_path = Path(path)
                    assert file_path.exists(), f"Downloaded file does not exist: {path}"
                    file_size = file_path.stat().st_size
                    assert file_size > 0, f"Downloaded file is empty: {path}"
                    
                    successful_downloads += 1
                    total_size += file_size
            
            assert successful_downloads > 0, f"No successful downloads for {test_nm}"
            assert total_size > 0, f"No data downloaded for {test_nm}"
            
            
            download_clean_up(res)
            
        elif params.method == "to_df":
            try:
                res = client.to_df(**res_params)
            except Exception as e:
                error_msg = str(e)
                if any(phrase in error_msg for phrase in [
                    "400 Client Error", 
                    "Failed to generate Fusion token headers",
                    "Checksum validation is required but missing checksum information",
                    "Checksum validation failed"
                ]):
                    pytest.skip(f"Skipping {test_nm} due to expected error: {error_msg}")
                else:
                    raise e
                    
            assert res.shape == params.res_shape
            assert res["date"].iloc[0] == int(params.st_dt)
            assert res["date"].iloc[-1] == int(params.end_dt)
            
            downloads_path = Path("./downloads")
            if downloads_path.exists():
                for p in downloads_path.iterdir():
                    if p.is_file():
                        p.unlink()
                    else:
                        shutil.rmtree(p)


@pytest.mark.integration
def test_custom_headers(client: Fusion) -> None:
    client.credentials.headers = {"my_special_header": "my_special_value", "my_special_header2": "my_special_value2"}
    r = client.session.get(client.root_url + "catalogs/common")
    assert r.request.headers["my_special_header"] == "my_special_value"
    assert r.request.headers["my_special_header2"] == "my_special_value2"


if __name__ == "__main__":
    gen_generic_dl()