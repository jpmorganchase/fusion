from __future__ import annotations
import hashlib
import shutil
from pathlib import Path
from typing import Optional
from functools import partial
from fusion import Fusion
import pandas as pd

from dataclasses import dataclass


@dataclass
class DownloadTestSpec:
    client_dl_params: dict[str : str | bool]
    method: str
    st_dt: str | None = None
    end_dt: str | None = None
    res_shape: tuple[int, int] | None = None
    md5_hash: str | None = None


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
        md5_hash="17746f55909185c4fae932128fe09e07",
    ),
    "test_download_parquet": DownloadTestSpec(
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
        md5_hash="17746f55909185c4fae932128fe09e07",
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

    from fusion._fusion import FusionCredentials
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

    print(print_res)


def test_generic_dl(client: Fusion) -> None:
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
            hash_out = hashlib.md5()
            for success, path, _ in res:
                if success:
                    with Path(path).open("rb") as f:
                        print(f"Reading {path}")
                        hash_out.update(f.read())
            assert hash_out.hexdigest() == params.md5_hash, f"Failed hash for {test_nm}"
            #download_clean_up(res)
        elif params.method == "to_df":
            res = client.to_df(**res_params)
            assert res.shape == params.res_shape
            assert res["date"].iloc[0] == int(params.st_dt)
            assert res["date"].iloc[-1] == int(params.end_dt)
            for p in Path("./downloads").iterdir():
                if p.is_file():
                    p.unlink()
                else:
                    shutil.rmtree(p)
        print(f"Passed integ tests for {test_nm}")  # noqa: T201


def test_custom_headers(client: Fusion) -> None:
    client.credentials.headers = {"my_special_header": "my_special_value", "my_special_header2": "my_special_value2"}
    r = client.session.get(client.root_url + "catalogs/common")
    assert r.request.headers["my_special_header"] == "my_special_value"
    assert r.request.headers["my_special_header2"] == "my_special_value2"
    print("Passed custom headers test")  # noqa: T201


if __name__ == "__main__":
    gen_generic_dl()
