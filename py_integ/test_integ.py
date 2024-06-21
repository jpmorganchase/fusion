import hashlib
import shutil
from pathlib import Path
from typing import Optional

from fusion import Fusion


def download_clean_up(download_res: Optional[list[tuple[bool, str, Optional[str]]]]) -> None:
    if not download_res:
        return

    for _, path_str, _ in download_res:
        Path(path_str).unlink(missing_ok=True)


def test_download_csv(client: Fusion) -> None:
    download_res = client.download(
        "FXO_SP", "20231201:20231208", dataset_format="csv", return_paths=True, force_download=True
    )
    # hash of all files in csv paths
    hash_out = hashlib.md5()
    for success, path, _ in download_res:
        if success:
            with Path(path).open("rb") as f:
                hash_out.update(f.read())
    download_clean_up(download_res)
    assert hash_out.hexdigest() == "5f9a51c38325947745c2e75afbb2d368"


def test_download_parquet(client: Fusion) -> None:
    download_res = client.download(
        "FXO_SP", "20231201:20231208", dataset_format="parquet", return_paths=True, force_download=True
    )
    hash_out = hashlib.md5()
    for success, path, _ in download_res:
        if success:
            with Path(path).open("rb") as f:
                hash_out.update(f.read())
    download_clean_up(download_res)
    assert hash_out.hexdigest() == "6c279fb1ed1bb56ef90b0b82b847d347"


def test_to_df_csv(client: Fusion) -> None:
    st_date = 20231201
    end_date = 20231208
    df_down = client.to_df("FXO_SP", f"{st_date}:{end_date}", dataset_format="csv")
    df_down = df_down.sort_values("date")
    assert df_down.shape == (366, 6)
    assert df_down["date"].iloc[0] == st_date
    assert df_down["date"].iloc[-1] == end_date
    for p in Path("./downloads").iterdir():
        if p.is_file():
            p.unlink()
        else:
            shutil.rmtree(p)


def test_to_df_parquet(client: Fusion) -> None:
    st_date = 20231201
    end_date = 20231208
    df_down = client.to_df("FXO_SP", f"{st_date}:{end_date}", dataset_format="parquet")
    df_down = df_down.sort_values("date")
    assert df_down.shape == (366, 6)
    assert df_down["date"].iloc[0] == st_date
    assert df_down["date"].iloc[-1] == end_date
    for p in Path("./downloads").iterdir():
        if p.is_file():
            p.unlink()
        else:
            shutil.rmtree(p)
