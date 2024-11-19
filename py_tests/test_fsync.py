import json
from pathlib import Path
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock, patch

from fusion._fusion import FusionCredentials
from fusion.fs_sync import _download, _url_to_path
from fusion.fusion_filesystem import FusionHTTPFileSystem
import pandas as pd


def test__url_to_path() -> None:
    # The function this is testing looks a bit broken. The test is here just to make sure we don't break it further.

    catalog = "my_catalog"
    dataset = "my_dataset"
    dt_str = "20200101"
    file_format = "csv"

    url = f"{catalog}/datasets/{dataset}/dataseries/{dt_str}/distributions/{file_format}"
    path = _url_to_path(url)
    exp_res = f"{catalog}/my_dataset/{dt_str}//{dataset}__{catalog}__{dt_str}.csv"
    assert path == exp_res


@patch.object(FusionHTTPFileSystem, "set_session", new_callable=AsyncMock)
@patch("fsspec.AbstractFileSystem", autospec=True)
@patch("fusion.fs_sync.Parallel")
@patch("fusion.fs_sync.delayed")
def test_download(
        mock_delayed: mock.Mock,
        mock_parallel: mock.Mock,
        mock_fs_class: mock.AsyncMock,
        mock_set_session: mock.AsyncMock,
        example_creds_dict: dict[str, Any],
        tmp_path: Path,
) -> None:
    """Test the download function of the fs_sync module."""
    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)

    fs = FusionHTTPFileSystem(credentials=creds)
    lfs = mock_fs_class.return_value
    input_df = pd.DataFrame(
        {
            "path_local": [None],
            "mtime": [None],
            "local_path": [None],
            "sha256_local": [None],
            "path_fusion": ["catalog/dataset/20200101//dataset__catalog__20200101.csv"],
            "url": ["catalog/datasets/dataset/datasetseries/20200101/distributions/csv"],
            "size": [100],
            "sha256_fusion": ["uyfkycxjurtxrx"],
        }
    )

    # Mock the delayed function to return the function itself
    mock_delayed.side_effect = lambda func, *args, **kwargs: func

    # Mock the Parallel object to return a callable that returns the expected result
    mock_parallel.return_value = lambda *args, **kwargs: [(True, "catalog/dataset/20200101//dataset__catalog__20200101.csv", None)]

    res = _download(fs, lfs, input_df, n_par=16)

    assert res == [(True, "catalog/dataset/20200101//dataset__catalog__20200101.csv", None)]


@patch.object(FusionHTTPFileSystem, "set_session", new_callable=AsyncMock)
@patch("fsspec.AbstractFileSystem", autospec=True)
@patch("fusion.fs_sync.Parallel")
@patch("fusion.fs_sync.delayed")
def test_download_no_progress(
        mock_delayed: mock.Mock,
        mock_parallel: mock.Mock,
        mock_fs_class: mock.AsyncMock,
        mock_set_session: mock.AsyncMock,
        example_creds_dict: dict[str, Any],
        tmp_path: Path,
) -> None:
    """Test the download function of the fs_sync module."""
    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)

    fs = FusionHTTPFileSystem(credentials=creds)
    lfs = mock_fs_class.return_value
    input_df = pd.DataFrame(
        {
            "path_local": [None],
            "mtime": [None],
            "local_path": [None],
            "sha256_local": [None],
            "path_fusion": ["catalog/dataset/20200101//dataset__catalog__20200101.csv"],
            "url": ["catalog/datasets/dataset/datasetseries/20200101/distributions/csv"],
            "size": [100],
            "sha256_fusion": ["uyfkycxjurtxrx"],
        }
    )

    # Mock the delayed function to return the function itself
    mock_delayed.side_effect = lambda func, *args, **kwargs: func

    # Mock the Parallel object to return a callable that returns the expected result
    mock_parallel.return_value = lambda *args, **kwargs: [(True, "catalog/dataset/20200101//dataset__catalog__20200101.csv", None)]

    res = _download(fs, lfs, input_df, n_par=16, show_progress=False)

    assert res == [(True, "catalog/dataset/20200101//dataset__catalog__20200101.csv", None)]


@patch.object(FusionHTTPFileSystem, "set_session", new_callable=AsyncMock)
@patch("fsspec.AbstractFileSystem", autospec=True)
def test_download_empty_df(
        mock_fs_class: mock.AsyncMock,
        mock_set_session: mock.AsyncMock,
        example_creds_dict: dict[str, Any],
        tmp_path: Path,
) -> None:
    """Test the download function of the fs_sync module."""
    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)

    fs = FusionHTTPFileSystem(credentials=creds)
    lfs = mock_fs_class.return_value
    input_df = pd.DataFrame(
        {
            "path_local": [],
            "mtime": [],
            "local_path": [],
            "sha256_local": [],
            "path_fusion": [],
            "url": [],
            "size": [],
            "sha256_fusion": [],
        }
    )

    res = _download(fs, lfs, input_df, n_par=16)

    assert res == []


def test_upload() -> None:
    """Test the upload function of the fs_sync module."""
    pass
