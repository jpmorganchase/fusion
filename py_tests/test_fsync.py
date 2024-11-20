import base64
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock, patch

import fsspec
import pandas as pd

from fusion._fusion import FusionCredentials
from fusion.fs_sync import _download, _generate_sha256_token, _get_fusion_df, _upload, _url_to_path
from fusion.fusion_filesystem import FusionHTTPFileSystem


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
    mock_set_session: mock.AsyncMock,  # noqa: ARG001
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
    mock_delayed.side_effect = lambda func, *args, **kwargs: func  # noqa: ARG005

    # Mock the Parallel object to return a callable that returns the expected result
    mock_parallel.return_value = lambda *args, **kwargs: [  # noqa: ARG005
        (True, "catalog/dataset/20200101//dataset__catalog__20200101.csv", None)
    ]

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
    mock_set_session: mock.AsyncMock,  # noqa: ARG001
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
    mock_delayed.side_effect = lambda func, *args, **kwargs: func  # noqa: ARG005

    # Mock the Parallel object to return a callable that returns the expected result
    mock_parallel.return_value = lambda *args, **kwargs: [  # noqa: ARG005
        (True, "catalog/dataset/20200101//dataset__catalog__20200101.csv", None)
    ]

    res = _download(fs, lfs, input_df, n_par=16, show_progress=False)

    assert res == [(True, "catalog/dataset/20200101//dataset__catalog__20200101.csv", None)]


@patch.object(FusionHTTPFileSystem, "set_session", new_callable=AsyncMock)
@patch("fsspec.AbstractFileSystem", autospec=True)
def test_download_empty_df(
    mock_fs_class: mock.AsyncMock,
    mock_set_session: mock.AsyncMock,  # noqa: ARG001
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


@patch.object(FusionHTTPFileSystem, "set_session", new_callable=AsyncMock)
@patch("fsspec.AbstractFileSystem", autospec=True)
def test_upload(
    mock_fs_class: mock.AsyncMock,
    mock_set_session: mock.AsyncMock,  # noqa: ARG001
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
            "path_local": ["catalog/dataset/20200101//dataset__catalog__20200101.csv"],
            "mtime": [123124123123],
            "local_path": ["home/catalog/dataset/20200101//dataset__catalog__20200101.csv"],
            "sha256_local": ["sjdfghaldfgb"],
            "path_fusion": ["catalog/dataset/20200101//dataset__catalog__20200101.csv"],
            "url": ["catalog/datasets/dataset/datasetseries/20200101/distributions/csv"],
            "size": [100],
            "sha256_fusion": ["uyfkycxjurtxrx"],
        }
    )

    res = _upload(fs, lfs, input_df, n_par=16, show_progress=False)

    assert res


def test_generate_sha256_token_single_chunk() -> None:
    """Test the _generate_sha256_token function of the fs_sync module."""
    fs = fsspec.filesystem("memory")
    path = "/test/file"
    data = b"hello world"
    fs.pipe(path, data)

    expected_hash = hashlib.sha256(data).digest()
    expected_token = base64.b64encode(expected_hash).decode()

    assert _generate_sha256_token(path, fs) == expected_token


def test_generate_sha256_token_multiple_chunks() -> None:
    """Test the _generate_sha256_token function of the fs_sync module."""
    fs = fsspec.filesystem("memory")
    path = "/test/file"
    data = b"hello world" * 1000  # large data to ensure multiple chunks
    fs.pipe(path, data)

    chunk_size = 5 * 2**10
    hash_sha256 = hashlib.sha256()
    for i in range(0, len(data), chunk_size):
        chunk = data[i : i + chunk_size]
        hash_sha256_chunk = hashlib.sha256(chunk).digest()
        hash_sha256.update(hash_sha256_chunk)

    expected_token = base64.b64encode(hash_sha256.digest()).decode()

    assert _generate_sha256_token(path, fs, chunk_size) == expected_token


@patch("fusion.fs_sync.fsspec.filesystem")
def test_get_fusion_df(mock_fs: mock.Mock) -> None:
    """Test the get_fusion_df function of the fs_sync module."""
    info_output = [
        {
            "key": "DATASET",
            "lastModified": "2024-11-20T09:04:28Z",
            "checksum": "SHA-123=sdjbvaldfuwi4ertbALVD",
            "distributions": [
                {
                    "key": "DATASET/20241119/distribution.csv",
                    "values": [
                        "2024-11-20T09:04:28Z",
                        "3075",
                        "SHA-256=dfaVNJAE49Y34AURPVB",
                        "catalog",
                        "DATASET",
                        "20241119",
                        "csv",
                        "api-bucket",
                        "sdjlfasjebfasdf",
                    ],
                },
                {
                    "key": "DATASET/20241119/distribution.parquet",
                    "values": [
                        "2024-11-20T09:04:23Z",
                        "3104",
                        "SHA-256=lSDBHAGQ4dfjlbse8934tgd",
                        "catalog",
                        "DATASET",
                        "20241119",
                        "parquet",
                        "api-bucket",
                        "sdfksdlfkdslfnadfe",
                    ],
                },
            ],
        }
    ]
    mock_fs_instance = mock_fs.return_value
    mock_fs_instance.info.return_value = {"changes": {"datasets": info_output}}

    # Define the input parameters
    fs_fusion = mock_fs_instance
    datasets_lst = ["DATASET"]
    catalog = "catalog"
    flatten = False
    dataset_format = None

    # Call the function
    result_df = _get_fusion_df(fs_fusion, datasets_lst, catalog, flatten, dataset_format)

    # Define the expected DataFrame
    expected_data = {
        "path": [
            "catalog/DATASET/20241119//DATASET__catalog__20241119.csv",
            "catalog/DATASET/20241119//DATASET__catalog__20241119.parquet"
        ],
        "url": [
            "catalog/datasets/DATASET/datasetseries/20241119/distributions/csv",
            "catalog/datasets/DATASET/datasetseries/20241119/distributions/parquet"
        ],
        "size": [3075, 3104],
        "sha256": ["dfaVNJAE49Y34AURPVB", "lSDBHAGQ4dfjlbse8934tgd"]
    }
    expected_df = pd.DataFrame(expected_data)
    expected_df["size"] = expected_df["size"].astype("object")

    # Assert the result
    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)


@patch("fusion.fs_sync.fsspec.filesystem")
def test_get_fusion_df_flatten(mock_fs: mock.Mock) -> None:
    """Test the _get_fusion_df function with flatten=True."""
    info_output = [
        {
            "key": "DATASET",
            "lastModified": "2024-11-20T09:04:28Z",
            "checksum": "SHA-123=sdjbvaldfuwi4ertbALVD",
            "distributions": [
                {
                    "key": "DATASET/20241119/distribution.csv",
                    "values": [
                        "2024-11-20T09:04:28Z",
                        "3075",
                        "SHA-256=dfaVNJAE49Y34AURPVB",
                        "catalog",
                        "DATASET",
                        "20241119",
                        "csv",
                        "api-bucket",
                        "sdjlfasjebfasdf",
                    ],
                },
                {
                    "key": "DATASET/20241119/distribution.parquet",
                    "values": [
                        "2024-11-20T09:04:23Z",
                        "3104",
                        "SHA-256=lSDBHAGQ4dfjlbse8934tgd",
                        "catalog",
                        "DATASET",
                        "20241119",
                        "parquet",
                        "api-bucket",
                        "sdfksdlfkdslfnadfe",
                    ],
                },
            ],
        }
    ]
    mock_fs_instance = mock_fs.return_value
    mock_fs_instance.info.return_value = {"changes": {"datasets": info_output}}

    # Define the input parameters
    fs_fusion = mock_fs_instance
    datasets_lst = ["DATASET"]
    catalog = "catalog"
    flatten = True
    dataset_format = None

    # Call the function
    result_df = _get_fusion_df(fs_fusion, datasets_lst, catalog, flatten, dataset_format)

    # Define the expected DataFrame
    expected_data = {
        "path": [
            "catalog/DATASET/DATASET__catalog__20241119.csv",
            "catalog/DATASET/DATASET__catalog__20241119.parquet"
        ],
        "url": [
            "catalog/datasets/DATASET/datasetseries/20241119/distributions/csv",
            "catalog/datasets/DATASET/datasetseries/20241119/distributions/parquet"
        ],
        "size": [3075, 3104],
        "sha256": ["dfaVNJAE49Y34AURPVB", "lSDBHAGQ4dfjlbse8934tgd"]
    }
    expected_df = pd.DataFrame(expected_data)
    expected_df["size"] = expected_df["size"].astype("object")

    # Assert the result
    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)


@patch("fusion.fs_sync.fsspec.filesystem")
def test_get_fusion_df_dataset_format(mock_fs: mock.Mock) -> None:
    """Test the _get_fusion_df function with dataset_format specified."""
    info_output = [
        {
            "key": "DATASET",
            "lastModified": "2024-11-20T09:04:28Z",
            "checksum": "SHA-123=sdjbvaldfuwi4ertbALVD",
            "distributions": [
                {
                    "key": "DATASET/20241119/distribution.csv",
                    "values": [
                        "2024-11-20T09:04:28Z",
                        "3075",
                        "SHA-256=dfaVNJAE49Y34AURPVB",
                        "catalog",
                        "DATASET",
                        "20241119",
                        "csv",
                        "api-bucket",
                        "sdjlfasjebfasdf",
                    ],
                },
                {
                    "key": "DATASET/20241119/distribution.parquet",
                    "values": [
                        "2024-11-20T09:04:23Z",
                        "3104",
                        "SHA-256=lSDBHAGQ4dfjlbse8934tgd",
                        "catalog",
                        "DATASET",
                        "20241119",
                        "parquet",
                        "api-bucket",
                        "sdfksdlfkdslfnadfe",
                    ],
                },
            ],
        }
    ]
    mock_fs_instance = mock_fs.return_value
    mock_fs_instance.info.return_value = {"changes": {"datasets": info_output}}

    # Define the input parameters
    fs_fusion = mock_fs_instance
    datasets_lst = ["DATASET"]
    catalog = "catalog"
    flatten = False
    dataset_format = "csv"

    # Call the function
    result_df = _get_fusion_df(fs_fusion, datasets_lst, catalog, flatten, dataset_format)

    # Define the expected DataFrame
    expected_data = {
        "path": [
            "catalog/DATASET/20241119//DATASET__catalog__20241119.csv"
        ],
        "url": [
            "catalog/datasets/DATASET/datasetseries/20241119/distributions/csv"
        ],
        "size": [3075],
        "sha256": ["dfaVNJAE49Y34AURPVB"]
    }
    expected_df = pd.DataFrame(expected_data)
    expected_df["size"] = expected_df["size"].astype("object")

    # Assert the result
    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)


@patch("fusion.fs_sync.fsspec.filesystem")
def test_get_fusion_df_no_changes(mock_fs: mock.Mock) -> None:
    """Test the _get_fusion_df function when there are no changes."""
    mock_fs_instance = mock_fs.return_value
    mock_fs_instance.info.return_value = {"changes": {"datasets": []}}

    # Define the input parameters
    fs_fusion = mock_fs_instance
    datasets_lst = ["DATASET"]
    catalog = "catalog"
    flatten = False
    dataset_format = None

    # Call the function
    result_df = _get_fusion_df(fs_fusion, datasets_lst, catalog, flatten, dataset_format)
    result_df = result_df.astype("object")

    # Define the expected DataFrame
    expected_df = pd.DataFrame(columns=["path", "url", "size", "sha256"])
    expected_df = expected_df.astype("object")

    # Assert the result
    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)
