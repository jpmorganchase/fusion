import io
import multiprocessing as mp
import tempfile
import threading
from collections.abc import Generator
from pathlib import Path
from queue import Queue
from typing import Any, Optional
from unittest.mock import MagicMock, Mock, patch

import fsspec
import joblib
import pandas as pd
import polars as pl
import pytest
import requests
import requests_mock
from pytest_mock import MockerFixture
from tqdm import tqdm

from fusion._fusion import FusionCredentials
from fusion.authentication import FusionOAuthAdapter
from fusion.fusion import Fusion
from fusion.utils import (
    PathLikeT,
    _filename_to_distribution,
    _stream_single_file_new_session_dry_run,
    _worker,
    cpu_count,
    csv_to_table,
    distribution_to_url,
    download_single_file_threading,
    get_session,
    is_dataset_raw,
    json_to_table,
    normalise_dt_param_str,
    parquet_to_table,
    path_to_url,
    read_csv,
    read_json,
    stream_single_file_new_session,
    stream_single_file_new_session_chunks,
    tqdm_joblib,
    upload_files,
    validate_file_names,
)


@pytest.fixture()
def sample_csv_path(tmp_path: Path) -> Path:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("col1,col2\nvalue1,value2\n")
    return csv_path


@pytest.fixture()
def sample_csv_path_str(sample_csv_path: Path) -> str:
    return str(sample_csv_path)


@pytest.fixture()
def sample_json_path(tmp_path: Path) -> Path:
    json_path = tmp_path / "sample.json"
    json_path.write_text('{"col1": "value1", "col2": "value2"}\n')
    return json_path


@pytest.fixture()
def sample_json_path_str(sample_json_path: Path) -> str:
    return str(sample_json_path)


@pytest.fixture()
def sample_parquet_path(tmp_path: Path) -> Path:
    parquet_path = tmp_path / "sample.parquet"

    # Generate sample parquet file using your preferred method
    def generate_sample_parquet_file(parquet_path: Path) -> None:
        data = {"col1": ["value1"], "col2": ["value2"]}
        test_df = pd.DataFrame(data)
        test_df.to_parquet(parquet_path)

    # Generate sample parquet file using your preferred method
    generate_sample_parquet_file(parquet_path)
    return parquet_path


@pytest.fixture()
def sample_parquet_paths(tmp_path: Path) -> list[Path]:
    parquet_paths = []
    for i in range(3):
        parquet_path = tmp_path / f"sample_{i}.parquet"

        # Generate sample parquet file using your preferred method
        def generate_sample_parquet_file(parquet_path: Path) -> None:
            data = {"col1": ["value1"], "col2": ["value2"]}
            test_df = pd.DataFrame(data)
            test_df.to_parquet(parquet_path)

        # Generate sample parquet file using your preferred method
        generate_sample_parquet_file(parquet_path)
        parquet_paths.append(parquet_path)
    return parquet_paths


@pytest.fixture()
def sample_parquet_paths_str(sample_parquet_paths: list[Path]) -> list[str]:
    return [str(p) for p in sample_parquet_paths]


def test_cpu_count() -> None:
    assert cpu_count() > 0


def test_csv_to_table(sample_csv_path_str: str) -> None:
    # Test with filter parameter
    table = csv_to_table(sample_csv_path_str)
    assert len(table) == 1
    assert table.column_names == ["col1", "col2"]

    # Test with select parameter
    table = csv_to_table(sample_csv_path_str, columns=["col1"])
    assert len(table) == 1
    assert table.column_names == ["col1"]

    # Test with both filter and select parameters
    table = csv_to_table(sample_csv_path_str, columns=["col1"])
    assert len(table) == 1
    assert table.column_names == ["col1"]


def test_json_to_table(sample_json_path_str: str) -> None:
    # Test with filter parameter
    table = json_to_table(sample_json_path_str)
    assert len(table) == 1
    assert table.column_names == ["col1", "col2"]

    # Test with select parameter
    table = json_to_table(sample_json_path_str, columns=["col1"])
    assert len(table) == 1
    assert table.column_names == ["col1"]

    # Test with both filter and select parameters
    table = json_to_table(sample_json_path_str, columns=["col1"])
    assert len(table) == 1
    assert table.column_names == ["col1"]


def test_parquet_to_table(sample_parquet_path: Path) -> None:
    table = parquet_to_table(sample_parquet_path)
    assert len(table) > 0
    # Add more assertions based on the expected structure of the parquet file

    # Test with filter parameter
    table_filtered = parquet_to_table(sample_parquet_path)
    assert len(table_filtered) == 1
    assert table_filtered.column_names == ["col1", "col2"]

    # Test with select parameter
    table_selected = parquet_to_table(sample_parquet_path, columns=["col1"])
    assert len(table_selected) == 1
    assert table_selected.column_names == ["col1"]

    # Test with both filter and select parameters
    table_filtered_selected = parquet_to_table(sample_parquet_path, columns=["col1"])
    assert len(table_filtered_selected) == 1
    assert table_filtered_selected.column_names == ["col1"]


def test_parquet_to_table_with_multiple_files(sample_parquet_paths: list[PathLikeT]) -> None:
    tables = parquet_to_table(sample_parquet_paths)
    num_rows_in_fixture = 3
    assert len(tables) == num_rows_in_fixture


def test_read_csv(sample_csv_path_str: str) -> None:
    dataframe = read_csv(sample_csv_path_str)
    assert len(dataframe) == 1
    assert list(dataframe.columns) == ["col1", "col2"]

    # Test with filter parameter
    dataframe_filtered = read_csv(sample_csv_path_str)
    assert len(dataframe_filtered) == 1
    assert list(dataframe_filtered.columns) == ["col1", "col2"]

    # Test with select parameter
    dataframe_selected = read_csv(sample_csv_path_str, columns=["col1"])
    assert len(dataframe_selected) == 1
    assert list(dataframe_selected.columns) == ["col1"]

    # Test with both filter and select parameters
    dataframe_filtered_selected = read_csv(sample_csv_path_str, columns=["col1"])
    assert len(dataframe_filtered_selected) == 1
    assert list(dataframe_filtered_selected.columns) == ["col1"]


def test_read_json(sample_json_path_str: str) -> None:
    dataframe = read_json(sample_json_path_str)
    assert len(dataframe) == 1
    assert list(dataframe.columns) == ["col1", "col2"]


def test_cpu_count_with_num_threads_env_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    test_num_threads = 8
    monkeypatch.setenv("NUM_THREADS", str(test_num_threads))
    assert cpu_count() == test_num_threads


def test_cpu_count_with_thread_pool_size_argument() -> None:
    test_pool_sz = 4
    assert cpu_count(thread_pool_size=test_pool_sz) == test_pool_sz


def test_cpu_count_with_is_threading_argument() -> None:
    def_thread_cnt = 10
    assert cpu_count(is_threading=True) == def_thread_cnt


def test_cpu_count_with_default_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NUM_THREADS", raising=False)
    assert cpu_count() == mp.cpu_count()


def test_read_parquet_with_pandas_dataframe_type(sample_parquet_path: Path) -> None:
    import pandas as pd

    from fusion.utils import read_parquet

    # Test with default parameters
    dataframe = read_parquet(sample_parquet_path)
    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe) > 0

    # Test with selected columns
    dataframe_selected = read_parquet(sample_parquet_path, columns=["col1"])
    assert isinstance(dataframe_selected, pd.DataFrame)
    assert len(dataframe_selected) > 0
    assert list(dataframe_selected.columns) == ["col1"]

    # Test with filters
    filters: list[tuple[Any, ...]] = [("col1", "==", "value1")]
    dataframe_filtered = read_parquet(sample_parquet_path, filters=filters)
    assert isinstance(dataframe_filtered, pd.DataFrame)
    assert len(dataframe_filtered) > 0

    # Test with both selected columns and filters
    dataframe_selected_filtered = read_parquet(sample_parquet_path, columns=["col1"], filters=filters)
    assert isinstance(dataframe_selected_filtered, pd.DataFrame)
    assert len(dataframe_selected_filtered) > 0
    assert list(dataframe_selected_filtered.columns) == ["col1"]


def test_read_parquet_with_polars_dataframe_type(sample_parquet_path: Path) -> None:
    import polars as pl

    from fusion.utils import read_parquet

    # Test with default parameters
    dataframe = read_parquet(sample_parquet_path, dataframe_type="polars")
    assert isinstance(dataframe, pl.DataFrame)
    assert len(dataframe) > 0

    # Test with selected columns
    dataframe_selected = read_parquet(sample_parquet_path, columns=["col1"], dataframe_type="polars")
    assert isinstance(dataframe_selected, pl.DataFrame)
    assert len(dataframe_selected) > 0
    assert list(dataframe_selected.columns) == ["col1"]

    # Test with filters
    filters: list[tuple[Any, ...]] = [("col1", "==", "value1")]
    dataframe_filtered = read_parquet(sample_parquet_path, filters=filters, dataframe_type="polars")
    assert isinstance(dataframe_filtered, pl.DataFrame)
    assert len(dataframe_filtered) > 0

    # Test with both selected columns and filters
    dataframe_selected_filtered = read_parquet(
        sample_parquet_path,
        columns=["col1"],
        filters=filters,
        dataframe_type="polars",
    )
    assert isinstance(dataframe_selected_filtered, pl.DataFrame)
    assert len(dataframe_selected_filtered) > 0
    assert list(dataframe_selected_filtered.columns) == ["col1"]


def test_read_parquet_with_unknown_dataframe_type(sample_parquet_path: Path) -> None:
    from fusion.utils import read_parquet

    # Test with unknown dataframe type
    with pytest.raises(ValueError, match="Unknown DataFrame type"):
        read_parquet(sample_parquet_path, dataframe_type="unknown")


def test_normalise_dt_param_with_datetime() -> None:
    import datetime

    from fusion.utils import _normalise_dt_param

    dt = datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc)
    result = _normalise_dt_param(dt)
    assert result == "2022-01-01"


def test_normalise_dt_param_with_date() -> None:
    import datetime

    from fusion.utils import _normalise_dt_param

    dt = datetime.date(2022, 1, 1)
    result = _normalise_dt_param(dt)
    assert result == "2022-01-01"


def test_normalise_dt_param_with_integer() -> None:
    from fusion.utils import _normalise_dt_param

    dt = 20220101
    result = _normalise_dt_param(dt)
    assert result == "2022-01-01"


def test_normalise_dt_param_with_valid_string_format_1() -> None:
    from fusion.utils import _normalise_dt_param

    dt = "2022-01-01"
    result = _normalise_dt_param(dt)
    assert result == "2022-01-01"


def test_normalise_dt_param_with_valid_string_format_2() -> None:
    from fusion.utils import _normalise_dt_param

    dt = "20220101"
    result = _normalise_dt_param(dt)
    assert result == "2022-01-01"


def test_normalise_dt_param_with_valid_string_format_3() -> None:
    from fusion.utils import _normalise_dt_param

    dt = "20220101T1200"
    result = _normalise_dt_param(dt)
    assert result == "2022-01-01-1200"


def test_normalise_dt_param_with_invalid_format() -> None:
    from fusion.utils import _normalise_dt_param

    dt = "2022/01/01"
    with pytest.raises(ValueError, match="is not in a recognised data format"):
        _normalise_dt_param(dt)


def test_normalise_dt_param_with_invalid_type() -> None:
    from fusion.utils import _normalise_dt_param

    dt = 32.23
    with pytest.raises(ValueError, match="is not in a recognised data format"):
        _normalise_dt_param(dt)  # type: ignore


def test_normalise_dt_param_str() -> None:
    # Test with a single date
    dt = "2022-01-01"
    result = normalise_dt_param_str(dt)
    assert result == ("2022-01-01",)

    # Test with a date range
    dt = "2022-01-01:2022-01-31"
    result = normalise_dt_param_str(dt)
    assert result == ("2022-01-01", "2022-01-31")

    dt = "2022-01-01:2022-01-01:2022-01-01"
    with pytest.raises(ValueError, match=f"Unable to parse {dt} as either a date or an interval"):
        normalise_dt_param_str(dt)


def test_read_csv_with_default_parameters(sample_csv_path_str: str) -> None:
    dataframe = read_csv(sample_csv_path_str)
    assert len(dataframe) == 1
    assert list(dataframe.columns) == ["col1", "col2"]


def test_read_csv_with_selected_columns(sample_csv_path_str: str) -> None:
    dataframe = read_csv(sample_csv_path_str, columns=["col1"])
    assert len(dataframe) == 1
    assert list(dataframe.columns) == ["col1"]


def test_read_csv_with_filters(sample_csv_path_str: str) -> None:
    filters: list[tuple[Any, ...]] = [("col1", "==", "value1")]
    dataframe = read_csv(sample_csv_path_str, filters=filters)
    assert len(dataframe) == 1
    assert list(dataframe.columns) == ["col1", "col2"]


def test_read_csv_with_pandas_dataframe_type(sample_csv_path_str: str) -> None:
    dataframe = read_csv(sample_csv_path_str, dataframe_type="pandas")
    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe) == 1
    assert list(dataframe.columns) == ["col1", "col2"]


def test_read_csv_with_polars_dataframe_type(sample_csv_path_str: str) -> None:
    dataframe = read_csv(sample_csv_path_str, dataframe_type="polars")
    assert isinstance(dataframe, pl.DataFrame)
    assert len(dataframe) == 1
    assert list(dataframe.columns) == ["col1", "col2"]


def test_read_xxx_with_unknown_dataframe_type(sample_csv_path_str: str, sample_json_path_str: str) -> None:
    with pytest.raises(ValueError, match="Unknown DataFrame type"):
        read_csv(sample_csv_path_str, dataframe_type="unknown")
    with pytest.raises(Exception, match="Unknown DataFrame type"):
        read_json(sample_json_path_str, dataframe_type="unknown")


def test_read_json_with_pandas_dataframe_type(sample_json_path_str: str) -> None:
    # Test with default parameters
    dataframe = read_json(sample_json_path_str)
    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe) > 0

    # Test with selected columns
    dataframe_selected = read_json(sample_json_path_str, columns=["col1"])
    assert isinstance(dataframe_selected, pd.DataFrame)
    assert len(dataframe_selected) > 0
    assert list(dataframe_selected.columns) == ["col1"]

    # Test with filters
    filters: list[tuple[Any, ...]] = [("col1", "==", "value1")]
    dataframe_filtered = read_json(sample_json_path_str, filters=filters)
    assert isinstance(dataframe_filtered, pd.DataFrame)
    assert len(dataframe_filtered) > 0

    # Test with both selected columns and filters
    dataframe_selected_filtered = read_json(sample_json_path_str, columns=["col1"], filters=filters)
    assert isinstance(dataframe_selected_filtered, pd.DataFrame)
    assert len(dataframe_selected_filtered) > 0
    assert list(dataframe_selected_filtered.columns) == ["col1"]


def test_read_json_with_polars_dataframe_type(sample_json_path_str: str) -> None:
    # Test with default parameters
    dataframe = read_json(sample_json_path_str, dataframe_type="polars")
    assert isinstance(dataframe, pl.DataFrame)
    assert len(dataframe) > 0

    # Test with selected columns
    dataframe_selected = read_json(sample_json_path_str, columns=["col1"], dataframe_type="polars")
    assert isinstance(dataframe_selected, pl.DataFrame)
    assert len(dataframe_selected) > 0
    assert list(dataframe_selected.columns) == ["col1"]

    # Test with filters
    filters: list[tuple[Any, ...]] = [("col1", "==", "value1")]
    dataframe_filtered = read_json(sample_json_path_str, filters=filters, dataframe_type="polars")
    assert isinstance(dataframe_filtered, pl.DataFrame)
    assert len(dataframe_filtered) > 0

    # Test with both selected columns and filters
    dataframe_selected_filtered = read_json(
        sample_json_path_str,
        columns=["col1"],
        filters=filters,
        dataframe_type="polars",
    )
    assert isinstance(dataframe_selected_filtered, pl.DataFrame)
    assert len(dataframe_selected_filtered) > 0
    assert list(dataframe_selected_filtered.columns) == ["col1"]


@pytest.fixture()
def fs_fusion() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def fs_local() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def loop() -> pd.DataFrame:
    import pandas as pd

    data = {"url": ["url1", "url2"], "path": ["path1", "path2"]}
    return pd.DataFrame(data)


def test_path_to_url() -> None:
    # Test with default parameters
    result = path_to_url("path/to/dataset__catalog__datasetseries.csv")
    assert result == "catalog/datasets/dataset/datasetseries/datasetseries/distributions/csv"

    result = path_to_url("path/to/dataset__catalog__datasetseries.csv", is_raw=True)
    assert result == "catalog/datasets/dataset/datasetseries/datasetseries/distributions/raw"

    result = path_to_url("path/to/dataset__catalog__datasetseries.csv", is_download=True)
    assert result == "catalog/datasets/dataset/datasetseries/datasetseries/distributions/csv/operationType/download"

    result = path_to_url("path/to/dataset__catalog__datasetseries.csv", is_raw=True, is_download=True)
    assert result == "catalog/datasets/dataset/datasetseries/datasetseries/distributions/raw/operationType/download"


def test_filename_to_distribution() -> None:
    file_name = "dataset__catalog__datasetseries.csv"
    catalog, dataset, datasetseries, file_format = _filename_to_distribution(file_name)
    assert catalog == "catalog"
    assert dataset == "dataset"
    assert datasetseries == "datasetseries"
    assert file_format == "csv"

    file_name = "anotherdataset__anothercatalog__anotherdatasetseries.parquet"
    catalog, dataset, datasetseries, file_format = _filename_to_distribution(file_name)
    assert catalog == "anothercatalog"
    assert dataset == "anotherdataset"
    assert datasetseries == "anotherdatasetseries"
    assert file_format == "parquet"


def test_distribution_to_url() -> None:
    from fusion.utils import distribution_to_url

    root_url = "https://api.fusion.jpmc.com/"
    catalog = "my_catalog"
    dataset = "my_dataset"
    datasetseries = "2020-04-04"
    file_format = "csv"
    bad_series_chs = ["/", "\\"]
    exp_res = (
        f"{root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/" f"{datasetseries}/distributions/{file_format}"
    )
    for ch in bad_series_chs:
        datasetseries = f"2020-04-04{ch}"
        result = distribution_to_url(root_url, dataset, datasetseries, file_format, catalog)
        assert result == exp_res

    # Test is_download
    datasetseries = "2020-04-04"
    exp_res = (
        f"{root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/"
        f"{datasetseries}/distributions/{file_format}/operationType/download"
    )
    for ch in bad_series_chs:
        datasetseries_mod = f"2020-04-04{ch}"
        result = distribution_to_url(root_url, dataset, datasetseries_mod, file_format, catalog, is_download=True)
        assert result == exp_res

    # Test samples
    exp_res = f"{root_url}catalogs/{catalog}/datasets/{dataset}/sample/distributions/csv"
    datasetseries = "sample"
    assert distribution_to_url(root_url, dataset, datasetseries, file_format, catalog) == exp_res


def test_distribution_to_filename() -> None:
    from fusion.utils import distribution_to_filename

    root_dir = "/tmp"
    catalog = "my_catalog"
    dataset = "my_dataset"
    datasetseries = "2020-04-04"
    file_format = "csv"
    exp_res = f"{root_dir}/{dataset}__{catalog}__{datasetseries}.{file_format}"
    bad_series_chs = ["/", "\\"]
    for ch in bad_series_chs:
        datasetseries_mod = f"2020-04-04{ch}"
        res = distribution_to_filename(root_dir, dataset, datasetseries_mod, file_format, catalog)
        assert res == exp_res

    # Test hive partitioning
    exp_res = f"{root_dir}/{dataset}.{file_format}"
    for ch in bad_series_chs:
        datasetseries_mod = f"2020-04-04{ch}"
        res = distribution_to_filename(root_dir, dataset, datasetseries_mod, file_format, catalog, partitioning="hive")
        assert res == exp_res

    root_dir = "c:\\tmp"
    exp_res = f"{root_dir}\\{dataset}__{catalog}__{datasetseries}.{file_format}"
    res = distribution_to_filename(root_dir, dataset, datasetseries, file_format, catalog)
    assert res == exp_res


@pytest.mark.parametrize(
    ("overwrite", "exists", "expected_result"),
    [
        (True, True, 0),  # Overwrite enabled, file exists
        (False, True, 0),  # Overwrite disabled, file exists
        (True, False, 0),  # Overwrite enabled, file does not exist
        (False, False, 0),  # Overwrite disabled, file does not exist
    ],
)
def test_stream_file(overwrite: bool, exists: bool, expected_result: int) -> None:
    session = Mock(spec=requests.Session)
    url = "http://example.com/data"
    output_file = Mock(spec=fsspec.spec.AbstractBufferedFile)
    start, end = 0, 10
    lock = threading.Lock()
    results: list[tuple[bool, str, Optional[str]]] = [(False, "", "")] * 1  # single element list
    idx = 0
    fs = Mock(spec=fsspec.AbstractFileSystem)
    fs.exists.return_value = exists

    # Create a mock response object with the necessary context manager methods
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_response.content = b"0123456789"
    # Mock the __enter__ method to return the mock response itself
    mock_response.__enter__ = Mock(return_value=mock_response)
    # Mock the __exit__ method to do nothing
    mock_response.__exit__ = Mock()

    with (
        patch("fsspec.filesystem", return_value=fs),
        patch.object(session, "get", return_value=mock_response),
    ):
        # The actual function to test might need to be imported if it exists elsewhere
        result = stream_single_file_new_session_chunks(
            session, url, output_file, start, end, lock, results, idx, overwrite=overwrite, fs=fs
        )

        # Assertions to verify the behavior
        assert result == expected_result
        if not overwrite and exists:
            fs.exists.assert_called_once_with(output_file)
            assert results[idx] == (True, output_file, None)
        else:
            output_file.seek.assert_called_once_with(start)
            output_file.write.assert_called_once_with(b"0123456789")
            assert results[idx] == (True, output_file, None)


def test_stream_single_file_new_session_dry_run(
    credentials: FusionCredentials, requests_mock: requests_mock.Mocker, fusion_obj: Fusion
) -> None:
    catalog = "my_catalog"
    dataset = "my_dataset"
    datasetseries = "2020-04-04"
    file_format = "csv"
    out_f = "/tmp/tmp.tmp"
    url = distribution_to_url(fusion_obj.root_url, dataset, datasetseries, file_format, catalog)
    requests_mock.head(url)
    assert (True, out_f, None) == _stream_single_file_new_session_dry_run(credentials, url, "/tmp/tmp.tmp")

    requests_mock.head(url, status_code=500)
    res = _stream_single_file_new_session_dry_run(credentials, url, "/tmp/tmp.tmp")
    assert not res[0]
    assert res[1] == out_f


def test_stream_file_exception() -> None:
    session = Mock(spec=requests.Session)
    url = "http://example.com/data"
    output_file = Mock(spec=fsspec.spec.AbstractBufferedFile)
    start, end = 0, 10
    lock = threading.Lock()
    results: list[tuple[bool, str, Optional[str]]] = [(False, "", "")] * 1  # single element list

    idx = 0
    fs = Mock(spec=fsspec.AbstractFileSystem)

    with (
        patch("fsspec.filesystem", return_value=fs),
        patch.object(session, "get", side_effect=requests.HTTPError("Error")),  # type: ignore
    ):
        result = stream_single_file_new_session_chunks(  # noqa: F821
            session, url, output_file, start, end, lock, results, idx, overwrite=True, fs=fs
        )

        assert result == 1
        assert results[idx] == (False, output_file, "Error")


TmpFsT = tuple[fsspec.spec.AbstractFileSystem, str]


@pytest.fixture()
def temp_fs() -> Generator[TmpFsT, None, None]:
    with (
        tempfile.TemporaryDirectory() as tmpdirname,
        patch(
            "fsspec.filesystem", return_value=fsspec.filesystem("file", auto_mkdir=True, root_path=tmpdirname)
        ) as mock_fs,
    ):
        yield mock_fs, tmpdirname


def test_stream_file_with_temp_fs(temp_fs: TmpFsT, requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    _, tmpdirname = temp_fs

    output_file_path = Path(tmpdirname) / "output.dat"
    output_file = fsspec.open(output_file_path, mode="wb").open()  # Open a writable file for test

    start, end = 0, 10
    lock = threading.Lock()
    results: list[tuple[bool, str, Optional[str]]] = [(False, "", "")]

    idx = 0

    catalog = "my_catalog"
    dataset = "my_dataset"
    datasetseries = "2020-04-04"
    file_format = "csv"
    url = distribution_to_url(fusion_obj.root_url, catalog, dataset, datasetseries, file_format)

    requests_mock.get(url, content=b"0123456789")

    try:
        result = stream_single_file_new_session_chunks(
            requests.Session(),
            url,
            output_file,
            start,
            end,
            lock,
            results,
            idx,
            overwrite=True,
            fs=None,  # Pass None to simulate the condition you want to test
        )
        output_file.flush()
        # Read back what was written to ensure correctness
        with output_file_path.open("rb") as f:
            file_contents = f.read()
        assert file_contents == b"0123456789"
        assert result == 0, "Function should return 0 on success"
        assert results[idx] == (True, output_file, None), "Results should be updated correctly"
    finally:
        output_file.close()


def gen_binary_data(n: int, pad_len: int) -> list[bytes]:
    return [bin(i)[2:].zfill(pad_len).encode() for i in range(n)]


def test_worker(requests_mock: requests_mock.Mocker, temp_fs: TmpFsT, fusion_obj: Fusion) -> None:
    catalog = "my_catalog"
    dataset = "my_dataset"
    datasetseries = "2020-04-04"
    file_format = "csv"
    url = distribution_to_url(fusion_obj.root_url, dataset, datasetseries, file_format, catalog)
    splits = 10
    chunk_sz = 10
    bin_data = gen_binary_data(splits, chunk_sz)
    mock_fs, tmpdirname = temp_fs

    output_file_path = Path(tmpdirname) / "output.dat"
    output_file = fsspec.open(output_file_path, mode="wb").open()  # Open a writable file for test

    # generate all the get mocks
    for i in range(splits):
        start = i * chunk_sz
        end = start + chunk_sz - 1
        requests_mock.get(f"{url}?downloadRange=bytes={start}-{end-1}", content=bin_data[i])

    max_threads = 5
    results = [None] * splits
    queue: Queue[tuple[int, int, int]] = Queue(max_threads)
    lock = threading.Lock()

    threads = []
    for _ in range(max_threads):
        t = threading.Thread(target=_worker, args=(queue, requests.Session(), url, output_file, lock, results))
        t.start()
        threads.append(t)

    for i in range(splits):
        queue.put((i, i * chunk_sz, i * chunk_sz + chunk_sz - 1))
    queue.join()

    for _ in range(max_threads):
        queue.put((-1, -1, -1))
    for t in threads:
        t.join()

    output_file.close()

    with output_file_path.open("rb") as f:
        file_contents = f.read()
    for ix, line in enumerate(bin_data):
        assert line == file_contents[ix * chunk_sz : (ix + 1) * chunk_sz]


def test_progress_update() -> None:
    num_inputs = 100
    inputs = list(range(num_inputs))

    def true_if_even(x: int) -> tuple[bool, int]:
        return (x % 2 == 0, x)

    with tqdm_joblib(tqdm(total=num_inputs)):
        res = joblib.Parallel(n_jobs=10)(joblib.delayed(true_if_even)(i) for i in inputs)

    assert len(res) == num_inputs


@pytest.fixture()
def mock_fs_fusion() -> MagicMock:
    fs = MagicMock()
    fs.ls.side_effect = lambda path: {
        "": ["catalog1", "catalog2"],
        "catalog1/datasets": ["dataset1", "dataset2"],
        "catalog2/datasets": ["dataset3"],
    }.get(path, [])
    return fs


def test_validate_correct_file_names(mock_fs_fusion: MagicMock) -> None:
    paths = ["path/to/dataset1__catalog1__20230101.csv"]
    expected = [True]
    assert validate_file_names(paths, mock_fs_fusion) == expected


def test_validate_incorrect_format_file_names(mock_fs_fusion: MagicMock) -> None:
    paths = ["path/to/incorrectformatfile.csv"]
    expected = [False]
    assert validate_file_names(paths, mock_fs_fusion) == expected


def test_validate_non_existing_catalog(mock_fs_fusion: MagicMock) -> None:
    paths = ["path/to/dataset1__catalog3__20230101.csv"]
    expected = [False]
    assert validate_file_names(paths, mock_fs_fusion) == expected


def test_validate_non_existing_dataset(mock_fs_fusion: MagicMock) -> None:
    paths = ["path/to/dataset4__catalog1__20230101.csv"]
    expected = [False]
    assert validate_file_names(paths, mock_fs_fusion) == expected


def test_validate_error_paths(mock_fs_fusion: MagicMock) -> None:
    paths = ["path/to/catalog1__20230101.csv"]
    expected = [False]
    assert validate_file_names(paths, mock_fs_fusion) == expected


def test_empty_input_list(mock_fs_fusion: MagicMock) -> None:
    paths: list[str] = []
    expected: list[bool] = []
    assert validate_file_names(paths, mock_fs_fusion) == expected


def test_filesystem_exceptions(mock_fs_fusion: MagicMock) -> None:
    mock_fs_fusion.ls.side_effect = Exception("Failed to list directories")
    paths = ["path/to/dataset1__catalog1__20230101.csv"]
    with pytest.raises(Exception, match="Failed to list directories"):
        validate_file_names(paths, mock_fs_fusion)


def test_get_session(mocker: MockerFixture, credentials: FusionCredentials, fusion_obj: Fusion) -> None:
    session = get_session(credentials, fusion_obj.root_url)
    assert session
    session = get_session(credentials, fusion_obj.root_url, get_retries=3)
    assert session

    # Mock out the request to raise an exception
    mocker.patch("fusion.utils._get_canonical_root_url", side_effect=Exception("Failed to get canonical root url"))
    session = get_session(credentials, fusion_obj.root_url)
    for mnt, adapter_obj in session.adapters.items():
        if isinstance(adapter_obj, FusionOAuthAdapter):
            assert mnt == "https://"


def test_download_single_file_threading(
    temp_fs: TmpFsT, requests_mock: requests_mock.Mocker, credentials: FusionCredentials, fusion_obj: Fusion
) -> None:
    catalog = "my_catalog"
    dataset = "my_dataset"
    datasetseries = "2020-04-04"
    file_format = "csv"
    url = distribution_to_url(fusion_obj.root_url, dataset, datasetseries, file_format, catalog)
    splits = 10
    chunk_sz = 10
    bin_data = gen_binary_data(splits, chunk_sz)
    mock_fs, tmpdirname = temp_fs

    output_file_path = Path(tmpdirname) / "output.dat"
    output_file = fsspec.open(output_file_path, mode="wb").open()  # Open a writable file for test
    # generate all the get mocks
    requests_mock.head(url, headers={"Content-Length": str(splits * chunk_sz)})
    for i in range(splits):
        start = i * chunk_sz
        end = start + chunk_sz - 1
        mock_url = f"{url}?downloadRange=bytes={start}-{end}"
        mock_data = bin_data[i]
        requests_mock.get(mock_url, content=mock_data)

    download_single_file_threading(credentials=credentials, url=url, output_file=output_file, chunk_size=chunk_sz)

    with output_file_path.open("rb") as f:
        file_contents = f.read()
    for ix, line in enumerate(bin_data):
        assert line == file_contents[ix * chunk_sz : (ix + 1) * chunk_sz]


def test_stream_single_file_new_session(
    temp_fs: TmpFsT, requests_mock: requests_mock.Mocker, credentials: FusionCredentials, fusion_obj: Fusion
) -> None:
    catalog = "my_catalog"
    dataset = "my_dataset"
    datasetseries = "2020-04-04"
    file_format = "csv"
    url = distribution_to_url(fusion_obj.root_url, dataset, datasetseries, file_format, catalog)
    splits = 10
    chunk_sz = 10
    bin_data = gen_binary_data(splits, chunk_sz)
    mock_fs, tmpdirname = temp_fs

    output_file_path = Path(tmpdirname) / "output.dat"
    output_file = fsspec.open(output_file_path, mode="wb").open()  # Open a writable file for test
    # generate all the get mocks
    requests_mock.head(url, headers={"Content-Length": str(splits * chunk_sz)})
    requests_mock.get(url, content=b"".join(bin_data))

    stream_single_file_new_session(credentials=credentials, url=url, output_file=output_file)

    with output_file_path.open("rb") as f:
        file_contents = f.read()
    for ix, line in enumerate(bin_data):
        assert line == file_contents[ix * chunk_sz : (ix + 1) * chunk_sz]


def test_stream_single_file_new_session_dry_run_from_parent(
    temp_fs: TmpFsT, requests_mock: requests_mock.Mocker, credentials: FusionCredentials, fusion_obj: Fusion
) -> None:
    catalog = "my_catalog"
    dataset = "my_dataset"
    datasetseries = "2020-04-04"
    file_format = "csv"
    url = distribution_to_url(fusion_obj.root_url, dataset, datasetseries, file_format, catalog)
    splits = 10
    chunk_sz = 10
    mock_fs, tmpdirname = temp_fs

    output_file_path = Path(tmpdirname) / "output.dat"
    output_file = fsspec.open(output_file_path, mode="wb").open()  # Open a writable file for test
    # generate all the get mocks
    requests_mock.head(url, headers={"Content-Length": str(splits * chunk_sz)})
    stream_single_file_new_session(credentials=credentials, url=url, output_file=output_file, dry_run=True)


def test_stream_single_file_new_session_file_exists(
    temp_fs: TmpFsT, credentials: FusionCredentials, fusion_obj: Fusion
) -> None:
    catalog = "my_catalog"
    dataset = "my_dataset"
    datasetseries = "2020-04-04"
    file_format = "csv"
    url = distribution_to_url(fusion_obj.root_url, dataset, datasetseries, file_format, catalog)
    splits = 10
    chunk_sz = 10
    mock_fs, tmpdirname = temp_fs

    output_file_path = Path(tmpdirname) / "output.dat"
    output_file = fsspec.open(output_file_path, mode="wb").open()  # Open a writable file for test
    output_file.write(b"0" * (splits * chunk_sz))
    # generate all the get mocks
    stream_single_file_new_session(credentials=credentials, url=url, output_file=output_file, overwrite=False)


def test_stream_single_file_new_session_with_exception(
    temp_fs: TmpFsT, mocker: MockerFixture, credentials: FusionCredentials, fusion_obj: Fusion
) -> None:
    catalog = "my_catalog"
    dataset = "my_dataset"
    datasetseries = "2020-04-04"
    file_format = "csv"
    url = distribution_to_url(fusion_obj.root_url, dataset, datasetseries, file_format, catalog)
    mock_fs, tmpdirname = temp_fs

    output_file_path = Path(tmpdirname) / "output.dat"
    output_file = fsspec.open(output_file_path, mode="wb").open()  # Open a writable file for test
    mocker.patch("fusion.utils.get_session", side_effect=Exception("Failed to get session"))
    res = stream_single_file_new_session(credentials=credentials, url=url, output_file=output_file)
    assert not res[0]
    assert res[1] == output_file


@pytest.fixture()
def mock_fs_fusion_w_cat() -> MagicMock:
    fs = MagicMock()
    # Mock the 'cat' method to return JSON strings as bytes
    fs.cat.side_effect = lambda path: {
        "catalog1/datasets/dataset1": b'{"isRawData": true}',
        "catalog1/datasets/dataset2": b'{"isRawData": false}',
    }.get(path, b"{}")  # Default empty JSON if path not found
    return fs


def test_is_dataset_raw(mock_fs_fusion_w_cat: MagicMock) -> None:
    paths = ["path/to/dataset1__catalog1.csv"]
    expected = [True]
    assert is_dataset_raw(paths, mock_fs_fusion_w_cat) == expected


def test_is_dataset_raw_fail(mock_fs_fusion_w_cat: MagicMock) -> None:
    paths = ["path/to/dataset2__catalog1.csv"]
    expected = [False]
    assert is_dataset_raw(paths, mock_fs_fusion_w_cat) == expected


def test_is_dataset_raw_empty_input_list(mock_fs_fusion_w_cat: MagicMock) -> None:
    paths: list[str] = []
    expected: list[bool] = []
    assert is_dataset_raw(paths, mock_fs_fusion_w_cat) == expected


def test_is_dataset_raw_filesystem_exceptions(mock_fs_fusion_w_cat: MagicMock) -> None:
    # Let's assume that the file not found results in an empty JSON, thus False
    mock_fs_fusion_w_cat.cat.side_effect = Exception("File not found")
    paths = ["path/to/dataset1__catalog1.csv"]
    with pytest.raises(Exception, match="File not found"):
        is_dataset_raw(paths, mock_fs_fusion_w_cat)


def test_is_dataset_raw_caching_of_results(mock_fs_fusion_w_cat: MagicMock) -> None:
    paths = ["path/to/dataset1__catalog1.csv", "path/to/dataset1__catalog1.csv"]
    # The filesystem's `cat` method should only be called once due to caching
    is_dataset_raw(paths, mock_fs_fusion_w_cat)
    mock_fs_fusion_w_cat.cat.assert_called_once()


@pytest.fixture()
def setup_fs() -> tuple[fsspec.AbstractFileSystem, fsspec.AbstractFileSystem]:
    fs_fusion = MagicMock(spec=fsspec.AbstractFileSystem)
    fs_local = MagicMock(spec=fsspec.AbstractFileSystem)
    fs_local.size.return_value = 4 * 2**20  # Less than chunk_size to test single-part upload
    fs_fusion.put.return_value = None
    return fs_fusion, fs_local


@pytest.fixture()
def upload_row() -> pd.Series:  # type: ignore
    return pd.Series({"url": "http://example.com/file", "path": "localfile/path/file.txt"})


@pytest.fixture()
def upload_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "url": ["http://example.com/file1", "http://example.com/file2", "http://example.com/file3"],
            "path": ["localfile/path/file1.txt", "localfile/path/file2.txt", "localfile/path/file3.txt"],
        }
    )


def test_upload_public(
    setup_fs: tuple[fsspec.AbstractFileSystem, fsspec.AbstractFileSystem], upload_rows: pd.DataFrame
) -> None:
    fs_fusion, fs_local = setup_fs

    res = upload_files(fs_fusion, fs_local, upload_rows, show_progress=True, parallel=False)
    assert res
    res = upload_files(fs_fusion, fs_local, upload_rows, show_progress=False, parallel=False)
    assert res

    fs_local.size.return_value = 5 * 2**20
    fs_local = io.BytesIO(b"some data to simulate file content" * 100)
    res = upload_files(fs_fusion, fs_local, upload_rows, show_progress=False, parallel=False)
    assert res


def test_upload_public_parallel(
    setup_fs: tuple[fsspec.AbstractFileSystem, fsspec.AbstractFileSystem], upload_rows: pd.DataFrame
) -> None:
    fs_fusion, fs_local = setup_fs

    res = upload_files(fs_fusion, fs_local, upload_rows, show_progress=False, parallel=True)
    assert res

    fs_local.size.return_value = 5 * 2**20
    fs_local = io.BytesIO(b"some data to simulate file content" * 100)
    res = upload_files(fs_fusion, fs_local, upload_rows, show_progress=False, parallel=True)
    assert res
