import io
import multiprocessing as mp
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import fsspec
import joblib
import pandas as pd
import polars as pl
import pytest
from pytest_mock import MockerFixture

from fusion._fusion import FusionCredentials
from fusion.authentication import FusionOAuthAdapter
from fusion.fusion import Fusion
from fusion.utils import (
    PathLikeT,
    _filename_to_distribution,
    convert_date_format,
    cpu_count,
    csv_to_table,
    get_session,
    is_dataset_raw,
    joblib_progress,
    json_to_table,
    make_bool,
    make_list,
    normalise_dt_param_str,
    parquet_to_table,
    path_to_url,
    read_csv,
    read_json,
    snake_to_camel,
    tidy_string,
    upload_files,
    validate_file_names,
)


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> Path:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("col1,col2\nvalue1,value2\n")
    return csv_path


@pytest.fixture
def sample_csv_path_str(sample_csv_path: Path) -> str:
    return str(sample_csv_path)


@pytest.fixture
def sample_json_path(tmp_path: Path) -> Path:
    json_path = tmp_path / "sample.json"
    json_path.write_text('{"col1": "value1", "col2": "value2"}\n')
    return json_path


@pytest.fixture
def sample_json_path_str(sample_json_path: Path) -> str:
    return str(sample_json_path)


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
def fs_fusion() -> MagicMock:
    return MagicMock()


@pytest.fixture
def fs_local() -> MagicMock:
    return MagicMock()


@pytest.fixture
def loop() -> pd.DataFrame:
    import pandas as pd

    data = {"url": ["url1", "url2"], "path": ["path1", "path2"]}
    return pd.DataFrame(data)


def test_path_to_url() -> None:
    # Test with default parameters
    result = path_to_url("path/to/dataset__catalog__datasetseries.csv")
    assert result == "catalog/datasets/dataset/datasetseries/datasetseries/distributions/csv"

    result = path_to_url("path/to/dataset__catalog__datasetseries.csv", is_raw=True)
    assert result == "catalog/datasets/dataset/datasetseries/datasetseries/distributions/csv"

    result = path_to_url("path/to/dataset__catalog__datasetseries.csv", is_download=True)
    assert result == "catalog/datasets/dataset/datasetseries/datasetseries/distributions/csv/operationType/download"

    result = path_to_url("path/to/dataset__catalog__datasetseries.csv", is_raw=True, is_download=True)
    assert result == "catalog/datasets/dataset/datasetseries/datasetseries/distributions/csv/operationType/download"

    result = path_to_url("path/to/dataset__catalog__datasetseries.pt", is_raw=True, is_download=True)
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
        f"{root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{datasetseries}/distributions/{file_format}"
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


TmpFsT = tuple[fsspec.spec.AbstractFileSystem, str]


@pytest.fixture
def temp_fs() -> Generator[TmpFsT, None, None]:
    with (
        tempfile.TemporaryDirectory() as tmpdirname,
        patch(
            "fsspec.filesystem", return_value=fsspec.filesystem("file", auto_mkdir=True, root_path=tmpdirname)
        ) as mock_fs,
    ):
        yield mock_fs, tmpdirname


def gen_binary_data(n: int, pad_len: int) -> list[bytes]:
    return [bin(i)[2:].zfill(pad_len).encode() for i in range(n)]


def test_progress_update() -> None:
    num_inputs = 100
    inputs = list(range(num_inputs))

    def true_if_even(x: int) -> tuple[bool, int]:
        return (x % 2 == 0, x)

    with joblib_progress("Uploading:", total=num_inputs):
        res = joblib.Parallel(n_jobs=10)(joblib.delayed(true_if_even)(i) for i in inputs)

    assert len(res) == num_inputs


@pytest.fixture
def mock_fs_fusion() -> MagicMock:
    fs = MagicMock()
    fs.ls.side_effect = lambda path: {
        "catalog1": ["catalog1", "catalog2"],
        "catalog2": ["catalog1", "catalog2"],
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


@pytest.fixture
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


@pytest.fixture
def setup_fs() -> tuple[fsspec.AbstractFileSystem, fsspec.AbstractFileSystem]:
    fs_fusion = MagicMock(spec=fsspec.AbstractFileSystem)
    fs_local = MagicMock(spec=fsspec.AbstractFileSystem)
    fs_local.size.return_value = 4 * 2**20  # Less than chunk_size to test single-part upload
    fs_fusion.put.return_value = None
    return fs_fusion, fs_local


@pytest.fixture
def upload_row() -> pd.Series:  # type: ignore
    return pd.Series({"url": "http://example.com/file", "path": "localfile/path/file.txt"})


@pytest.fixture
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


def test_tidy_string() -> None:
    """Test the tidy_string function."""
    bad_string = " string with  spaces  and  multiple  spaces  "

    assert tidy_string(bad_string) == "string with spaces and multiple spaces"


def test_make_list_from_string() -> None:
    """Test make list from string."""
    string_obj = "Hello, hi, hey"
    string_to_list = make_list(string_obj)
    assert isinstance(string_to_list, list)
    exp_len = 3
    assert len(string_to_list) == exp_len
    assert string_to_list[0] == "Hello"
    assert string_to_list[1] == "hi"
    assert string_to_list[2] == "hey"


def test_make_list_from_list() -> None:
    """Test make list from list."""

    list_obj = ["hi", "hi"]
    list_to_list = make_list(list_obj)
    assert isinstance(list_to_list, list)
    exp_len = 2
    assert len(list_to_list) == exp_len
    assert list_to_list[0] == "hi"
    assert list_to_list[1] == "hi"


def test_make_list_from_nonstring() -> None:
    """Test make list from non string."""
    any_obj = 1
    obj_to_list = make_list(any_obj)
    assert isinstance(obj_to_list, list)
    exp_len = 1
    assert len(obj_to_list) == exp_len
    assert obj_to_list[0] == cast(str, any_obj)


def test_make_bool_string() -> None:
    """Test make bool."""

    input_ = "string"
    output_ = make_bool(input_)
    assert output_ is True


def test_make_bool_hidden_false() -> None:
    """Test make bool."""

    input_1 = "False"
    input_2 = "false"
    input_3 = "FALSE"
    input_4 = "0"

    output_1 = make_bool(input_1)
    output_2 = make_bool(input_2)
    output_3 = make_bool(input_3)
    output_4 = make_bool(input_4)

    assert output_1 is False
    assert output_2 is False
    assert output_3 is False
    assert output_4 is False


def test_make_bool_bool() -> None:
    """Test make bool."""

    input_ = True
    output_ = make_bool(input_)
    assert output_ is True


def test_make_bool_1() -> None:
    """Test make bool."""

    input_ = 1
    output_ = make_bool(input_)
    assert output_ is True


def test_make_bool_0() -> None:
    """Test make bool."""

    input_ = 0
    output_ = make_bool(input_)
    assert output_ is False


def test_convert_date_format_month() -> None:
    """Test convert date format."""
    date = "May 6, 2024"
    output_ = convert_date_format(date)
    exp_output = "2024-05-06"
    assert output_ == exp_output


def test_convert_format_one_string() -> None:
    """Test convert date format."""
    date = "20240506"
    output_ = convert_date_format(date)
    exp_output = "2024-05-06"
    assert output_ == exp_output


def test_convert_format_slash() -> None:
    """Test convert date format."""
    date = "2024/05/06"
    output_ = convert_date_format(date)
    exp_output = "2024-05-06"
    assert output_ == exp_output


def test_snake_to_camel() -> None:
    """Test snake to camel."""
    input_ = "this_is_snake"
    output_ = snake_to_camel(input_)
    exp_output = "thisIsSnake"
    assert output_ == exp_output
