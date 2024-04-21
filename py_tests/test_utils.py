import multiprocessing as mp
from unittest.mock import MagicMock

import pandas as pd
import polars as pl
import pytest

from fusion.utils import (
    _filename_to_distribution,
    cpu_count,
    csv_to_table,
    json_to_table,
    normalise_dt_param_str,  # Added import for the new function
    parquet_to_table,
    path_to_url,
    read_csv,
    read_json,
)


@pytest.fixture
def sample_csv_path(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("col1,col2\nvalue1,value2\n")
    return csv_path


@pytest.fixture
def sample_json_path(tmp_path):
    json_path = tmp_path / "sample.json"
    json_path.write_text('{"col1": "value1", "col2": "value2"}\n')
    return json_path


@pytest.fixture
def sample_parquet_path(tmp_path):
    parquet_path = tmp_path / "sample.parquet"

    # Generate sample parquet file using your preferred method
    def generate_sample_parquet_file(parquet_path):
        data = {"col1": ["value1"], "col2": ["value2"]}
        df = pd.DataFrame(data)
        df.to_parquet(parquet_path)

    # Generate sample parquet file using your preferred method
    generate_sample_parquet_file(parquet_path)
    return parquet_path


def test_cpu_count():
    assert cpu_count() > 0


def test_csv_to_table(sample_csv_path):
    # Test with filter parameter
    table = csv_to_table(sample_csv_path)
    assert len(table) == 1
    assert table.column_names == ["col1", "col2"]

    # Test with select parameter
    table = csv_to_table(sample_csv_path, columns=["col1"])
    assert len(table) == 1
    assert table.column_names == ["col1"]

    # Test with both filter and select parameters
    table = csv_to_table(sample_csv_path, columns=["col1"])
    assert len(table) == 1
    assert table.column_names == ["col1"]


def test_json_to_table(sample_json_path):
    # Test with filter parameter
    table = json_to_table(sample_json_path)
    assert len(table) == 1
    assert table.column_names == ["col1", "col2"]

    # Test with select parameter
    table = json_to_table(sample_json_path, columns=["col1"])
    assert len(table) == 1
    assert table.column_names == ["col1"]

    # Test with both filter and select parameters
    table = json_to_table(sample_json_path, columns=["col1"])
    assert len(table) == 1
    assert table.column_names == ["col1"]


def test_parquet_to_table(sample_parquet_path):
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


def test_read_csv(sample_csv_path):
    dataframe = read_csv(sample_csv_path)
    assert len(dataframe) == 1
    assert list(dataframe.columns) == ["col1", "col2"]

    # Test with filter parameter
    dataframe_filtered = read_csv(sample_csv_path)
    assert len(dataframe_filtered) == 1
    assert list(dataframe_filtered.columns) == ["col1", "col2"]

    # Test with select parameter
    dataframe_selected = read_csv(sample_csv_path, columns=["col1"])
    assert len(dataframe_selected) == 1
    assert list(dataframe_selected.columns) == ["col1"]

    # Test with both filter and select parameters
    dataframe_filtered_selected = read_csv(sample_csv_path, columns=["col1"])
    assert len(dataframe_filtered_selected) == 1
    assert list(dataframe_filtered_selected.columns) == ["col1"]


def test_read_json(sample_json_path):
    dataframe = read_json(sample_json_path)
    assert len(dataframe) == 1
    assert list(dataframe.columns) == ["col1", "col2"]


def test_cpu_count_with_num_threads_env_variable(monkeypatch):
    test_num_threads = 8
    monkeypatch.setenv("NUM_THREADS", str(test_num_threads))
    assert cpu_count() == test_num_threads


def test_cpu_count_with_thread_pool_size_argument():
    test_pool_sz = 4
    assert cpu_count(thread_pool_size=test_pool_sz) == test_pool_sz


def test_cpu_count_with_is_threading_argument():
    def_thread_cnt = 10
    assert cpu_count(is_threading=True) == def_thread_cnt


def test_cpu_count_with_default_behavior(monkeypatch):
    monkeypatch.delenv("NUM_THREADS", raising=False)
    assert cpu_count() == mp.cpu_count()


def test_read_parquet_with_pandas_dataframe_type(sample_parquet_path):
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
    dataframe_filtered = read_parquet(sample_parquet_path, filters=[("col1", "==", "value1")])
    assert isinstance(dataframe_filtered, pd.DataFrame)
    assert len(dataframe_filtered) > 0

    # Test with both selected columns and filters
    dataframe_selected_filtered = read_parquet(
        sample_parquet_path, columns=["col1"], filters=[("col1", "==", "value1")]
    )
    assert isinstance(dataframe_selected_filtered, pd.DataFrame)
    assert len(dataframe_selected_filtered) > 0
    assert list(dataframe_selected_filtered.columns) == ["col1"]


def test_read_parquet_with_polars_dataframe_type(sample_parquet_path):
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
    dataframe_filtered = read_parquet(sample_parquet_path, filters=[("col1", "==", "value1")], dataframe_type="polars")
    assert isinstance(dataframe_filtered, pl.DataFrame)
    assert len(dataframe_filtered) > 0

    # Test with both selected columns and filters
    dataframe_selected_filtered = read_parquet(
        sample_parquet_path,
        columns=["col1"],
        filters=[("col1", "==", "value1")],
        dataframe_type="polars",
    )
    assert isinstance(dataframe_selected_filtered, pl.DataFrame)
    assert len(dataframe_selected_filtered) > 0
    assert list(dataframe_selected_filtered.columns) == ["col1"]


def test_read_parquet_with_unknown_dataframe_type(sample_parquet_path):
    from fusion.utils import read_parquet

    # Test with unknown dataframe type
    with pytest.raises(ValueError):
        read_parquet(sample_parquet_path, dataframe_type="unknown")


def test_normalise_dt_param_with_datetime():
    import datetime

    from fusion.utils import _normalise_dt_param

    dt = datetime.datetime(2022, 1, 1)
    result = _normalise_dt_param(dt)
    assert result == "2022-01-01"


def test_normalise_dt_param_with_date():
    import datetime

    from fusion.utils import _normalise_dt_param

    dt = datetime.date(2022, 1, 1)
    result = _normalise_dt_param(dt)
    assert result == "2022-01-01"


def test_normalise_dt_param_with_integer():
    from fusion.utils import _normalise_dt_param

    dt = 20220101
    result = _normalise_dt_param(dt)
    assert result == "2022-01-01"


def test_normalise_dt_param_with_valid_string_format_1():
    from fusion.utils import _normalise_dt_param

    dt = "2022-01-01"
    result = _normalise_dt_param(dt)
    assert result == "2022-01-01"


def test_normalise_dt_param_with_valid_string_format_2():
    from fusion.utils import _normalise_dt_param

    dt = "20220101"
    result = _normalise_dt_param(dt)
    assert result == "2022-01-01"


def test_normalise_dt_param_with_valid_string_format_3():
    from fusion.utils import _normalise_dt_param

    dt = "20220101T1200"
    result = _normalise_dt_param(dt)
    assert result == "2022-01-01-1200"


def test_normalise_dt_param_with_invalid_format():
    from fusion.utils import _normalise_dt_param

    dt = "2022/01/01"
    with pytest.raises(ValueError):
        _normalise_dt_param(dt)


def test_normalise_dt_param_with_invalid_type():
    from fusion.utils import _normalise_dt_param

    dt = 32.23
    with pytest.raises(ValueError):
        _normalise_dt_param(dt)


def test_normalise_dt_param_str():
    # Test with a single date
    dt = "2022-01-01"
    result = normalise_dt_param_str(dt)
    assert result == ("2022-01-01",)

    # Test with a date range
    dt = "2022-01-01:2022-01-31"
    result = normalise_dt_param_str(dt)
    assert result == ("2022-01-01", "2022-01-31")

    dt = "2022-01-01:2022-01-01:2022-01-01"
    with pytest.raises(ValueError):
        normalise_dt_param_str(dt)


def test_read_csv_with_default_parameters(sample_csv_path):
    dataframe = read_csv(sample_csv_path)
    assert len(dataframe) == 1
    assert list(dataframe.columns) == ["col1", "col2"]


def test_read_csv_with_selected_columns(sample_csv_path):
    dataframe = read_csv(sample_csv_path, columns=["col1"])
    assert len(dataframe) == 1
    assert list(dataframe.columns) == ["col1"]


def test_read_csv_with_filters(sample_csv_path):
    dataframe = read_csv(sample_csv_path, filters=[("col1", "==", "value1")])
    assert len(dataframe) == 1
    assert list(dataframe.columns) == ["col1", "col2"]


def test_read_csv_with_pandas_dataframe_type(sample_csv_path):
    dataframe = read_csv(sample_csv_path, dataframe_type="pandas")
    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe) == 1
    assert list(dataframe.columns) == ["col1", "col2"]


def test_read_csv_with_polars_dataframe_type(sample_csv_path):
    dataframe = read_csv(sample_csv_path, dataframe_type="polars")
    assert isinstance(dataframe, pl.DataFrame)
    assert len(dataframe) == 1
    assert list(dataframe.columns) == ["col1", "col2"]


def test_read_csv_with_unknown_dataframe_type(sample_csv_path):
    with pytest.raises(ValueError):
        read_csv(sample_csv_path, dataframe_type="unknown")


def test_read_json_with_pandas_dataframe_type(sample_json_path):
    # Test with default parameters
    dataframe = read_json(sample_json_path)
    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe) > 0

    # Test with selected columns
    dataframe_selected = read_json(sample_json_path, columns=["col1"])
    assert isinstance(dataframe_selected, pd.DataFrame)
    assert len(dataframe_selected) > 0
    assert list(dataframe_selected.columns) == ["col1"]

    # Test with filters
    dataframe_filtered = read_json(sample_json_path, filters=[("col1", "==", "value1")])
    assert isinstance(dataframe_filtered, pd.DataFrame)
    assert len(dataframe_filtered) > 0

    # Test with both selected columns and filters
    dataframe_selected_filtered = read_json(sample_json_path, columns=["col1"], filters=[("col1", "==", "value1")])
    assert isinstance(dataframe_selected_filtered, pd.DataFrame)
    assert len(dataframe_selected_filtered) > 0
    assert list(dataframe_selected_filtered.columns) == ["col1"]


def test_read_json_with_polars_dataframe_type(sample_json_path):
    # Test with default parameters
    dataframe = read_json(sample_json_path, dataframe_type="polars")
    assert isinstance(dataframe, pl.DataFrame)
    assert len(dataframe) > 0

    # Test with selected columns
    dataframe_selected = read_json(sample_json_path, columns=["col1"], dataframe_type="polars")
    assert isinstance(dataframe_selected, pl.DataFrame)
    assert len(dataframe_selected) > 0
    assert list(dataframe_selected.columns) == ["col1"]

    # Test with filters
    dataframe_filtered = read_json(sample_json_path, filters=[("col1", "==", "value1")], dataframe_type="polars")
    assert isinstance(dataframe_filtered, pl.DataFrame)
    assert len(dataframe_filtered) > 0

    # Test with both selected columns and filters
    dataframe_selected_filtered = read_json(
        sample_json_path,
        columns=["col1"],
        filters=[("col1", "==", "value1")],
        dataframe_type="polars",
    )
    assert isinstance(dataframe_selected_filtered, pl.DataFrame)
    assert len(dataframe_selected_filtered) > 0
    assert list(dataframe_selected_filtered.columns) == ["col1"]


@pytest.fixture
def fs_fusion():
    return MagicMock()


@pytest.fixture
def fs_local():
    return MagicMock()


@pytest.fixture
def loop():
    import pandas as pd

    data = {"url": ["url1", "url2"], "path": ["path1", "path2"]}
    return pd.DataFrame(data)


def test_path_to_url():
    # Test with default parameters
    result = path_to_url("path/to/dataset__catalog__datasetseries.csv")
    assert result == "catalog/datasets/dataset/datasetseries/datasetseries/distributions/csv"

    # Test with is_raw=True
    result = path_to_url("path/to/dataset__catalog__datasetseries.csv", is_raw=True)
    assert result == "catalog/datasets/dataset/datasetseries/datasetseries/distributions/raw"

    # Test with is_download=True
    result = path_to_url("path/to/dataset__catalog__datasetseries.csv", is_download=True)
    assert result == "catalog/datasets/dataset/datasetseries/datasetseries/distributions/csv/operationType/download"

    # Test with both is_raw=True and is_download=True
    result = path_to_url("path/to/dataset__catalog__datasetseries.csv", is_raw=True, is_download=True)
    assert result == "catalog/datasets/dataset/datasetseries/datasetseries/distributions/raw/operationType/download"


def test_filename_to_distribution():
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
