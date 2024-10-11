import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests
import requests_mock
from pytest_mock import MockerFixture

from fusion._fusion import FusionCredentials
from fusion.fusion import Fusion
from fusion.utils import _normalise_dt_param, distribution_to_url


def test_rust_ok() -> None:
    from fusion import rust_ok

    assert rust_ok()


def test__get_canonical_root_url() -> None:
    from fusion.utils import _get_canonical_root_url

    some_url = "https://fusion.jpmorgan.com/api/v1/a_given_resource"
    root_url = "https://fusion.jpmorgan.com"
    assert root_url == _get_canonical_root_url(some_url)


def test_fusion_credentials(example_creds_dict: dict[str, Any], tmp_path: Path) -> None:
    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)
    assert FusionCredentials.from_file(credentials_file)


def test_fusion_init(credentials: FusionCredentials) -> None:
    fusion = Fusion(credentials=credentials)
    assert fusion


def test_fusion_credentials_no_pxy(example_creds_dict_no_pxy: dict[str, Any], tmp_path: Path) -> None:
    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict_no_pxy, f)
    assert FusionCredentials.from_file(credentials_file)


def test_fusion_credentials_empty_pxy(example_creds_dict_empty_pxy: dict[str, Any], tmp_path: Path) -> None:
    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict_empty_pxy, f)

    assert FusionCredentials.from_file(credentials_file)


def test_use_catalog(credentials: FusionCredentials) -> None:
    fusion = Fusion(credentials=credentials)
    def_cat = "def_cat"
    fusion.default_catalog = def_cat
    assert def_cat == fusion._use_catalog(None)
    new_cat = "new_cat"
    assert new_cat == fusion._use_catalog(new_cat)


def test_date_parsing() -> None:
    assert _normalise_dt_param(20201212) == "2020-12-12"
    assert _normalise_dt_param("20201212") == "2020-12-12"
    assert _normalise_dt_param("2020-12-12") == "2020-12-12"
    assert _normalise_dt_param(datetime.date(2020, 12, 12)) == "2020-12-12"
    dtm = datetime.datetime(2020, 12, 12, 23, 55, 59, 342380, tzinfo=datetime.timezone.utc)
    assert _normalise_dt_param(dtm) == "2020-12-12"


@pytest.mark.parametrize("ref_int", [-1, 0, 1, 2])
@pytest.mark.parametrize("pluraliser", [None, "s", "es"])
def test_res_plural(ref_int: int, pluraliser: str) -> None:
    from fusion.authentication import _res_plural

    res = _res_plural(ref_int, pluraliser)
    if abs(ref_int) == 1:
        assert res == ""
    else:
        assert res == pluraliser


def test_is_url() -> None:
    from fusion.authentication import _is_url
    assert _is_url("https://www.google.com")
    assert _is_url("http://www.google.com/some/path?qp1=1&qp2=2")
    assert not _is_url("www.google.com")
    assert not _is_url("google.com")
    assert not _is_url("google")
    assert not _is_url("googlecom")
    assert not _is_url("googlecom.")
    assert not _is_url(3.141)  # type: ignore


def test_fusion_class(fusion_obj: Fusion) -> None:
    assert fusion_obj
    assert repr(fusion_obj)
    assert fusion_obj.default_catalog == "common"
    fusion_obj.default_catalog = "other"
    assert fusion_obj.default_catalog == "other"


def test_get_fusion_filesystem(fusion_obj: Fusion) -> None:
    filesystem = fusion_obj.get_fusion_filesystem()
    assert filesystem is not None


def test__call_for_dataframe_success(requests_mock: requests_mock.Mocker) -> None:
    # Mock the response from the API endpoint
    url = "https://fusion.jpmorgan.com/api/v1/a_given_resource"
    expected_data = {"resources": [{"id": 1, "name": "Resource 1"}, {"id": 2, "name": "Resource 2"}]}
    requests_mock.get(url, json=expected_data)

    # Create a mock session
    session = requests.Session()

    # Call the _call_for_dataframe function
    test_df = Fusion._call_for_dataframe(url, session)

    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(expected_data["resources"])
    pd.testing.assert_frame_equal(test_df, expected_df)


def test__call_for_dataframe_error(requests_mock: requests_mock.Mocker) -> None:
    # Mock the response from the API endpoint with an error status code
    url = "https://fusion.jpmorgan.com/api/v1/a_given_resource"
    requests_mock.get(url, status_code=500)

    # Create a mock session
    session = requests.Session()

    # Call the _call_for_dataframe function and expect an exception to be raised
    with pytest.raises(requests.exceptions.HTTPError):
        Fusion._call_for_dataframe(url, session)


def test__call_for_bytes_object_success(requests_mock: requests_mock.Mocker) -> None:
    # Mock the response from the API endpoint
    url = "https://fusion.jpmorgan.com/api/v1/a_given_resource"
    expected_data = b"some binary data"
    requests_mock.get(url, content=expected_data)

    # Create a mock session
    session = requests.Session()

    # Call the _call_for_bytes_object function
    data = Fusion._call_for_bytes_object(url, session)

    # Check if the data is returned correctly
    assert data.getbuffer() == expected_data


def test__call_for_bytes_object_fail(requests_mock: requests_mock.Mocker) -> None:
    # Mock the response from the API endpoint with an error status code
    url = "https://fusion.jpmorgan.com/api/v1/a_given_resource"
    requests_mock.get(url, status_code=500)

    # Create a mock session
    session = requests.Session()

    # Call the _call_for_dataframe function and expect an exception to be raised
    with pytest.raises(requests.exceptions.HTTPError):
        Fusion._call_for_bytes_object(url, session)


def test_list_catalogs_success(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    # Mock the response from the API endpoint
    url = "https://fusion.jpmorgan.com/api/v1/catalogs/"
    expected_data = {"resources": [{"id": 1, "name": "Catalog 1"}, {"id": 2, "name": "Catalog 2"}]}
    requests_mock.get(url, json=expected_data)

    # Call the list_catalogs method
    test_df = fusion_obj.list_catalogs()

    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(expected_data["resources"])
    pd.testing.assert_frame_equal(test_df, expected_df)


def test_list_catalogs_fail(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    # Mock the response from the API endpoint with an error status code
    url = "https://fusion.jpmorgan.com/api/v1/catalogs/"
    requests_mock.get(url, status_code=500)

    # Call the list_catalogs method and expect an exception to be raised
    with pytest.raises(requests.exceptions.HTTPError):
        fusion_obj.list_catalogs()


def test_catalog_resources_success(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    # Mock the response from the API endpoint

    new_catalog = "catalog_id"
    url = f"{fusion_obj.root_url}catalogs/{new_catalog}"
    expected_data = {"resources": [{"id": 1, "name": "Resource 1"}, {"id": 2, "name": "Resource 2"}]}
    requests_mock.get(url, json=expected_data)

    # Call the catalog_resources method
    test_df = fusion_obj.catalog_resources(new_catalog)

    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(expected_data["resources"])
    pd.testing.assert_frame_equal(test_df, expected_df)


def test_list_products_success(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    new_catalog = "catalog_id"
    url = f"{fusion_obj.root_url}catalogs/{new_catalog}/products"
    server_mock_data = {
        "resources": [{"category": ["FX"], "region": ["US"]}, {"category": ["FX"], "region": ["US", "EU"]}]
    }
    expected_data = {"resources": [{"category": "FX", "region": "US"}, {"category": "FX", "region": "US, EU"}]}

    requests_mock.get(url, json=server_mock_data)

    # Call the catalog_resources method
    test_df = fusion_obj.list_products(catalog=new_catalog, max_results=2)
    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(expected_data["resources"])
    pd.testing.assert_frame_equal(test_df, expected_df)


def test_list_products_contains_success(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    new_catalog = "catalog_id"
    url = f"{fusion_obj.root_url}catalogs/{new_catalog}/products"
    server_mock_data = {
        "resources": [
            {"identifier": "1", "description": "some desc", "category": ["FX"], "region": ["US"]},
            {"identifier": "2", "description": "some desc", "category": ["FX"], "region": ["US", "EU"]},
        ]
    }
    expected_data = {
        "resources": [
            {"identifier": "1", "description": "some desc", "category": "FX", "region": "US"},
        ]
    }
    expected_df = pd.DataFrame(expected_data["resources"])

    requests_mock.get(url, json=server_mock_data)

    # Call the catalog_resources method
    test_df = fusion_obj.list_products(catalog=new_catalog, max_results=2, contains=["1"])
    # Check if the dataframe is created correctly
    pd.testing.assert_frame_equal(test_df, expected_df)

    test_df = fusion_obj.list_products(catalog=new_catalog, max_results=2, contains="1")
    # Check if the dataframe is created correctly
    pd.testing.assert_frame_equal(test_df, expected_df)

    test_df = fusion_obj.list_products(catalog=new_catalog, max_results=2, contains="1", id_contains=True)
    # Check if the dataframe is created correctly
    pd.testing.assert_frame_equal(test_df, expected_df)


def test_list_datasets_success(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    new_catalog = "catalog_id"
    url = f"{fusion_obj.root_url}catalogs/{new_catalog}/datasets"
    server_mock_data = {
        "resources": [{"category": ["FX"], "region": ["US"]}, {"category": ["FX"], "region": ["US", "EU"]}]
    }
    expected_data = {"resources": [{"region": "US", "category": "FX"}, {"region": "US, EU", "category": "FX"}]}

    requests_mock.get(url, json=server_mock_data)

    # Call the catalog_resources method
    test_df = fusion_obj.list_datasets(catalog=new_catalog, max_results=2)
    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(expected_data["resources"])
    pd.testing.assert_frame_equal(test_df, expected_df)


def test_list_datasets_contains_success(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    new_catalog = "catalog_id"
    url = f"{fusion_obj.root_url}catalogs/{new_catalog}/datasets"
    server_mock_data = {
        "resources": [
            {"identifier": "ONE", "description": "some desc", "category": ["FX"], "region": ["US"], "status": "active"},
            {
                "identifier": "TWO",
                "description": "some desc",
                "category": ["FX"],
                "region": ["US", "EU"],
                "status": "inactive",
            },
        ]
    }
    expected_data = {
        "resources": [
            {"identifier": "ONE", "region": "US", "category": "FX", "description": "some desc", "status": "active"}
        ]
    }

    expected_df = pd.DataFrame(expected_data["resources"])

    requests_mock.get(url, json=server_mock_data)

    select_prod = "prod_a"
    prod_url = f"{fusion_obj.root_url}catalogs/{new_catalog}/productDatasets"
    server_prod_mock_data = {
        "resources": [
            {"product": select_prod, "dataset": "one"},
            {"product": "prod_b", "dataset": "two"},
        ]
    }
    requests_mock.get(prod_url, json=server_prod_mock_data)

    # Call the catalog_resources method
    test_df = fusion_obj.list_datasets(catalog=new_catalog, max_results=2, contains=["ONE"])
    # Check if the dataframe is created correctly
    pd.testing.assert_frame_equal(test_df, expected_df)

    test_df = fusion_obj.list_datasets(catalog=new_catalog, max_results=2, contains="ONE")
    # Check if the dataframe is created correctly
    pd.testing.assert_frame_equal(test_df, expected_df)

    test_df = fusion_obj.list_datasets(catalog=new_catalog, max_results=2, contains="ONE", id_contains=True)
    # Check if the dataframe is created correctly
    pd.testing.assert_frame_equal(test_df, expected_df)

    test_df = fusion_obj.list_datasets(catalog=new_catalog, max_results=2, product=select_prod)
    # Check if the dataframe is created correctly
    pd.testing.assert_frame_equal(test_df, expected_df)

    test_df = fusion_obj.list_datasets(catalog=new_catalog, max_results=2, status="active")
    # Check if the dataframe is created correctly
    pd.testing.assert_frame_equal(test_df, expected_df)


def test_dataset_resources_success(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    new_catalog = "catalog_id"
    dataset = "my_dataset"
    url = f"{fusion_obj.root_url}catalogs/{new_catalog}/datasets/{dataset}"
    expected_data = {"resources": [{"id": 1, "name": "Resource 1"}, {"id": 2, "name": "Resource 2"}]}
    requests_mock.get(url, json=expected_data)

    # Call the catalog_resources method
    test_df = fusion_obj.dataset_resources(dataset, new_catalog)

    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(expected_data["resources"])
    pd.testing.assert_frame_equal(test_df, expected_df)


def test_list_dataset_attributes(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    new_catalog = "catalog_id"
    dataset = "my_dataset"
    url = f"{fusion_obj.root_url}catalogs/{new_catalog}/datasets/{dataset}/attributes"

    core_cols = [
        "identifier",
        "title",
        "dataType",
        "isDatasetKey",
        "description",
        "source",
    ]

    server_mock_data = {
        "resources": [
            {
                "index": 0,
                "identifier": "attr_1",
                "title": "some title",
                "dataType": "string",
                "other_meta_attr": "some val",
                "status": "active",
            },
            {
                "index": 1,
                "identifier": "attr_2",
                "title": "some title",
                "dataType": "int",
                "other_meta_attr": "some val",
                "status": "active",
            },
        ]
    }
    expected_data = {
        "resources": [
            {
                "identifier": "attr_1",
                "title": "some title",
                "dataType": "string",
            },
            {
                "identifier": "attr_2",
                "title": "some title",
                "dataType": "int",
            },
        ]
    }

    expected_df = pd.DataFrame(expected_data["resources"])

    requests_mock.get(url, json=server_mock_data)

    # Call the catalog_resources method
    test_df = fusion_obj.list_dataset_attributes(dataset, catalog=new_catalog)
    # Check if the dataframe is created correctly
    pd.testing.assert_frame_equal(test_df, expected_df)
    assert all(col in core_cols for col in test_df.columns)

    ext_expected_data = {
        "resources": [
            {
                "index": 0,
                "identifier": "attr_1",
                "title": "some title",
                "dataType": "string",
                "other_meta_attr": "some val",
                "status": "active",
            },
            {
                "index": 1,
                "identifier": "attr_2",
                "title": "some title",
                "dataType": "int",
                "other_meta_attr": "some val",
                "status": "active",
            },
        ]
    }

    ext_expected_df = pd.DataFrame(ext_expected_data["resources"])
    # Call the catalog_resources method
    test_df = fusion_obj.list_dataset_attributes(dataset, catalog=new_catalog, display_all_columns=True)

    # Check if the dataframe is created correctly
    pd.testing.assert_frame_equal(test_df, ext_expected_df)


def test_list_datasetmembers_success(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    new_catalog = "catalog_id"
    dataset = "my_dataset"
    url = f"{fusion_obj.root_url}catalogs/{new_catalog}/datasets/{dataset}/datasetseries"
    expected_data = {"resources": [{"id": 1, "name": "Resource 1"}, {"id": 2, "name": "Resource 2"}]}
    requests_mock.get(url, json=expected_data)

    # Call the catalog_resources method
    test_df = fusion_obj.list_datasetmembers(dataset, new_catalog, max_results=2)

    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(expected_data["resources"])
    pd.testing.assert_frame_equal(test_df, expected_df)


def test_datasetmember_resources_success(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    new_catalog = "catalog_id"
    dataset = "my_dataset"
    series = "2022-02-02"
    url = f"{fusion_obj.root_url}catalogs/{new_catalog}/datasets/{dataset}/datasetseries/{series}"
    expected_data = {"resources": [{"id": 1, "name": "Resource 1"}, {"id": 2, "name": "Resource 2"}]}
    requests_mock.get(url, json=expected_data)

    # Call the catalog_resources method
    test_df = fusion_obj.datasetmember_resources(dataset, series, new_catalog)

    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(expected_data["resources"])
    pd.testing.assert_frame_equal(test_df, expected_df)


def test_list_distributions_success(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    new_catalog = "catalog_id"
    dataset = "my_dataset"
    series = "2022-02-02"
    url = f"{fusion_obj.root_url}catalogs/{new_catalog}/datasets/{dataset}/datasetseries/{series}/distributions"
    expected_data = {"resources": [{"id": 1, "name": "Resource 1"}, {"id": 2, "name": "Resource 2"}]}
    requests_mock.get(url, json=expected_data)

    # Call the catalog_resources method
    test_df = fusion_obj.list_distributions(dataset, series, new_catalog)

    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(expected_data["resources"])
    pd.testing.assert_frame_equal(test_df, expected_df)


def test__resolve_distro_tuples(mocker: MockerFixture, fusion_obj: Fusion) -> None:
    catalog = "my_catalog"
    with (
        pytest.raises(AssertionError),
        mocker.patch.object(fusion_obj, "list_datasetmembers", return_value=pd.DataFrame()),
    ):
        fusion_obj._resolve_distro_tuples("dataset", "catalog", "series")

    valid_ds_members = pd.DataFrame(
        {
            "@id": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "identifier": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "dataset": ["dataset", "dataset", "dataset"],
            "createdDate": ["2020-01-01", "2020-01-02", "2020-01-03"],
        }
    )
    exp_tuples = [
        (catalog, "dataset", "2020-01-01", "parquet"),
        (catalog, "dataset", "2020-01-02", "parquet"),
        (catalog, "dataset", "2020-01-03", "parquet"),
    ]

    with (
        mocker.patch.object(fusion_obj, "list_datasetmembers", return_value=valid_ds_members),
    ):
        res = fusion_obj._resolve_distro_tuples("dataset", catalog=catalog, dt_str="2020-01-01:2020-01-03")
        assert res == exp_tuples
        res = fusion_obj._resolve_distro_tuples("dataset", catalog=catalog, dt_str=":")
        assert res == exp_tuples
        res = fusion_obj._resolve_distro_tuples("dataset", catalog=catalog, dt_str="2020-01-01:2020-01-02")
        assert res == exp_tuples[0:2]
        res = fusion_obj._resolve_distro_tuples("dataset", catalog=catalog, dt_str="2020-01-02:2020-01-03")
        assert res == exp_tuples[1:]
        res = fusion_obj._resolve_distro_tuples("dataset", catalog=catalog)
        assert res == [exp_tuples[-1]]
        res = fusion_obj._resolve_distro_tuples("dataset", catalog=catalog, dt_str="2020-01-03")
        assert res == [exp_tuples[-1]]
        res = fusion_obj._resolve_distro_tuples("dataset", catalog=catalog, dt_str="latest")
        assert res == [exp_tuples[-1]]


def test_to_bytes(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    catalog = "my_catalog"
    dataset = "my_dataset"
    datasetseries = "2020-04-04"
    file_format = "csv"
    url = distribution_to_url(fusion_obj.root_url, catalog, dataset, datasetseries, file_format)
    expected_data = b"some binary data"
    requests_mock.get(url, content=expected_data)

    data = fusion_obj.to_bytes(catalog, dataset, datasetseries, file_format)

    # Check if the data is returned correctly
    assert data.getbuffer() == expected_data


@pytest.mark.skip(reason="MUST FIX")
def test_download_main(mocker: MockerFixture, fusion_obj: Fusion) -> None:
    catalog = "my_catalog"
    dataset = "my_dataset"
    dt_str = "20200101:20200103"
    file_format = "csv"

    dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
    patch_res = [(catalog, dataset, dt, "parquet") for dt in dates]

    mocker.patch.object(
        fusion_obj,
        "_resolve_distro_tuples",
        return_value=patch_res,
    )

    dwn_load_res = [
        (True, f"{fusion_obj.download_folder}/{dataset}__{catalog}__{dt}.{file_format}", None) for dt in dates
    ]
    mocker.patch(
        "fusion.fusion.download_single_file_threading",
        return_value=dwn_load_res,
    )

    mocker.patch("fusion.fusion.stream_single_file_new_session", return_value=dwn_load_res[0])

    res = fusion_obj.download(dataset=dataset, dt_str=dt_str, dataset_format=file_format, catalog=catalog)
    assert not res

    res = fusion_obj.download(
        dataset=dataset, dt_str=dt_str, dataset_format=file_format, catalog=catalog, return_paths=True
    )
    assert res
    assert len(res[0]) == len(dates)

    res = fusion_obj.download(
        dataset=dataset,
        dt_str=dt_str,
        dataset_format=file_format,
        catalog=catalog,
        return_paths=True,
        show_progress=False,
    )
    assert res
    assert len(res) == len(dates)

    res = fusion_obj.download(
        dataset=dataset,
        dt_str=dt_str,
        dataset_format=file_format,
        catalog=catalog,
        return_paths=True,
        partitioning="hive",
    )
    assert res
    assert len(res) == len(dates)

    res = fusion_obj.download(
        dataset=dataset, dt_str="latest", dataset_format=file_format, catalog=catalog, return_paths=True
    )
    assert res
    assert len(res[0]) == len(dates)

    mocker.patch("fusion.fusion.stream_single_file_new_session", return_value=(False, "my_file.dat", "Some Err"))
    res = fusion_obj.download(
        dataset=dataset,
        dt_str=dt_str,
        dataset_format=file_format,
        catalog=catalog,
        return_paths=True,
        show_progress=False,
    )
    assert res
    assert len(res[0]) == len(dates)
    for r in res:
        assert not r[0]

    res = fusion_obj.download(
        dataset=dataset, dt_str="sample", dataset_format=file_format, catalog=catalog, return_paths=True
    )
    assert res
    assert len(res) == 1
    assert res[0][0]
    assert "sample" in res[0][1]


def test_to_df(mocker: MockerFixture, tmp_path: Path, data_table_as_csv: str, fusion_obj: Fusion) -> None:
    catalog = "my_catalog"
    dataset = "my_dataset"
    dates = ["2020-01-01", "2020-01-02", "2020-01-03"]

    files = [f"{tmp_path}/{dataset}__{catalog}__{dt}.csv" for dt in dates]
    for f in files:
        with Path(f).open("w") as f_h:
            f_h.write(data_table_as_csv)

    patch_res = [(True, file, None) for file in files]

    mocker.patch.object(
        fusion_obj,
        "download",
        return_value=patch_res,
    )

    res = fusion_obj.to_df(dataset, f"{dates[0]}:{dates[-1]}", "csv", catalog=catalog)
    assert len(res) > 0


def test_to_table(mocker: MockerFixture, tmp_path: Path, data_table_as_csv: str, fusion_obj: Fusion) -> None:
    catalog = "my_catalog"
    dataset = "my_dataset"
    dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
    fmt = "csv"

    files = [f"{tmp_path}/{dataset}__{catalog}__{dt}.{fmt}" for dt in dates]
    for f in files:
        with Path(f).open("w") as f_h:
            f_h.write(data_table_as_csv)

    patch_res = [(True, file, None) for file in files]

    mocker.patch.object(
        fusion_obj,
        "download",
        return_value=patch_res,
    )

    res = fusion_obj.to_table(dataset, f"{dates[0]}:{dates[-1]}", fmt, catalog=catalog)
    assert len(res) > 0


def test_list_dataset_lineage(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    dataset = "dataset_id"
    catalog = "catalog_id"
    url_dataset = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}"
    requests_mock.get(url_dataset, status_code=200)
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/lineage"
    expected_data = {
        "relations": [
            {
                "source": {"dataset": "source_dataset", "catalog": "source_catalog"},
                "destination": {"dataset": dataset, "catalog": catalog},
            },
            {
                "source": {"dataset": dataset, "catalog": catalog},
                "destination": {"dataset": "destination_dataset", "catalog": "destination_catalog"},
            },
        ],
        "datasets": [
            {"identifier": "source_dataset", "title": "Source Dataset"},
            {"identifier": "destination_dataset", "status": "Active", "title": "Destination Dataset"},
        ],
    }
    requests_mock.get(url, json=expected_data)

    # Call the list_dataset_lineage method
    test_df = fusion_obj.list_dataset_lineage(dataset, catalog=catalog)

    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(
        {
            "type": ["source", "produced"],
            "dataset_identifier": ["source_dataset", "destination_dataset"],
            "title": ["Source Dataset", "Destination Dataset"],
            "catalog": ["source_catalog", "destination_catalog"],
        }
    )
    pd.testing.assert_frame_equal(test_df, expected_df)


def test_list_dataset_lineage_max_results(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    dataset = "dataset_id"
    catalog = "catalog_id"
    url_dataset = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}"
    requests_mock.get(url_dataset, status_code=200)
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/lineage"
    expected_data = {
        "relations": [
            {
                "source": {"dataset": "source_dataset", "catalog": "source_catalog"},
                "destination": {"dataset": dataset, "catalog": catalog},
            },
            {
                "source": {"dataset": dataset, "catalog": catalog},
                "destination": {"dataset": "destination_dataset", "catalog": "destination_catalog"},
            },
        ],
        "datasets": [
            {"identifier": "source_dataset", "status": "Active", "title": "Source Dataset"},
            {"identifier": "destination_dataset", "status": "Active", "title": "Destination Dataset"},
        ],
    }
    requests_mock.get(url, json=expected_data)

    # Call the list_dataset_lineage method
    test_df = fusion_obj.list_dataset_lineage(dataset, catalog=catalog, max_results=1)

    # Check if the dataframe is created correctly
    assert len(test_df) == 1


def test_list_dataset_lineage_resticted(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    dataset_id = "dataset_id"
    catalog = "catalog_id"
    url_dataset = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset_id}"
    requests_mock.get(url_dataset, status_code=200)
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset_id}/lineage"

    expected_data = {
        "relations": [
            {
                "source": {"dataset": "source_dataset", "catalog": "source_catalog"},
                "destination": {"dataset": dataset_id, "catalog": catalog},
            },
            {
                "source": {"dataset": dataset_id, "catalog": catalog},
                "destination": {"dataset": "destination_dataset", "catalog": "destination_catalog"},
            },
        ],
        "datasets": [
            {"identifier": "source_dataset", "status": "Restricted"},
            {"identifier": "destination_dataset", "status": "Active", "title": "Destination Dataset"},
        ],
    }
    requests_mock.get(url, json=expected_data)

    # Call the list_dataset_lineage method
    test_df = fusion_obj.list_dataset_lineage(dataset_id, catalog=catalog)

    # Check if the dataframe is created correctly
    expected_df = pd.DataFrame(
        {
            "type": ["source", "produced"],
            "dataset_identifier": ["Access Restricted", "destination_dataset"],
            "title": ["Access Restricted", "Destination Dataset"],
            "catalog": ["Access Restricted", "destination_catalog"],
        }
    )
    pd.testing.assert_frame_equal(test_df, expected_df)


def test_list_dataset_lineage_dataset_not_found(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    dataset_id = "dataset_id"
    catalog = "catalog_id"
    url_dataset = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset_id}"
    requests_mock.get(url_dataset, status_code=404)

    with pytest.raises(requests.exceptions.HTTPError):
        fusion_obj.list_dataset_lineage(dataset_id, catalog=catalog)


def test_create_dataset_lineage_from_df(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    base_dataset = "base_dataset"
    source_dataset = "source_dataset"
    source_dataset_catalog = "source_catalog"
    catalog = "common"
    status_code = 200
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{base_dataset}/lineage"
    expected_data = {"source": [{"dataset": source_dataset, "catalog": source_dataset_catalog}]}
    requests_mock.post(url, json=expected_data)

    data = [{"dataset": "source_dataset", "catalog": "source_catalog"}]
    df_input = pd.DataFrame(data)

    # Call the create_dataset_lineage method
    resp = fusion_obj.create_dataset_lineage(
        base_dataset=base_dataset,
        source_dataset_catalog_mapping=df_input,
        catalog=catalog, 
        return_resp_obj=True
    )

    # Check if the response is correct
    assert resp is not None
    if resp is not None:
        assert resp.status_code == status_code


def test_create_dataset_lineage_from_list(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    base_dataset = "base_dataset"
    source_dataset = "source_dataset"
    source_dataset_catalog = "source_catalog"
    catalog = "common"
    status_code = 200
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{base_dataset}/lineage"
    expected_data = {"source": [{"dataset": source_dataset, "catalog": source_dataset_catalog}]}
    requests_mock.post(url, json=expected_data)

    data = [{"dataset": "source_dataset", "catalog": "source_catalog"}]

    # Call the create_dataset_lineage method
    resp = fusion_obj.create_dataset_lineage(
        base_dataset=base_dataset,
        source_dataset_catalog_mapping=data,
        catalog=catalog,
        return_resp_obj=True
    )

    # Check if the response is correct
    assert resp is not None
    if resp is not None:
        assert resp.status_code == status_code


def test_create_dataset_lineage_valueerror(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    base_dataset = "base_dataset"
    source_dataset = "source_dataset"
    source_dataset_catalog = "source_catalog"
    catalog = "common"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{base_dataset}/lineage"
    expected_data = {"source": [{"dataset": source_dataset, "catalog": source_dataset_catalog}]}
    requests_mock.post(url, json=expected_data)

    data = {"dataset": "source_dataset", "catalog": "source_catalog"}

    with pytest.raises(
        ValueError, match="source_dataset_catalog_mapping must be a pandas DataFrame or a list of dictionaries."
    ):
        fusion_obj.create_dataset_lineage(
            base_dataset=base_dataset,
            source_dataset_catalog_mapping=data,  # type: ignore
            catalog=catalog
        )


def test_create_dataset_lineage_httperror(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    base_dataset = "base_dataset"
    source_dataset = "source_dataset"
    source_dataset_catalog = "source_catalog"
    catalog = "common"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{base_dataset}/lineage"
    expected_data = {"source": [{"dataset": source_dataset, "catalog": source_dataset_catalog}]}
    data = [{"dataset": "source_dataset", "catalog": "source_catalog"}]
    requests_mock.post(url, status_code=500, json=expected_data)

    with pytest.raises(requests.exceptions.HTTPError):
        fusion_obj.create_dataset_lineage(
            base_dataset=base_dataset,
            source_dataset_catalog_mapping=data,
            catalog=catalog
        )
