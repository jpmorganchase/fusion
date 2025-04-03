import datetime
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests
import requests_mock
from opensearchpy import OpenSearch
from pytest_mock import MockerFixture

from fusion._fusion import FusionCredentials
from fusion.attributes import Attribute
from fusion.exceptions import CredentialError, FileFormatError
from fusion.fusion import Fusion
from fusion.fusion_types import Types
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


def test_fusion_init_from_path(example_creds_dict: dict[str, Any], tmp_path: Path) -> None:
    credentials_file = tmp_path / "client_credentials_test.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)
    fusion = Fusion(credentials=str(credentials_file))
    assert fusion


def test_fusion_init_cred_value_error(example_creds_dict: dict[str, Any]) -> None:
    with pytest.raises(
        ValueError, match="credentials must be a path to a credentials file or FusionCredentials object"
    ) as error_info:
        Fusion(credentials=example_creds_dict)  # type: ignore
    assert str(error_info.value) == "credentials must be a path to a credentials file or FusionCredentials object"


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


def test_list_datasets_type_filter(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    new_catalog = "catalog_id"
    url = f"{fusion_obj.root_url}catalogs/{new_catalog}/datasets"
    server_mock_data = {
        "resources": [
            {
                "identifier": "ONE",
                "description": "some desc",
                "category": ["FX"],
                "region": ["US"],
                "status": "active",
                "type": "type1",
            },
            {
                "identifier": "TWO",
                "description": "some desc",
                "category": ["FX"],
                "region": ["US", "EU"],
                "status": "inactive",
                "type": "type2",
            },
        ]
    }
    expected_data = {
        "resources": [
            {
                "identifier": "ONE",
                "region": "US",
                "category": "FX",
                "description": "some desc",
                "status": "active",
                "type": "type1",
            }
        ]
    }

    expected_df = pd.DataFrame(expected_data["resources"])

    requests_mock.get(url, json=server_mock_data)

    test_df = fusion_obj.list_datasets(catalog=new_catalog, max_results=2, dataset_type="type1")

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


def test_download_no_access(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    dt_str = "20200101"
    file_format = "csv"

    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}"

    expected_data = {
        "catalog": {
            "@id": "my_catalog/",
            "description": "my catalog",
            "title": "my catalog",
            "identifier": "my_catalog",
        },
        "title": "Test Dataset",
        "identifier": "TEST_DATASET",
        "category": ["category"],
        "shortAbstract": "short abstract",
        "description": "description",
        "frequency": "Once",
        "isInternalOnlyDataset": False,
        "isThirdPartyData": True,
        "isRestricted": False,
        "isRawData": True,
        "maintainer": "maintainer",
        "source": "source",
        "region": ["region"],
        "publisher": "publisher",
        "subCategory": ["subCategory"],
        "tags": ["tag1", "tag2"],
        "createdDate": "2020-05-05",
        "modifiedDate": "2020-05-05",
        "deliveryChannel": ["API"],
        "language": "English",
        "status": "Available",
        "type": "Source",
        "containerType": "Snapshot-Full",
        "snowflake": "snowflake",
        "complexity": "complexity",
        "isImmutable": False,
        "isMnpi": False,
        "isPii": False,
        "isPci": False,
        "isClient": False,
        "isPublic": False,
        "isInternal": False,
        "isConfidential": False,
        "isHighlyConfidential": False,
        "isActive": False,
        "@id": "TEST_DATASET/",
    }

    requests_mock.get(url, json=expected_data)

    with pytest.raises(
        CredentialError, match="You are not subscribed to TEST_DATASET in catalog my_catalog. Please request access."
    ):
        fusion_obj.download(dataset=dataset, dt_str=dt_str, dataset_format=file_format, catalog=catalog)


def test_download_format_not_available(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    dt_str = "20200101"
    file_format = "pdf"

    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}"

    expected_data = {
        "catalog": {
            "@id": "my_catalog/",
            "description": "my catalog",
            "title": "my catalog",
            "identifier": "my_catalog",
        },
        "title": "Test Dataset",
        "identifier": "TEST_DATASET",
        "category": ["category"],
        "shortAbstract": "short abstract",
        "description": "description",
        "frequency": "Once",
        "isInternalOnlyDataset": False,
        "isThirdPartyData": True,
        "isRestricted": False,
        "isRawData": True,
        "maintainer": "maintainer",
        "source": "source",
        "region": ["region"],
        "publisher": "publisher",
        "subCategory": ["subCategory"],
        "tags": ["tag1", "tag2"],
        "createdDate": "2020-05-05",
        "modifiedDate": "2020-05-05",
        "deliveryChannel": ["API"],
        "language": "English",
        "status": "Subscribed",
        "type": "Source",
        "containerType": "Snapshot-Full",
        "snowflake": "snowflake",
        "complexity": "complexity",
        "isImmutable": False,
        "isMnpi": False,
        "isPii": False,
        "isPci": False,
        "isClient": False,
        "isPublic": False,
        "isInternal": False,
        "isConfidential": False,
        "isHighlyConfidential": False,
        "isActive": False,
        "@id": "TEST_DATASET/",
    }

    requests_mock.get(url, json=expected_data)

    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/changes?datasets={dataset}"

    expected_resp = {
        "lastModified": "2025-03-18T09:04:22Z",
        "checksum": "SHA-256=vFdIF:HSLDBV:VBLHD/xe8Mom9yqooZA=-1",
        "metadata": {
            "fields": [
                "lastModified",
                "size",
                "checksum",
                "catalog",
                "dataset",
                "seriesMember",
                "distribution",
                "storageProvider",
                "version",
            ]
        },
        "datasets": [
            {
                "key": "TEST_DATASET",
                "lastModified": "2025-03-18T09:04:22Z",
                "checksum": "SHA-256=vSLKFGNSDFGJBADFGsjfgl/xe8Mom9yqooZA=-1",
                "distributions": [
                    {
                        "key": "TEST_DATASET/20250317/distribution.csv",
                        "values": [
                            "2025-03-18T09:04:22Z",
                            "3054",
                            "SHA-256=vlfaDJFb:VbSdfOHLvnL/xe8Mom9yqooZA=-1",
                            "my_catalog",
                            "TEST_DATASET",
                            "20250317",
                            "csv",
                            "api-bucket",
                            "SJLDHGF;eflSBVLS",
                        ],
                    },
                    {
                        "key": "TEST_DATASET/20250317/distribution.parquet",
                        "values": [
                            "2025-03-18T09:04:19Z",
                            "3076",
                            "SHA-256=7yfQDQq/M1VE4S0SKJDHfblDHFVBldvLXlv5Q=-1",
                            "my_catalog",
                            "TEST_DATASET",
                            "20250317",
                            "parquet",
                            "api-bucket",
                            "SJDFB;IUEBRF;dvbuLSDVc",
                        ],
                    },
                ],
            }
        ],
    }

    requests_mock.get(url, json=expected_resp)

    with pytest.raises(
        FileFormatError,
        match=re.escape(
            "Dataset format pdf is not available for TEST_DATASET in catalog my_catalog. "
            "Available formats are ['csv', 'parquet']."
        ),
    ):
        fusion_obj.download(dataset=dataset, dt_str=dt_str, dataset_format=file_format, catalog=catalog)


def test_download_multiple_format_error(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    dt_str = "20200101"

    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}"

    expected_data = {
        "catalog": {
            "@id": "my_catalog/",
            "description": "my catalog",
            "title": "my catalog",
            "identifier": "my_catalog",
        },
        "title": "Test Dataset",
        "identifier": "TEST_DATASET",
        "category": ["category"],
        "shortAbstract": "short abstract",
        "description": "description",
        "frequency": "Once",
        "isInternalOnlyDataset": False,
        "isThirdPartyData": True,
        "isRestricted": False,
        "isRawData": True,
        "maintainer": "maintainer",
        "source": "source",
        "region": ["region"],
        "publisher": "publisher",
        "subCategory": ["subCategory"],
        "tags": ["tag1", "tag2"],
        "createdDate": "2020-05-05",
        "modifiedDate": "2020-05-05",
        "deliveryChannel": ["API"],
        "language": "English",
        "status": "Subscribed",
        "type": "Source",
        "containerType": "Snapshot-Full",
        "snowflake": "snowflake",
        "complexity": "complexity",
        "isImmutable": False,
        "isMnpi": False,
        "isPii": False,
        "isPci": False,
        "isClient": False,
        "isPublic": False,
        "isInternal": False,
        "isConfidential": False,
        "isHighlyConfidential": False,
        "isActive": False,
        "@id": "TEST_DATASET/",
    }

    requests_mock.get(url, json=expected_data)

    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/changes?datasets={dataset}"

    expected_resp = {
        "lastModified": "2025-03-18T09:04:22Z",
        "checksum": "SHA-256=vFdIF:HSLDBV:VBLHD/xe8Mom9yqooZA=-1",
        "metadata": {
            "fields": [
                "lastModified",
                "size",
                "checksum",
                "catalog",
                "dataset",
                "seriesMember",
                "distribution",
                "storageProvider",
                "version",
            ]
        },
        "datasets": [
            {
                "key": "TEST_DATASET",
                "lastModified": "2025-03-18T09:04:22Z",
                "checksum": "SHA-256=vSLKFGNSDFGJBADFGsjfgl/xe8Mom9yqooZA=-1",
                "distributions": [
                    {
                        "key": "TEST_DATASET/20250317/distribution.csv",
                        "values": [
                            "2025-03-18T09:04:22Z",
                            "3054",
                            "SHA-256=vlfaDJFb:VbSdfOHLvnL/xe8Mom9yqooZA=-1",
                            "my_catalog",
                            "TEST_DATASET",
                            "20250317",
                            "csv",
                            "api-bucket",
                            "SJLDHGF;eflSBVLS",
                        ],
                    },
                    {
                        "key": "TEST_DATASET/20250317/distribution.parquet",
                        "values": [
                            "2025-03-18T09:04:19Z",
                            "3076",
                            "SHA-256=7yfQDQq/M1VE4S0SKJDHfblDHFVBldvLXlv5Q=-1",
                            "my_catalog",
                            "TEST_DATASET",
                            "20250317",
                            "parquet",
                            "api-bucket",
                            "SJDFB;IUEBRF;dvbuLSDVc",
                        ],
                    },
                ],
            }
        ],
    }

    requests_mock.get(url, json=expected_resp)

    with pytest.raises(
        FileFormatError,
        match=re.escape(
            "Multiple formats found for TEST_DATASET in catalog my_catalog. Dataset format is required to"
            "download. Available formats are ['csv', 'parquet']."
        ),
    ):
        fusion_obj.download(dataset=dataset, dt_str=dt_str, dataset_format=None, catalog=catalog)


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
        base_dataset=base_dataset, source_dataset_catalog_mapping=df_input, catalog=catalog, return_resp_obj=True
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
        base_dataset=base_dataset, source_dataset_catalog_mapping=data, catalog=catalog, return_resp_obj=True
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
            catalog=catalog,
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
            base_dataset=base_dataset, source_dataset_catalog_mapping=data, catalog=catalog
        )


def test_list_product_dataset_mapping_dataset_list(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test list Product Dataset Mapping method."""
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/productDatasets"
    expected_data = {
        "resources": [
            {"product": "P00001", "dataset": "D00001"},
            {"product": "P00002", "dataset": "D00002"},
        ]
    }
    requests_mock.get(url, json=expected_data)

    resp = fusion_obj.list_product_dataset_mapping(dataset=["D00001"], catalog=catalog)
    assert all(resp == pd.DataFrame({"product": ["P00001"], "dataset": ["D00001"]}))


def test_list_product_dataset_mapping_dataset_str(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test list Product Dataset Mapping method."""
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/productDatasets"
    expected_data = {
        "resources": [
            {"product": "P00001", "dataset": "D00001"},
            {"product": "P00002", "dataset": "D00002"},
        ]
    }
    requests_mock.get(url, json=expected_data)

    resp = fusion_obj.list_product_dataset_mapping(dataset="D00001", catalog=catalog)
    assert all(resp == pd.DataFrame({"product": ["P00001"], "dataset": ["D00001"]}))


def test_list_product_dataset_mapping_product_str(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test list Product Dataset Mapping method."""
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/productDatasets"
    expected_data = {
        "resources": [
            {"product": "P00001", "dataset": "D00001"},
            {"product": "P00002", "dataset": "D00002"},
        ]
    }
    requests_mock.get(url, json=expected_data)

    resp = fusion_obj.list_product_dataset_mapping(product="P00001", catalog=catalog)
    assert all(resp == pd.DataFrame({"product": ["P00001"], "dataset": ["D00001"]}))


def test_list_product_dataset_mapping_product_list(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test list Product Dataset Mapping method."""
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/productDatasets"
    expected_data = {
        "resources": [
            {"product": "P00001", "dataset": "D00001"},
            {"product": "P00002", "dataset": "D00002"},
        ]
    }
    requests_mock.get(url, json=expected_data)

    resp = fusion_obj.list_product_dataset_mapping(product=["P00001"], catalog=catalog)
    assert all(resp == pd.DataFrame({"product": ["P00001"], "dataset": ["D00001"]}))


def test_list_product_dataset_mapping_product_no_filter(
    requests_mock: requests_mock.Mocker, fusion_obj: Fusion
) -> None:
    """Test list Product Dataset Mapping method."""
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/productDatasets"
    expected_data = {"resources": [{"product": "P00001", "dataset": "D00001"}]}
    requests_mock.get(url, json=expected_data)

    resp = fusion_obj.list_product_dataset_mapping(catalog=catalog)
    assert all(resp == pd.DataFrame({"product": ["P00001"], "dataset": ["D00001"]}))


def test_fusion_product(fusion_obj: Fusion) -> None:
    """Test Fusion Product class from client."""
    test_product = fusion_obj.product(title="Test Product", identifier="Test Product", releaseDate="May 5, 2020")
    assert test_product.title == "Test Product"
    assert test_product.identifier == "TEST_PRODUCT"
    assert test_product.category is None
    assert test_product.shortAbstract == "Test Product"
    assert test_product.description == "Test Product"
    assert test_product.isActive is True
    assert test_product.isRestricted is None
    assert test_product.maintainer is None
    assert test_product.region == ["Global"]
    assert test_product.publisher == "J.P. Morgan"
    assert test_product.subCategory is None
    assert test_product.tag is None
    assert test_product.deliveryChannel == ["API"]
    assert test_product.theme is None
    assert test_product.releaseDate == "2020-05-05"
    assert test_product.language == "English"
    assert test_product.status == "Available"
    assert test_product.image == ""
    assert test_product.logo == ""
    assert test_product.dataset is None
    assert test_product._client == fusion_obj


def test_fusion_dataset(fusion_obj: Fusion) -> None:
    """Test Fusion Dataset class from client"""
    test_dataset = fusion_obj.dataset(
        title="Test Dataset",
        identifier="Test Dataset",
        category="Test",
        product="TEST_PRODUCT",
    )

    assert str(test_dataset)
    assert repr(test_dataset)
    assert test_dataset.title == "Test Dataset"
    assert test_dataset.identifier == "TEST_DATASET"
    assert test_dataset.category == ["Test"]
    assert test_dataset.description == "Test Dataset"
    assert test_dataset.frequency == "Once"
    assert test_dataset.is_internal_only_dataset is False
    assert test_dataset.is_third_party_data is True
    assert test_dataset.is_restricted is None
    assert test_dataset.is_raw_data is True
    assert test_dataset.maintainer == "J.P. Morgan Fusion"
    assert test_dataset.source is None
    assert test_dataset.region is None
    assert test_dataset.publisher == "J.P. Morgan"
    assert test_dataset.product == ["TEST_PRODUCT"]
    assert test_dataset.sub_category is None
    assert test_dataset.tags is None
    assert test_dataset.created_date is None
    assert test_dataset.modified_date is None
    assert test_dataset.delivery_channel == ["API"]
    assert test_dataset.language == "English"
    assert test_dataset.status == "Available"
    assert test_dataset.type_ == "Source"
    assert test_dataset.container_type == "Snapshot-Full"
    assert test_dataset.snowflake is None
    assert test_dataset.complexity is None
    assert test_dataset.is_immutable is None
    assert test_dataset.is_mnpi is None
    assert test_dataset.is_pii is None
    assert test_dataset.is_pci is None
    assert test_dataset.is_client is None
    assert test_dataset.is_public is None
    assert test_dataset.is_internal is None
    assert test_dataset.is_confidential is None
    assert test_dataset.is_highly_confidential is None
    assert test_dataset.is_active is None
    assert test_dataset.client == fusion_obj


def test_fusion_attribute(fusion_obj: Fusion) -> None:
    """Test Fusion Attribute class from client."""
    test_attribute = fusion_obj.attribute(
        title="Test Attribute",
        identifier="Test Attribute",
        index=0,
        isDatasetKey=True,
        dataType="String",
        availableFrom="May 5, 2020",
    )
    assert str(test_attribute)
    assert repr(test_attribute)
    assert test_attribute.title == "Test Attribute"
    assert test_attribute.identifier == "test_attribute"
    assert test_attribute.index == 0
    assert test_attribute.isDatasetKey
    assert test_attribute.dataType == Types.String
    assert test_attribute.description == "Test Attribute"
    assert test_attribute.source is None
    assert test_attribute.sourceFieldId == "test_attribute"
    assert test_attribute.isInternalDatasetKey is None
    assert test_attribute.isExternallyVisible is True
    assert test_attribute.unit is None
    assert test_attribute.multiplier == 1.0
    assert test_attribute.isMetric is None
    assert test_attribute.isPropagationEligible is None
    assert test_attribute.availableFrom == "2020-05-05"
    assert test_attribute.deprecatedFrom is None
    assert test_attribute.term == "bizterm1"
    assert test_attribute.dataset is None
    assert test_attribute.attributeType is None
    assert test_attribute._client == fusion_obj


def test_fusion_attributes(fusion_obj: Fusion) -> None:
    """Test Fusion Attributes class from client."""
    test_attributes = fusion_obj.attributes(
        [
            Attribute(
                title="Test Attribute",
                identifier="Test Attribute",
                index=0,
                is_dataset_key=True,
                data_type=Types.String,
                available_from="May 5, 2020",
            )
        ]
    )
    assert str(test_attributes)
    assert repr(test_attributes)
    assert test_attributes.attributes[0].title == "Test Attribute"
    assert test_attributes.attributes[0].identifier == "test_attribute"
    assert test_attributes.attributes[0].index == 0
    assert test_attributes.attributes[0].isDatasetKey
    assert test_attributes.attributes[0].dataType == Types.String
    assert test_attributes.attributes[0].description == "Test Attribute"
    assert test_attributes.attributes[0].source is None
    assert test_attributes.attributes[0].sourceFieldId == "test_attribute"
    assert test_attributes.attributes[0].isInternalDatasetKey is None
    assert test_attributes.attributes[0].isExternallyVisible is True
    assert test_attributes.attributes[0].unit is None
    assert test_attributes.attributes[0].multiplier == 1.0
    assert test_attributes.attributes[0].isMetric is None
    assert test_attributes.attributes[0].isPropagationEligible is None
    assert test_attributes.attributes[0].availableFrom == "2020-05-05"
    assert test_attributes.attributes[0].deprecatedFrom is None
    assert test_attributes.attributes[0].term == "bizterm1"
    assert test_attributes.attributes[0].dataset is None
    assert test_attributes.attributes[0].attributeType is None
    assert test_attributes._client == fusion_obj


def test_fusion_create_product(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test create product from client."""
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/products/TEST_PRODUCT"
    expected_data = {
        "title": "Test Product",
        "identifier": "TEST_PRODUCT",
        "category": ["category"],
        "shortAbstract": "short abstract",
        "description": "description",
        "isActive": True,
        "isRestricted": False,
        "maintainer": ["maintainer"],
        "region": ["region"],
        "publisher": "publisher",
        "subCategory": ["subCategory"],
        "tag": ["tag1", "tag2"],
        "deliveryChannel": ["API"],
        "theme": "theme",
        "releaseDate": "2020-05-05",
        "language": "English",
        "status": "Available",
        "image": "",
        "logo": "",
    }
    requests_mock.post(url, json=expected_data)

    my_product = fusion_obj.product(
        title="Test Product",
        identifier="TEST_PRODUCT",
        category=["category"],
        shortAbstract="short abstract",
        description="description",
        isActive=True,
        isRestricted=False,
        maintainer=["maintainer"],
        region=["region"],
        publisher="publisher",
        subCategory=["subCategory"],
        tag=["tag1", "tag2"],
        deliveryChannel=["API"],
        theme="theme",
        releaseDate="2020-05-05",
        language="English",
        status="Available",
        image="",
        logo="",
    )
    status_code = 200
    resp = my_product.create(catalog=catalog, client=fusion_obj, return_resp_obj=True)
    assert isinstance(resp, requests.models.Response)
    assert resp.status_code == status_code


def test_fusion_create_dataset_dict(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test create dataset from client."""
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/TEST_DATASET"
    expected_data = {
        "title": "Test Dataset",
        "identifier": "TEST_DATASET",
        "category": ["category"],
        "shortAbstract": "short abstract",
        "description": "description",
        "frequency": "Once",
        "isInternalOnlyDataset": False,
        "isThirdPartyData": True,
        "isRestricted": False,
        "isRawData": False,
        "maintainer": "maintainer",
        "source": "source",
        "region": ["region"],
        "publisher": "publisher",
        "subCategory": ["subCategory"],
        "tags": ["tag1", "tag2"],
        "createdDate": "2020-05-05",
        "modifiedDate": "2020-05-05",
        "deliveryChannel": ["API"],
        "language": "English",
        "status": "Available",
        "type": "Source",
        "containerType": "Snapshot-Full",
        "snowflake": "snowflake",
        "complexity": "complexity",
        "isImmutable": False,
        "isMnpi": False,
        "isPii": False,
        "isPci": False,
        "isClient": False,
        "isPublic": False,
        "isInternal": False,
        "isConfidential": False,
        "isHighlyConfidential": False,
        "isActive": False,
    }
    requests_mock.post(url, json=expected_data)

    dataset_dict = {
        "title": "Test Dataset",
        "identifier": "TEST_DATASET",
        "category": ["category"],
        "shortAbstract": "short abstract",
        "description": "description",
        "frequency": "Once",
        "isInternalOnlyDataset": False,
        "isThirdPartyData": True,
        "isRestricted": False,
        "isRawData": False,
        "maintainer": "maintainer",
        "source": "source",
        "region": ["region"],
        "publisher": "publisher",
        "subCategory": ["subCategory"],
        "tags": ["tag1", "tag2"],
        "createdDate": "2020-05-05",
        "modifiedDate": "2020-05-05",
        "deliveryChannel": ["API"],
        "language": "English",
        "status": "Available",
        "type": "Source",
        "containerType": "Snapshot-Full",
        "snowflake": "snowflake",
        "complexity": "complexity",
        "isImmutable": False,
        "isMnpi": False,
        "isPii": False,
        "isPci": False,
        "isClient": False,
        "isPublic": False,
        "isInternal": False,
        "isConfidential": False,
        "isHighlyConfidential": False,
        "isActive": False,
    }
    dataset_obj = fusion_obj.dataset(identifier="TEST_DATASET").from_object(dataset_dict)
    resp = dataset_obj.create(client=fusion_obj, catalog=catalog, return_resp_obj=True)
    status_code = 200
    assert isinstance(resp, requests.models.Response)
    assert resp.status_code == status_code


def test_fusion_create_attributes(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test create attributes from client."""
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/attributes"

    expected_data = {
        "attributes": [
            {
                "title": "Test Attribute",
                "identifier": "Test Attribute",
                "index": 0,
                "isDatasetKey": True,
                "dataType": "string",
                "description": "Test Attribute",
                "source": None,
                "sourceFieldId": "test_attribute",
                "isInternalDatasetKey": None,
                "isExternallyVisible": True,
                "unit": None,
                "multiplier": 1.0,
                "isMetric": None,
                "isPropagationEligible": None,
                "availableFrom": "2020-05-05",
                "deprecatedFrom": None,
                "term": "bizterm1",
                "dataset": None,
                "attributeType": None,
            }
        ]
    }

    requests_mock.put(url, json=expected_data)

    test_attributes = fusion_obj.attributes(
        [
            fusion_obj.attribute(
                title="Test Attribute",
                identifier="Test Attribute",
                index=0,
                isDatasetKey=True,
                dataType="String",
                availableFrom="May 5, 2020",
            )
        ]
    )
    resp = test_attributes.create(client=fusion_obj, catalog=catalog, dataset=dataset, return_resp_obj=True)
    status_code = 200
    assert isinstance(resp, requests.Response)
    assert resp.status_code == status_code


def test_fusion_delete_datasetmembers(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test delete datasetmembers"""
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    datasetseries = "20200101"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{datasetseries}"
    requests_mock.delete(url, status_code=200)

    resp = fusion_obj.delete_datasetmembers(dataset, datasetseries, catalog=catalog, return_resp_obj=True)
    status_code = 200
    assert resp is not None
    assert isinstance(resp[0], requests.Response)
    assert resp[0].status_code == status_code
    resp_len = 1
    assert len(resp) == resp_len


def test_fusion_delete_datasetmembers_multiple(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test delete datasetmembers"""
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    datasetseries = ["20200101", "20200101"]
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{datasetseries[0]}"
    requests_mock.delete(url, status_code=200)

    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{datasetseries[1]}"
    requests_mock.delete(url, status_code=200)

    resp = fusion_obj.delete_datasetmembers(dataset, datasetseries, catalog=catalog, return_resp_obj=True)
    status_code = 200
    assert resp is not None
    assert isinstance(resp[0], requests.Response)
    assert resp[0].status_code == status_code
    assert isinstance(resp[1], requests.Response)
    assert resp[1].status_code == status_code
    resp_len = 2
    assert len(resp) == resp_len


def test_fusion_delete_all_datasetmembers(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test delete datasetmembers"""
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries"
    requests_mock.delete(url, status_code=200)

    resp = fusion_obj.delete_all_datasetmembers(dataset, catalog=catalog, return_resp_obj=True)
    status_code = 200
    assert resp is not None
    assert isinstance(resp, requests.Response)
    assert resp.status_code == status_code


def test_list_registered_attributes(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test list registered attributes."""
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/attributes"
    core_cols = [
        "identifier",
        "title",
        "dataType",
        "publisher",
        "description",
        "applicationId",
    ]

    server_mock_data = {
        "resources": [
            {
                "identifier": "attr_1",
                "title": "some title",
                "dataType": "string",
                "publisher": "J.P Morgan",
                "applicationId": {"id": "12345", "type": "application"},
                "catalog": {"@id": "12345/", "description": "catalog"},
            },
            {
                "identifier": "attr_2",
                "title": "some title",
                "dataType": "int",
                "publisher": "J.P Morgan",
                "applicationId": {"id": "12345", "type": "application"},
                "catalog": {"@id": "12345/", "description": "catalog"},
            },
        ]
    }
    expected_data = {
        "resources": [
            {
                "identifier": "attr_1",
                "title": "some title",
                "dataType": "string",
                "publisher": "J.P Morgan",
                "applicationId": {"id": "12345", "type": "application"},
            },
            {
                "identifier": "attr_2",
                "title": "some title",
                "dataType": "int",
                "publisher": "J.P Morgan",
                "applicationId": {"id": "12345", "type": "application"},
            },
        ]
    }

    expected_df = pd.DataFrame(expected_data["resources"])

    requests_mock.get(url, json=server_mock_data)

    # Call the catalog_resources method
    test_df = fusion_obj.list_registered_attributes(catalog=catalog)
    # Check if the dataframe is created correctly
    pd.testing.assert_frame_equal(test_df, expected_df)
    assert all(col in core_cols for col in test_df.columns)


def test_fusion_report(fusion_obj: Fusion) -> None:
    """Test Fusion Report class from client"""
    test_report = fusion_obj.report(
        title="Test Report", identifier="Test Report", category="Test", application_id="12345", report={"tier": "tier"}
    )

    assert str(test_report)
    assert repr(test_report)
    assert test_report.title == "Test Report"
    assert test_report.identifier == "TEST_REPORT"
    assert test_report.category == ["Test"]
    assert test_report.description == "Test Report"
    assert test_report.frequency == "Once"
    assert test_report.is_internal_only_dataset is False
    assert test_report.is_third_party_data is True
    assert test_report.is_restricted is None
    assert test_report.is_raw_data is True
    assert test_report.maintainer == "J.P. Morgan Fusion"
    assert test_report.source is None
    assert test_report.region is None
    assert test_report.publisher == "J.P. Morgan"
    assert test_report.product is None
    assert test_report.sub_category is None
    assert test_report.tags is None
    assert test_report.created_date is None
    assert test_report.modified_date is None
    assert test_report.delivery_channel == ["API"]
    assert test_report.language == "English"
    assert test_report.status == "Available"
    assert test_report.type_ == "Report"
    assert test_report.container_type == "Snapshot-Full"
    assert test_report.snowflake is None
    assert test_report.complexity is None
    assert test_report.is_immutable is None
    assert test_report.is_mnpi is None
    assert test_report.is_pii is None
    assert test_report.is_pci is None
    assert test_report.is_client is None
    assert test_report.is_public is None
    assert test_report.is_internal is None
    assert test_report.is_confidential is None
    assert test_report.is_highly_confidential is None
    assert test_report.is_active is None
    assert test_report.client == fusion_obj
    assert test_report.application_id == {"id": "12345", "type": "Application (SEAL)"}
    assert test_report.report == {"tier": "tier"}
    assert test_report._client == fusion_obj
    assert test_report.owners is None


def test_fusion_input_dataflow(fusion_obj: Fusion) -> None:
    """Test Fusion Input Dataflow class from client"""
    test_input_dataflow = fusion_obj.input_dataflow(
        title="Test Input Dataflow",
        identifier="Test Input Dataflow",
        category="Test",
        application_id="12345",
        producer_application_id={"id": "12345", "type": "Application (SEAL)"},
        consumer_application_id={"id": "12345", "type": "Application (SEAL)"},
    )

    assert str(test_input_dataflow)
    assert repr(test_input_dataflow)
    assert test_input_dataflow.title == "Test Input Dataflow"
    assert test_input_dataflow.identifier == "TEST_INPUT_DATAFLOW"
    assert test_input_dataflow.category == ["Test"]
    assert test_input_dataflow.description == "Test Input Dataflow"
    assert test_input_dataflow.frequency == "Once"
    assert test_input_dataflow.is_internal_only_dataset is False
    assert test_input_dataflow.is_third_party_data is True
    assert test_input_dataflow.is_restricted is None
    assert test_input_dataflow.is_raw_data is True
    assert test_input_dataflow.maintainer == "J.P. Morgan Fusion"
    assert test_input_dataflow.source is None
    assert test_input_dataflow.region is None
    assert test_input_dataflow.publisher == "J.P. Morgan"
    assert test_input_dataflow.product is None
    assert test_input_dataflow.sub_category is None
    assert test_input_dataflow.tags is None
    assert test_input_dataflow.created_date is None
    assert test_input_dataflow.modified_date is None
    assert test_input_dataflow.delivery_channel == ["API"]
    assert test_input_dataflow.language == "English"
    assert test_input_dataflow.status == "Available"
    assert test_input_dataflow.type_ == "Flow"
    assert test_input_dataflow.container_type == "Snapshot-Full"
    assert test_input_dataflow.snowflake is None
    assert test_input_dataflow.complexity is None
    assert test_input_dataflow.is_immutable is None
    assert test_input_dataflow.is_mnpi is None
    assert test_input_dataflow.is_pii is None
    assert test_input_dataflow.is_pci is None
    assert test_input_dataflow.is_client is None
    assert test_input_dataflow.is_public is None
    assert test_input_dataflow.is_internal is None
    assert test_input_dataflow.is_confidential is None
    assert test_input_dataflow.is_highly_confidential is None
    assert test_input_dataflow.is_active is None
    assert test_input_dataflow.client == fusion_obj
    assert test_input_dataflow.application_id == {"id": "12345", "type": "Application (SEAL)"}
    assert test_input_dataflow.producer_application_id == {"id": "12345", "type": "Application (SEAL)"}
    assert test_input_dataflow.consumer_application_id == [{"id": "12345", "type": "Application (SEAL)"}]
    assert test_input_dataflow.flow_details == {"flowDirection": "Input"}


def test_fusion_output_dataflow(fusion_obj: Fusion) -> None:
    """Test Fusion Output Dataflow class from client"""
    test_output_dataflow = fusion_obj.output_dataflow(
        title="Test Output Dataflow",
        identifier="Test Output Dataflow",
        category="Test",
        application_id="12345",
        producer_application_id={"id": "12345", "type": "Application (SEAL)"},
        consumer_application_id={"id": "12345", "type": "Application (SEAL)"},
    )

    assert str(test_output_dataflow)
    assert repr(test_output_dataflow)
    assert test_output_dataflow.title == "Test Output Dataflow"
    assert test_output_dataflow.identifier == "TEST_OUTPUT_DATAFLOW"
    assert test_output_dataflow.category == ["Test"]
    assert test_output_dataflow.description == "Test Output Dataflow"
    assert test_output_dataflow.frequency == "Once"
    assert test_output_dataflow.is_internal_only_dataset is False
    assert test_output_dataflow.is_third_party_data is True
    assert test_output_dataflow.is_restricted is None
    assert test_output_dataflow.is_raw_data is True
    assert test_output_dataflow.maintainer == "J.P. Morgan Fusion"
    assert test_output_dataflow.source is None
    assert test_output_dataflow.region is None
    assert test_output_dataflow.publisher == "J.P. Morgan"
    assert test_output_dataflow.product is None
    assert test_output_dataflow.sub_category is None
    assert test_output_dataflow.tags is None
    assert test_output_dataflow.created_date is None
    assert test_output_dataflow.modified_date is None
    assert test_output_dataflow.delivery_channel == ["API"]
    assert test_output_dataflow.language == "English"
    assert test_output_dataflow.status == "Available"
    assert test_output_dataflow.type_ == "Flow"
    assert test_output_dataflow.container_type == "Snapshot-Full"
    assert test_output_dataflow.snowflake is None
    assert test_output_dataflow.complexity is None
    assert test_output_dataflow.is_immutable is None
    assert test_output_dataflow.is_mnpi is None
    assert test_output_dataflow.is_pii is None
    assert test_output_dataflow.is_pci is None
    assert test_output_dataflow.is_client is None
    assert test_output_dataflow.is_public is None
    assert test_output_dataflow.is_internal is None
    assert test_output_dataflow.is_confidential is None
    assert test_output_dataflow.application_id == {"id": "12345", "type": "Application (SEAL)"}
    assert test_output_dataflow.producer_application_id == {"id": "12345", "type": "Application (SEAL)"}
    assert test_output_dataflow.consumer_application_id == [{"id": "12345", "type": "Application (SEAL)"}]
    assert test_output_dataflow.flow_details == {"flowDirection": "Output"}
    assert test_output_dataflow.client == fusion_obj


def test_list_indexes_summary(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test list indexes from client."""
    catalog = "my_catalog"
    knowledge_base = "MY_KB"
    url = f"{fusion_obj.root_url}dataspaces/{catalog}/datasets/{knowledge_base}/indexes/"
    expected_data = [
        {
            "settings": {
                "index": {
                    "knn": "true",
                    "creation_date": "2020-05-05",
                    "number_of_shards": 1,
                    "number_of_replicas": 1,
                    "provided_name": "dataspace-mydataspace-dataset-mydataset-index-myindex",
                }
            },
            "mappings": {
                "properties": {
                    "chunk-id": {"type": "text"},
                    "vector": {"type": "knn_vector", "dimension": 1536},
                    "id": {"type": "text", "fields": '{"keyword": {"type": "keyword", "ignore_above": 256}}'},
                    "content": {"type": "text"},
                }
            },
        }
    ]
    requests_mock.get(url, json=expected_data)

    resp = fusion_obj.list_indexes(catalog=catalog, knowledge_base=knowledge_base)
    exp_df = (
        pd.DataFrame(
            {
                "index_name": ["myindex"],
                "vector_field_name": ["vector"],
                "vector_dimension": [1536],
            }
        )
        .set_index("index_name")
        .transpose()
    )

    assert all(resp == exp_df)


def test_list_indexes_full(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test list indexes from client."""
    catalog = "my_catalog"
    knowledge_base = "MY_KB"
    url = f"{fusion_obj.root_url}dataspaces/{catalog}/datasets/{knowledge_base}/indexes/"
    expected_data = [
        {
            "settings": {
                "index": {
                    "knn": "true",
                    "creation_date": "1737647317617",
                    "number_of_shards": 1,
                    "number_of_replicas": 1,
                    "provided_name": "dataspace-mydataspace-dataset-mydataset-index-myindex",
                }
            },
            "mappings": {
                "properties": {
                    "chunk-id": {"type": "text"},
                    "vector": {"type": "knn_vector", "dimension": 1536},
                    "id": {"type": "text", "fields": '{"keyword": {"type": "keyword", "ignore_above": 256}}'},
                    "content": {"type": "text"},
                }
            },
        }
    ]
    requests_mock.get(url, json=expected_data)

    resp = fusion_obj.list_indexes(catalog=catalog, knowledge_base=knowledge_base, show_details=True)

    df_resp = pd.json_normalize(expected_data)
    df2 = df_resp.transpose()
    df2.index = df2.index.map(str)
    df2.columns = pd.Index(df2.loc["settings.index.provided_name"])
    df2 = df2.rename(columns=lambda x: x.split("index-")[-1])
    df2.columns.names = ["index_name"]
    df2.loc["settings.index.creation_date"] = pd.to_datetime(df2.loc["settings.index.creation_date"], unit="ms")
    multi_index = [index.split(".", 1) for index in df2.index]
    df2.index = pd.MultiIndex.from_tuples(multi_index)

    assert all(resp == df2)


def test_get_fusion_vector_store_client(fusion_obj: Fusion) -> None:
    """Test the get_fusion_vector_store_client method."""

    result = fusion_obj.get_fusion_vector_store_client("knowledge_base")

    assert isinstance(result, OpenSearch)


def test_list_datasetmembers_distributions(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test list_datasetmembers_distributions method."""
    catalog = "my_catalog"
    dataset = "MY_DATASET"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/changes?datasets={dataset}"
    expected_resp = {
        "lastModified": "2025-03-18T09:04:22Z",
        "checksum": "SHA-256=vFdIF:HSLDBV:VBLHD/xe8Mom9yqooZA=-1",
        "metadata": {
            "fields": [
                "lastModified",
                "size",
                "checksum",
                "catalog",
                "dataset",
                "seriesMember",
                "distribution",
                "storageProvider",
                "version",
            ]
        },
        "datasets": [
            {
                "key": "MY_DATASET",
                "lastModified": "2025-03-18T09:04:22Z",
                "checksum": "SHA-256=vSLKFGNSDFGJBADFGsjfgl/xe8Mom9yqooZA=-1",
                "distributions": [
                    {
                        "key": "MY_DATASET/20250317/distribution.csv",
                        "values": [
                            "2025-03-18T09:04:22Z",
                            "3054",
                            "SHA-256=vlfaDJFb:VbSdfOHLvnL/xe8Mom9yqooZA=-1",
                            "my_catalog",
                            "MY_DATASET",
                            "20250317",
                            "csv",
                            "api-bucket",
                            "SJLDHGF;eflSBVLS",
                        ],
                    },
                    {
                        "key": "MY_DATASET/20250317/distribution.parquet",
                        "values": [
                            "2025-03-18T09:04:19Z",
                            "3076",
                            "SHA-256=7yfQDQq/M1VE4S0SKJDHfblDHFVBldvLXlv5Q=-1",
                            "my_catalog",
                            "MY_DATASET",
                            "20250317",
                            "parquet",
                            "api-bucket",
                            "SJDFB;IUEBRF;dvbuLSDVc",
                        ],
                    },
                ],
            }
        ],
    }

    requests_mock.get(url, json=expected_resp)

    expected_data = [("20250317", "csv"), ("20250317", "parquet")]
    expected_df = pd.DataFrame(expected_data, columns=["identifier", "format"])

    resp = fusion_obj.list_datasetmembers_distributions(catalog=catalog, dataset=dataset)

    assert all(resp == expected_df)
