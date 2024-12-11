"""Test case for dataset module."""

import json
from collections.abc import Generator
from typing import Any

import pandas as pd
import pytest
import requests
import requests_mock

from fusion import Fusion
from fusion.dataset import Dataset


def test_dataset_class() -> None:
    """Test Dataset class."""
    test_dataset = Dataset(
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


def test_dataset_class_application_id() -> None:
    """Test Dataset class."""
    test_dataset = Dataset(
        identifier="Test Dataset",
        category="Test",
        product="TEST_PRODUCT",
        application_id="12345",
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
    assert test_dataset.application_id == {"id": "12345", "type": "Application (SEAL)"}


def test_dataset_class_application_id_dict() -> None:
    """Test Dataset class."""
    test_dataset = Dataset(
        identifier="Test Dataset",
        category="Test",
        product="TEST_PRODUCT",
        application_id={"id": "12345", "type": "Alternative (TYPE)"},
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
    assert test_dataset.application_id == {"id": "12345", "type": "Alternative (TYPE)"}


def test_dataset_client_value_error() -> None:
    """Test Dataset client value error."""
    my_dataset = Dataset(identifier="Test Dataset")
    with pytest.raises(ValueError, match="A Fusion client object is required.") as error_info:
        my_dataset._use_client(client=None)
    assert str(error_info.value) == "A Fusion client object is required."


def test_dataset_class_from_series() -> None:
    """Test Dataset class."""
    test_dataset = Dataset._from_series(
        pd.Series(
            {
                "title": "Test Dataset",
                "identifier": "Test Dataset",
                "category": "Test",
                "product": "TEST_PRODUCT",
            }
        )
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


def test_dataset_class_from_dict() -> None:
    """Test Dataset class."""
    test_dataset = Dataset._from_dict(
        {
            "title": "Test Dataset",
            "identifier": "Test Dataset",
            "category": "Test",
            "product": "TEST_PRODUCT",
        }
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


def test_dataset_class_from_csv(mock_dataset_pd_read_csv: Generator[pd.DataFrame, Any, None]) -> None:  # noqa: ARG001
    """Test Dataset class."""
    test_dataset = Dataset._from_csv("datasets.csv")

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


def test_dataset_class_from_object_dataset() -> None:
    """Test Dataset from_object Dataset object input."""
    dataset_obj = Dataset(
        title="Test Dataset",
        identifier="Test Dataset",
        category="Test",
        product="TEST_PRODUCT",
    )
    test_dataset = Dataset(identifier="test").from_object(dataset_obj)
    assert str(dataset_obj)
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


def test_dataset_class_from_object_dict() -> None:
    """Test Dataset class."""
    test_dict = {
        "title": "Test Dataset",
        "identifier": "Test Dataset",
        "category": "Test",
        "product": "TEST_PRODUCT",
        "application_id": "12345",
    }
    test_dataset = Dataset(identifier="test").from_object(test_dict)

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
    assert test_dataset.application_id == {"id": "12345", "type": "Application (SEAL)"}


def test_dataset_class_from_object_dict_app_id_dict() -> None:
    """Test Dataset class."""
    test_dict = {
        "title": "Test Dataset",
        "identifier": "Test Dataset",
        "category": "Test",
        "product": "TEST_PRODUCT",
        "application_id": {"id": "12345", "type": "Application (SEAL)"},
    }
    test_dataset = Dataset(identifier="test").from_object(test_dict)

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
    assert test_dataset.application_id == {"id": "12345", "type": "Application (SEAL)"}


def test_dataset_class_from_object_json() -> None:
    """Test Dataset class."""
    test_json = json.dumps(
        {
            "title": "Test Dataset",
            "identifier": "Test Dataset",
            "category": "Test",
            "product": "TEST_PRODUCT",
        }
    )
    test_dataset = Dataset(identifier="test").from_object(test_json)

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


def test_dataset_class_from_object_csv(mock_dataset_pd_read_csv: Generator[pd.DataFrame, Any, None]) -> None:  # noqa: ARG001
    """Test Dataset class."""
    test_dataset = Dataset(identifier="test").from_object("datasets.csv")

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


def test_dataset_class_from_object_series() -> None:
    """Test Dataset class."""
    test_series = pd.Series(
        {
            "title": "Test Dataset",
            "identifier": "Test Dataset",
            "category": "Test",
            "product": "TEST_PRODUCT",
        }
    )
    test_dataset = Dataset(identifier="test").from_object(test_series)

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


def test_dataset_class_from_object_failure() -> None:
    """Test Dataset class."""
    unsupported_obj = 123
    with pytest.raises(TypeError) as error_info:
        Dataset(identifier="test").from_object(unsupported_obj)  # type: ignore
    assert str(error_info.value) == f"Could not resolve the object provided: {unsupported_obj}"


def test_dataset_class_from_catalog(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test list Dataset from_catalog method."""
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets"

    expected_data = {
        "resources": [
            {
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
            },
        ],
    }
    requests_mock.get(url, json=expected_data)

    url2 = f"{fusion_obj.root_url}catalogs/{catalog}/productDatasets"
    expected_data2 = {
        "resources": [
            {"product": "TEST_PRODUCT", "dataset": "TEST_DATASET"},
            {"product": "TEST_PRODUCT2", "dataset": "TEST_DATASET2"},
        ]
    }
    requests_mock.get(url2, json=expected_data2)

    my_dataset = Dataset(identifier="TEST_DATASET").from_catalog(client=fusion_obj, catalog=catalog)
    assert isinstance(my_dataset, Dataset)
    assert my_dataset.title == "Test Dataset"
    assert my_dataset.identifier == "TEST_DATASET"
    assert my_dataset.category == ["category"]
    assert my_dataset.description == "description"
    assert my_dataset.frequency == "Once"
    assert my_dataset.is_internal_only_dataset is False
    assert my_dataset.is_third_party_data is True
    assert my_dataset.is_restricted is False
    assert my_dataset.is_raw_data is True
    assert my_dataset.maintainer == "maintainer"
    assert my_dataset.source == ["source"]
    assert my_dataset.region == ["region"]
    assert my_dataset.publisher == "publisher"
    assert my_dataset.product == ["TEST_PRODUCT"]
    assert my_dataset.sub_category == ["subCategory"]
    assert my_dataset.tags == ["tag1", "tag2"]
    assert my_dataset.created_date == "2020-05-05"
    assert my_dataset.modified_date == "2020-05-05"
    assert my_dataset.delivery_channel == ["API"]
    assert my_dataset.language == "English"
    assert my_dataset.status == "Available"
    assert my_dataset.type_ == "Source"
    assert my_dataset.container_type == "Snapshot-Full"
    assert my_dataset.snowflake == "snowflake"
    assert my_dataset.complexity == "complexity"
    assert my_dataset.is_immutable is False
    assert my_dataset.is_mnpi is False
    assert my_dataset.is_pii is False
    assert my_dataset.is_pci is False
    assert my_dataset.is_client is False
    assert my_dataset.is_public is False
    assert my_dataset.is_internal is False
    assert my_dataset.is_confidential is False
    assert my_dataset.is_highly_confidential is False
    assert my_dataset.is_active is False
    assert isinstance(my_dataset.client, Fusion)


def test_dataset_class_from_catalog_client_implied(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test list Dataset from_catalog method."""
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets"

    expected_data = {
        "resources": [
            {
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
            },
        ],
    }
    requests_mock.get(url, json=expected_data)

    url2 = f"{fusion_obj.root_url}catalogs/{catalog}/productDatasets"
    expected_data2 = {
        "resources": [
            {"product": "TEST_PRODUCT", "dataset": "TEST_DATASET"},
            {"product": "TEST_PRODUCT2", "dataset": "TEST_DATASET2"},
        ]
    }
    requests_mock.get(url2, json=expected_data2)
    my_dataset_id = Dataset(identifier="TEST_DATASET")
    my_dataset_id.client = fusion_obj
    my_dataset = my_dataset_id.from_catalog(catalog=catalog)
    assert isinstance(my_dataset, Dataset)
    assert my_dataset.title == "Test Dataset"
    assert my_dataset.identifier == "TEST_DATASET"
    assert my_dataset.category == ["category"]
    assert my_dataset.description == "description"
    assert my_dataset.frequency == "Once"
    assert my_dataset.is_internal_only_dataset is False
    assert my_dataset.is_third_party_data is True
    assert my_dataset.is_restricted is False
    assert my_dataset.is_raw_data is True
    assert my_dataset.maintainer == "maintainer"
    assert my_dataset.source == ["source"]
    assert my_dataset.region == ["region"]
    assert my_dataset.publisher == "publisher"
    assert my_dataset.product == ["TEST_PRODUCT"]
    assert my_dataset.sub_category == ["subCategory"]
    assert my_dataset.tags == ["tag1", "tag2"]
    assert my_dataset.created_date == "2020-05-05"
    assert my_dataset.modified_date == "2020-05-05"
    assert my_dataset.delivery_channel == ["API"]
    assert my_dataset.language == "English"
    assert my_dataset.status == "Available"
    assert my_dataset.type_ == "Source"
    assert my_dataset.container_type == "Snapshot-Full"
    assert my_dataset.snowflake == "snowflake"
    assert my_dataset.complexity == "complexity"
    assert my_dataset.is_immutable is False
    assert my_dataset.is_mnpi is False
    assert my_dataset.is_pii is False
    assert my_dataset.is_pci is False
    assert my_dataset.is_client is False
    assert my_dataset.is_public is False
    assert my_dataset.is_internal is False
    assert my_dataset.is_confidential is False
    assert my_dataset.is_highly_confidential is False
    assert my_dataset.is_active is False
    assert isinstance(my_dataset._client, Fusion)


def test_dataset_class_from_catalog_no_product(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test list Dataset from_catalog method."""
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets"

    expected_data = {
        "resources": [
            {
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
            },
        ],
    }
    requests_mock.get(url, json=expected_data)

    url2 = f"{fusion_obj.root_url}catalogs/{catalog}/productDatasets"
    expected_data2 = {
        "resources": [
            {"product": "TEST_PRODUCT2", "dataset": "TEST_DATASET2"},
        ]
    }
    requests_mock.get(url2, json=expected_data2)

    my_dataset = Dataset(identifier="TEST_DATASET").from_catalog(client=fusion_obj, catalog=catalog)
    assert isinstance(my_dataset, Dataset)
    assert my_dataset.title == "Test Dataset"
    assert my_dataset.identifier == "TEST_DATASET"
    assert my_dataset.category == ["category"]
    assert my_dataset.description == "description"
    assert my_dataset.frequency == "Once"
    assert my_dataset.is_internal_only_dataset is False
    assert my_dataset.is_third_party_data is True
    assert my_dataset.is_restricted is False
    assert my_dataset.is_raw_data is True
    assert my_dataset.maintainer == "maintainer"
    assert my_dataset.source == ["source"]
    assert my_dataset.region == ["region"]
    assert my_dataset.publisher == "publisher"
    assert my_dataset.product is None
    assert my_dataset.sub_category == ["subCategory"]
    assert my_dataset.tags == ["tag1", "tag2"]
    assert my_dataset.created_date == "2020-05-05"
    assert my_dataset.modified_date == "2020-05-05"
    assert my_dataset.delivery_channel == ["API"]
    assert my_dataset.language == "English"
    assert my_dataset.status == "Available"
    assert my_dataset.type_ == "Source"
    assert my_dataset.container_type == "Snapshot-Full"
    assert my_dataset.snowflake == "snowflake"
    assert my_dataset.complexity == "complexity"
    assert my_dataset.is_immutable is False
    assert my_dataset.is_mnpi is False
    assert my_dataset.is_pii is False
    assert my_dataset.is_pci is False
    assert my_dataset.is_client is False
    assert my_dataset.is_public is False
    assert my_dataset.is_internal is False
    assert my_dataset.is_confidential is False
    assert my_dataset.is_highly_confidential is False
    assert my_dataset.is_active is False


def test_create_dataset_from_dict(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test create Dataset method."""
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
    }
    dataset_obj = Dataset._from_dict(dataset_dict)
    resp = dataset_obj.create(client=fusion_obj, catalog=catalog, return_resp_obj=True)
    status_code = 200
    assert isinstance(resp, requests.Response)
    assert resp.status_code == status_code


def test_update_dataset(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test update Dataset method."""
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}"
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
    }
    requests_mock.put(url, json=expected_data)

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
    }
    dataset_obj = Dataset._from_dict(dataset_dict)
    resp = dataset_obj.update(client=fusion_obj, catalog=catalog, return_resp_obj=True)
    status_code = 200
    assert isinstance(resp, requests.Response)
    assert resp.status_code == status_code


def test_delete_dataset(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test delete Dataset method."""
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}"
    requests_mock.delete(url)

    resp = Dataset(identifier=dataset).delete(client=fusion_obj, catalog=catalog, return_resp_obj=True)
    status_code = 200
    assert isinstance(resp, requests.Response)
    assert resp.status_code == status_code


def test_copy_dataset(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test copy Dataset method."""
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets"

    expected_data = {
        "resources": [
            {
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
            },
        ],
    }
    requests_mock.get(url, json=expected_data)

    url2 = f"{fusion_obj.root_url}catalogs/{catalog}/productDatasets"
    expected_data2 = {
        "resources": [
            {"product": "TEST_PRODUCT", "dataset": "TEST_DATASET"},
            {"product": "TEST_PRODUCT2", "dataset": "TEST_DATASET2"},
        ]
    }
    requests_mock.get(url2, json=expected_data2)
    catalog_new = "catalog_new"
    url3 = f"{fusion_obj.root_url}catalogs/{catalog_new}/datasets/TEST_DATASET"
    expected_data3 = {
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
    }
    requests_mock.post(url3, json=expected_data3)
    resp = Dataset(identifier="TEST_DATASET").copy(
        client=fusion_obj, catalog_from=catalog, catalog_to=catalog_new, return_resp_obj=True
    )
    status_code = 200
    assert isinstance(resp, requests.Response)
    assert resp.status_code == status_code


def test_dataset_case_switching() -> None:
    """Test dataset class case switching."""
    my_dataset = Dataset(
        identifier="TEST_DATASET",
        title="Test Dataset",
        category="Test",
        product="TEST_PRODUCT",
        frequency="Once",
        is_internal_only_dataset=False,
        is_third_party_data=True,
        is_restricted=False,
        is_raw_data=True,
        maintainer="J.P. Morgan Fusion",
        source="source",
        region="region",
        publisher="J.P. Morgan",
        sub_category="subCategory",
        tags="tag1, tag2",
        created_date="2020-05-05",
        modified_date="2020-05-05",
        delivery_channel="API",
        language="English",
        status="Available",
        type_="Source",
        container_type="Snapshot-Full",
        snowflake="snowflake",
    )

    camel_case_dict = my_dataset.to_dict()

    assert camel_case_dict == {
        "identifier": "TEST_DATASET",
        "title": "Test Dataset",
        "category": ["Test"],
        "description": "Test Dataset",
        "frequency": "Once",
        "isInternalOnlyDataset": False,
        "isThirdPartyData": True,
        "isRestricted": False,
        "isRawData": True,
        "maintainer": "J.P. Morgan Fusion",
        "source": ["source"],
        "region": ["region"],
        "publisher": "J.P. Morgan",
        "product": ["TEST_PRODUCT"],
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
        "complexity": None,
        "isImmutable": None,
        "isMnpi": None,
        "isPci": None,
        "isPii": None,
        "isClient": None,
        "isPublic": None,
        "isInternal": None,
        "isConfidential": None,
        "isHighlyConfidential": None,
        "isActive": None,
        "owners": None,
        "applicationId": None,
    }

    dataset_from_camel_dict = Dataset("TEST_DATASET").from_object(camel_case_dict)

    assert dataset_from_camel_dict == my_dataset

    assert dataset_from_camel_dict.sub_category == dataset_from_camel_dict.subCategory
    assert dataset_from_camel_dict.delivery_channel == dataset_from_camel_dict.deliveryChannel
    assert dataset_from_camel_dict.is_internal_only_dataset == dataset_from_camel_dict.isInternalOnlyDataset
    assert dataset_from_camel_dict.is_third_party_data == dataset_from_camel_dict.isThirdPartyData
    assert dataset_from_camel_dict.is_restricted == dataset_from_camel_dict.isRestricted
    assert dataset_from_camel_dict.is_raw_data == dataset_from_camel_dict.isRawData
    assert dataset_from_camel_dict.created_date == dataset_from_camel_dict.createdDate
    assert dataset_from_camel_dict.modified_date == dataset_from_camel_dict.modifiedDate
    assert dataset_from_camel_dict.container_type == dataset_from_camel_dict.containerType
    assert dataset_from_camel_dict.is_immutable == dataset_from_camel_dict.isImmutable
    assert dataset_from_camel_dict.is_mnpi == dataset_from_camel_dict.isMnpi
    assert dataset_from_camel_dict.is_pii == dataset_from_camel_dict.isPii
    assert dataset_from_camel_dict.is_pci == dataset_from_camel_dict.isPci
    assert dataset_from_camel_dict.is_client == dataset_from_camel_dict.isClient
    assert dataset_from_camel_dict.is_public == dataset_from_camel_dict.isPublic
    assert dataset_from_camel_dict.is_internal == dataset_from_camel_dict.isInternal
    assert dataset_from_camel_dict.is_confidential == dataset_from_camel_dict.isConfidential
    assert dataset_from_camel_dict.is_highly_confidential == dataset_from_camel_dict.isHighlyConfidential
    assert dataset_from_camel_dict.is_active == dataset_from_camel_dict.isActive
    assert dataset_from_camel_dict.application_id == dataset_from_camel_dict.applicationId


def test_activate_dataset(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test activate Dataset method."""
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    url_get = f"{fusion_obj.root_url}catalogs/{catalog}/datasets"
    url_put = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}"
    url2 = f"{fusion_obj.root_url}catalogs/{catalog}/productDatasets"

    expected_data_get = {
        "resources": [
            {
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
            },
        ],
    }

    expected_data2 = {
        "resources": [
            {"product": "TEST_PRODUCT", "dataset": "TEST_DATASET"},
        ]
    }
    expected_data_put = expected_data_get["resources"][0]
    expected_data_put["isActive"] = True

    requests_mock.get(url_get, json=expected_data_get)
    requests_mock.put(url_put, json=expected_data_put)
    requests_mock.get(url2, json=expected_data2)

    dataset_obj = Dataset(identifier=dataset)
    resp = dataset_obj.activate(client=fusion_obj, catalog=catalog, return_resp_obj=True)

    request_body = (
        json.loads(resp.request.body.decode("utf-8"))
        if resp and resp.request and isinstance(resp.request.body, bytes)
        else {}
    )

    assert request_body["isActive"] is True


def test_add_to_product(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test add_to_product method."""
    catalog = "my_catalog"
    product = "TEST_PRODUCT"
    dataset = "TEST_DATASET"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/productDatasets"

    expected_data = {
        "product": product,
        "datasets": [dataset],
    }
    requests_mock.put(url, json=expected_data)

    dataset_obj = Dataset(identifier=dataset)
    resp = dataset_obj.add_to_product(product=product, client=fusion_obj, catalog=catalog, return_resp_obj=True)

    request_body = (
        json.loads(resp.request.body.decode("utf-8"))
        if resp and resp.request and isinstance(resp.request.body, bytes)
        else {}
    )

    assert request_body == expected_data
    assert isinstance(resp, requests.Response)
    status_code = 200
    assert resp.status_code == status_code


def test_dataset_getattr_existing_attribute() -> None:
    """Test __getattr__ method for existing attribute."""
    test_dataset = Dataset(
        identifier="Test Dataset",
        category="Test",
        product="TEST_PRODUCT",
    )

    assert test_dataset.title == "Test Dataset"
    assert test_dataset.identifier == "TEST_DATASET"
    assert test_dataset.category == ["Test"]
    assert test_dataset.product == ["TEST_PRODUCT"]


def test_dataset_getattr_non_existing_attribute() -> None:
    """Test __getattr__ method for non-existing attribute."""
    test_dataset = Dataset(
        identifier="Test Dataset",
        category="Test",
        product="TEST_PRODUCT",
    )

    with pytest.raises(AttributeError) as error_info:
        _ = test_dataset.non_existing_attribute
    assert str(error_info.value) == "'Dataset' object has no attribute 'non_existing_attribute'"


def test_remove_from_product(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test remove_from_product method."""
    catalog = "my_catalog"
    product = "TEST_PRODUCT"
    dataset = "TEST_DATASET"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/productDatasets/{product}/{dataset}"

    requests_mock.delete(url)

    dataset_obj = Dataset(identifier=dataset)
    resp = dataset_obj.remove_from_product(product=product, client=fusion_obj, catalog=catalog, return_resp_obj=True)

    assert isinstance(resp, requests.Response)
    status_code = 200
    assert resp.status_code == status_code
