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
    assert test_dataset.isInternalOnlyDataset is False
    assert test_dataset.isThirdPartyData is True
    assert test_dataset.isRestricted is None
    assert test_dataset.isRawData is True
    assert test_dataset.maintainer == "J.P. Morgan Fusion"
    assert test_dataset.source is None
    assert test_dataset.region is None
    assert test_dataset.publisher == "J.P. Morgan"
    assert test_dataset.product == ["TEST_PRODUCT"]
    assert test_dataset.subCategory is None
    assert test_dataset.tags is None
    assert test_dataset.createdDate is None
    assert test_dataset.modifiedDate is None
    assert test_dataset.deliveryChannel == ["API"]
    assert test_dataset.language == "English"
    assert test_dataset.status == "Available"
    assert test_dataset.type_ == "Source"
    assert test_dataset.containerType == "Snapshot-Full"
    assert test_dataset.snowflake is None
    assert test_dataset.complexity is None
    assert test_dataset.isImmutable is None
    assert test_dataset.isMnpi is None
    assert test_dataset.isPii is None
    assert test_dataset.isPci is None
    assert test_dataset.isClient is None
    assert test_dataset.isPublic is None
    assert test_dataset.isInternal is None
    assert test_dataset.isConfidential is None
    assert test_dataset.isHighlyConfidential is None
    assert test_dataset.isActive is None


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
    assert test_dataset.isInternalOnlyDataset is False
    assert test_dataset.isThirdPartyData is True
    assert test_dataset.isRestricted is None
    assert test_dataset.isRawData is True
    assert test_dataset.maintainer == "J.P. Morgan Fusion"
    assert test_dataset.source is None
    assert test_dataset.region is None
    assert test_dataset.publisher == "J.P. Morgan"
    assert test_dataset.product == ["TEST_PRODUCT"]
    assert test_dataset.subCategory is None
    assert test_dataset.tags is None
    assert test_dataset.createdDate is None
    assert test_dataset.modifiedDate is None
    assert test_dataset.deliveryChannel == ["API"]
    assert test_dataset.language == "English"
    assert test_dataset.status == "Available"
    assert test_dataset.type_ == "Source"
    assert test_dataset.containerType == "Snapshot-Full"
    assert test_dataset.snowflake is None
    assert test_dataset.complexity is None
    assert test_dataset.isImmutable is None
    assert test_dataset.isMnpi is None
    assert test_dataset.isPii is None
    assert test_dataset.isPci is None
    assert test_dataset.isClient is None
    assert test_dataset.isPublic is None
    assert test_dataset.isInternal is None
    assert test_dataset.isConfidential is None
    assert test_dataset.isHighlyConfidential is None
    assert test_dataset.isActive is None


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
    assert test_dataset.isInternalOnlyDataset is False
    assert test_dataset.isThirdPartyData is True
    assert test_dataset.isRestricted is None
    assert test_dataset.isRawData is True
    assert test_dataset.maintainer == "J.P. Morgan Fusion"
    assert test_dataset.source is None
    assert test_dataset.region is None
    assert test_dataset.publisher == "J.P. Morgan"
    assert test_dataset.product == ["TEST_PRODUCT"]
    assert test_dataset.subCategory is None
    assert test_dataset.tags is None
    assert test_dataset.createdDate is None
    assert test_dataset.modifiedDate is None
    assert test_dataset.deliveryChannel == ["API"]
    assert test_dataset.language == "English"
    assert test_dataset.status == "Available"
    assert test_dataset.type_ == "Source"
    assert test_dataset.containerType == "Snapshot-Full"
    assert test_dataset.snowflake is None
    assert test_dataset.complexity is None
    assert test_dataset.isImmutable is None
    assert test_dataset.isMnpi is None
    assert test_dataset.isPii is None
    assert test_dataset.isPci is None
    assert test_dataset.isClient is None
    assert test_dataset.isPublic is None
    assert test_dataset.isInternal is None
    assert test_dataset.isConfidential is None
    assert test_dataset.isHighlyConfidential is None
    assert test_dataset.isActive is None


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
    assert test_dataset.isInternalOnlyDataset is False
    assert test_dataset.isThirdPartyData is True
    assert test_dataset.isRestricted is None
    assert test_dataset.isRawData is True
    assert test_dataset.maintainer == "J.P. Morgan Fusion"
    assert test_dataset.source is None
    assert test_dataset.region is None
    assert test_dataset.publisher == "J.P. Morgan"
    assert test_dataset.product == ["TEST_PRODUCT"]
    assert test_dataset.subCategory is None
    assert test_dataset.tags is None
    assert test_dataset.createdDate is None
    assert test_dataset.modifiedDate is None
    assert test_dataset.deliveryChannel == ["API"]
    assert test_dataset.language == "English"
    assert test_dataset.status == "Available"
    assert test_dataset.type_ == "Source"
    assert test_dataset.containerType == "Snapshot-Full"
    assert test_dataset.snowflake is None
    assert test_dataset.complexity is None
    assert test_dataset.isImmutable is None
    assert test_dataset.isMnpi is None
    assert test_dataset.isPii is None
    assert test_dataset.isPci is None
    assert test_dataset.isClient is None
    assert test_dataset.isPublic is None
    assert test_dataset.isInternal is None
    assert test_dataset.isConfidential is None
    assert test_dataset.isHighlyConfidential is None
    assert test_dataset.isActive is None


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
    assert test_dataset.isInternalOnlyDataset is False
    assert test_dataset.isThirdPartyData is True
    assert test_dataset.isRestricted is None
    assert test_dataset.isRawData is True
    assert test_dataset.maintainer == "J.P. Morgan Fusion"
    assert test_dataset.source is None
    assert test_dataset.region is None
    assert test_dataset.publisher == "J.P. Morgan"
    assert test_dataset.product == ["TEST_PRODUCT"]
    assert test_dataset.subCategory is None
    assert test_dataset.tags is None
    assert test_dataset.createdDate is None
    assert test_dataset.modifiedDate is None
    assert test_dataset.deliveryChannel == ["API"]
    assert test_dataset.language == "English"
    assert test_dataset.status == "Available"
    assert test_dataset.type_ == "Source"
    assert test_dataset.containerType == "Snapshot-Full"
    assert test_dataset.snowflake is None
    assert test_dataset.complexity is None
    assert test_dataset.isImmutable is None
    assert test_dataset.isMnpi is None
    assert test_dataset.isPii is None
    assert test_dataset.isPci is None
    assert test_dataset.isClient is None
    assert test_dataset.isPublic is None
    assert test_dataset.isInternal is None
    assert test_dataset.isConfidential is None
    assert test_dataset.isHighlyConfidential is None
    assert test_dataset.isActive is None


def test_dataset_class_from_object_dict() -> None:
    """Test Dataset class."""
    test_dict = {
        "title": "Test Dataset",
        "identifier": "Test Dataset",
        "category": "Test",
        "product": "TEST_PRODUCT",
    }
    test_dataset = Dataset(identifier="test").from_object(test_dict)

    assert str(test_dataset)
    assert repr(test_dataset)
    assert test_dataset.title == "Test Dataset"
    assert test_dataset.identifier == "TEST_DATASET"
    assert test_dataset.category == ["Test"]
    assert test_dataset.description == "Test Dataset"
    assert test_dataset.frequency == "Once"
    assert test_dataset.isInternalOnlyDataset is False
    assert test_dataset.isThirdPartyData is True
    assert test_dataset.isRestricted is None
    assert test_dataset.isRawData is True
    assert test_dataset.maintainer == "J.P. Morgan Fusion"
    assert test_dataset.source is None
    assert test_dataset.region is None
    assert test_dataset.publisher == "J.P. Morgan"
    assert test_dataset.product == ["TEST_PRODUCT"]
    assert test_dataset.subCategory is None
    assert test_dataset.tags is None
    assert test_dataset.createdDate is None
    assert test_dataset.modifiedDate is None
    assert test_dataset.deliveryChannel == ["API"]
    assert test_dataset.language == "English"
    assert test_dataset.status == "Available"
    assert test_dataset.type_ == "Source"
    assert test_dataset.containerType == "Snapshot-Full"
    assert test_dataset.snowflake is None
    assert test_dataset.complexity is None
    assert test_dataset.isImmutable is None
    assert test_dataset.isMnpi is None
    assert test_dataset.isPii is None
    assert test_dataset.isPci is None
    assert test_dataset.isClient is None
    assert test_dataset.isPublic is None
    assert test_dataset.isInternal is None
    assert test_dataset.isConfidential is None
    assert test_dataset.isHighlyConfidential is None
    assert test_dataset.isActive is None


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
    assert test_dataset.isInternalOnlyDataset is False
    assert test_dataset.isThirdPartyData is True
    assert test_dataset.isRestricted is None
    assert test_dataset.isRawData is True
    assert test_dataset.maintainer == "J.P. Morgan Fusion"
    assert test_dataset.source is None
    assert test_dataset.region is None
    assert test_dataset.publisher == "J.P. Morgan"
    assert test_dataset.product == ["TEST_PRODUCT"]
    assert test_dataset.subCategory is None
    assert test_dataset.tags is None
    assert test_dataset.createdDate is None
    assert test_dataset.modifiedDate is None
    assert test_dataset.deliveryChannel == ["API"]
    assert test_dataset.language == "English"
    assert test_dataset.status == "Available"
    assert test_dataset.type_ == "Source"
    assert test_dataset.containerType == "Snapshot-Full"
    assert test_dataset.snowflake is None
    assert test_dataset.complexity is None
    assert test_dataset.isImmutable is None
    assert test_dataset.isMnpi is None
    assert test_dataset.isPii is None
    assert test_dataset.isPci is None
    assert test_dataset.isClient is None
    assert test_dataset.isPublic is None
    assert test_dataset.isInternal is None
    assert test_dataset.isConfidential is None
    assert test_dataset.isHighlyConfidential is None
    assert test_dataset.isActive is None


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
    assert test_dataset.isInternalOnlyDataset is False
    assert test_dataset.isThirdPartyData is True
    assert test_dataset.isRestricted is None
    assert test_dataset.isRawData is True
    assert test_dataset.maintainer == "J.P. Morgan Fusion"
    assert test_dataset.source is None
    assert test_dataset.region is None
    assert test_dataset.publisher == "J.P. Morgan"
    assert test_dataset.product == ["TEST_PRODUCT"]
    assert test_dataset.subCategory is None
    assert test_dataset.tags is None
    assert test_dataset.createdDate is None
    assert test_dataset.modifiedDate is None
    assert test_dataset.deliveryChannel == ["API"]
    assert test_dataset.language == "English"
    assert test_dataset.status == "Available"
    assert test_dataset.type_ == "Source"
    assert test_dataset.containerType == "Snapshot-Full"
    assert test_dataset.snowflake is None
    assert test_dataset.complexity is None
    assert test_dataset.isImmutable is None
    assert test_dataset.isMnpi is None
    assert test_dataset.isPii is None
    assert test_dataset.isPci is None
    assert test_dataset.isClient is None
    assert test_dataset.isPublic is None
    assert test_dataset.isInternal is None
    assert test_dataset.isConfidential is None
    assert test_dataset.isHighlyConfidential is None
    assert test_dataset.isActive is None


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
    assert test_dataset.isInternalOnlyDataset is False
    assert test_dataset.isThirdPartyData is True
    assert test_dataset.isRestricted is None
    assert test_dataset.isRawData is True
    assert test_dataset.maintainer == "J.P. Morgan Fusion"
    assert test_dataset.source is None
    assert test_dataset.region is None
    assert test_dataset.publisher == "J.P. Morgan"
    assert test_dataset.product == ["TEST_PRODUCT"]
    assert test_dataset.subCategory is None
    assert test_dataset.tags is None
    assert test_dataset.createdDate is None
    assert test_dataset.modifiedDate is None
    assert test_dataset.deliveryChannel == ["API"]
    assert test_dataset.language == "English"
    assert test_dataset.status == "Available"
    assert test_dataset.type_ == "Source"
    assert test_dataset.containerType == "Snapshot-Full"
    assert test_dataset.snowflake is None
    assert test_dataset.complexity is None
    assert test_dataset.isImmutable is None
    assert test_dataset.isMnpi is None
    assert test_dataset.isPii is None
    assert test_dataset.isPci is None
    assert test_dataset.isClient is None
    assert test_dataset.isPublic is None
    assert test_dataset.isInternal is None
    assert test_dataset.isConfidential is None
    assert test_dataset.isHighlyConfidential is None
    assert test_dataset.isActive is None


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
    assert my_dataset.isInternalOnlyDataset is False
    assert my_dataset.isThirdPartyData is True
    assert my_dataset.isRestricted is False
    assert my_dataset.isRawData is True
    assert my_dataset.maintainer == "maintainer"
    assert my_dataset.source == ["source"]
    assert my_dataset.region == ["region"]
    assert my_dataset.publisher == "publisher"
    assert my_dataset.product == ["TEST_PRODUCT"]
    assert my_dataset.subCategory == ["subCategory"]
    assert my_dataset.tags == ["tag1", "tag2"]
    assert my_dataset.createdDate == "2020-05-05"
    assert my_dataset.modifiedDate == "2020-05-05"
    assert my_dataset.deliveryChannel == ["API"]
    assert my_dataset.language == "English"
    assert my_dataset.status == "Available"
    assert my_dataset.type_ == "Source"
    assert my_dataset.containerType == "Snapshot-Full"
    assert my_dataset.snowflake == "snowflake"
    assert my_dataset.complexity == "complexity"
    assert my_dataset.isImmutable is False
    assert my_dataset.isMnpi is False
    assert my_dataset.isPii is False
    assert my_dataset.isPci is False
    assert my_dataset.isClient is False
    assert my_dataset.isPublic is False
    assert my_dataset.isInternal is False
    assert my_dataset.isConfidential is False
    assert my_dataset.isHighlyConfidential is False
    assert my_dataset.isActive is False
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
    assert my_dataset.isInternalOnlyDataset is False
    assert my_dataset.isThirdPartyData is True
    assert my_dataset.isRestricted is False
    assert my_dataset.isRawData is True
    assert my_dataset.maintainer == "maintainer"
    assert my_dataset.source == ["source"]
    assert my_dataset.region == ["region"]
    assert my_dataset.publisher == "publisher"
    assert my_dataset.product is None
    assert my_dataset.subCategory == ["subCategory"]
    assert my_dataset.tags == ["tag1", "tag2"]
    assert my_dataset.createdDate == "2020-05-05"
    assert my_dataset.modifiedDate == "2020-05-05"
    assert my_dataset.deliveryChannel == ["API"]
    assert my_dataset.language == "English"
    assert my_dataset.status == "Available"
    assert my_dataset.type_ == "Source"
    assert my_dataset.containerType == "Snapshot-Full"
    assert my_dataset.snowflake == "snowflake"
    assert my_dataset.complexity == "complexity"
    assert my_dataset.isImmutable is False
    assert my_dataset.isMnpi is False
    assert my_dataset.isPii is False
    assert my_dataset.isPci is False
    assert my_dataset.isClient is False
    assert my_dataset.isPublic is False
    assert my_dataset.isInternal is False
    assert my_dataset.isConfidential is False
    assert my_dataset.isHighlyConfidential is False
    assert my_dataset.isActive is False


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
