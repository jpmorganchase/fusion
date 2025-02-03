"""Test case for attributes module."""

from typing import cast

import pandas as pd
import pytest
import requests
import requests_mock

from fusion import Fusion
from fusion.attributes import Attribute, Attributes
from fusion.fusion_types import Types


def test_attribute_class() -> None:
    """Test attribute class."""
    test_attribute = Attribute(
        title="Test Attribute",
        identifier="Test Attribute",
        index=0,
        is_dataset_key=True,
        data_type=cast(Types, "string"),
        available_from="May 5, 2020",
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


def test_attribute_client_value_error() -> None:
    """Test attribute client value error."""
    my_attribute = Attribute(
        title="Test Attribute",
        identifier="Test Attribute",
        index=0,
    )
    with pytest.raises(ValueError, match="A Fusion client object is required.") as error_info:
        my_attribute._use_client(client=None)
    assert str(error_info.value) == "A Fusion client object is required."


def test_attributes_client_value_error() -> None:
    """Test attribute client value error."""
    my_attributes = Attributes()
    with pytest.raises(ValueError, match="A Fusion client object is required.") as error_info:
        my_attributes._use_client(client=None)
    assert str(error_info.value) == "A Fusion client object is required."


def test_attribute_class_from_series() -> None:
    """Test attribute class from series."""
    test_series = pd.Series(
        {
            "title": "Test Attribute",
            "identifier": "Test Attribute",
            "index": 0,
            "isDatasetKey": True,
            "dataType": "string",
            "availableFrom": "May 5, 2020",
        }
    )
    test_attribute = Attribute._from_series(test_series)
    assert str(test_attribute)
    assert repr(test_attribute)
    assert test_attribute.title == "Test Attribute"
    assert test_attribute.identifier == "test_attribute"
    assert test_attribute.index == 0
    assert test_attribute.isDatasetKey is True
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


def test_attribute_class_from_dict() -> None:
    """Test attribute class from dict."""
    test_dict = {
        "title": "Test Attribute",
        "identifier": "Test Attribute",
        "index": 0,
        "isDatasetKey": True,
        "dataType": "string",
        "availableFrom": "May 5, 2020",
    }
    test_attribute = Attribute._from_dict(test_dict)
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


def test_attribute_class_from_object_dict() -> None:
    """Test attribute class from object"""
    test_dict = {
        "title": "Test Attribute",
        "identifier": "Test Attribute",
        "index": 0,
        "isDatasetKey": True,
        "dataType": "string",
        "availableFrom": "May 5, 2020",
    }
    test_attribute = Attribute(identifier="test_attribute", index=0).from_object(test_dict)
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


def test_attribute_class_from_object_series() -> None:
    """Test attribute class from object"""
    test_series = pd.Series(
        {
            "title": "Test Attribute",
            "identifier": "Test Attribute",
            "index": 0,
            "isDatasetKey": True,
            "dataType": "string",
            "availableFrom": "May 5, 2020",
        }
    )
    test_attribute = Attribute(identifier="test_attribute", index=0).from_object(test_series)
    assert str(test_attribute)
    assert repr(test_attribute)
    assert test_attribute.title == "Test Attribute"
    assert test_attribute.identifier == "test_attribute"
    assert test_attribute.index == 0
    assert test_attribute.isDatasetKey is True
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


def test_attribute_class_to_dict() -> None:
    """Test attribute class to dict method"""
    test_attribute = Attribute(
        title="Test Attribute",
        identifier="Test Attribute",
        index=0,
        is_dataset_key=True,
        data_type=cast(Types, "string"),
        available_from="May 5, 2020",
    )
    assert test_attribute.to_dict() == {
        "title": "Test Attribute",
        "identifier": "test_attribute",
        "index": 0,
        "isDatasetKey": True,
        "dataType": "String",
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
        "applicationId": None,
        "publisher": None,
        "isCriticalDataElement": None,
    }


def test_attribute_create(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test creation of individual attribute."""
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    attribute = "test_attribute"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/attributes/{attribute}"

    expected_data = {
        "title": "Test Attribute",
        "identifier": "test_attribute",
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

    requests_mock.put(url, json=expected_data)

    test_attribute = Attribute(
        title="Test Attribute",
        identifier="Test Attribute",
        index=0,
        is_dataset_key=True,
        data_type=cast(Types, "string"),
        available_from="May 5, 2020",
    )
    resp = test_attribute.create(client=fusion_obj, catalog=catalog, dataset=dataset, return_resp_obj=True)
    status_code = 200
    assert isinstance(resp, requests.Response)
    assert resp.status_code == status_code


def test_attribute_delete(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test deletion of individual attribute."""
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    attribute = "test_attribute"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/attributes/{attribute}"

    requests_mock.delete(url, status_code=204)

    test_attribute = Attribute(
        identifier="test_attribute",
        index=0,
    )
    resp = test_attribute.delete(client=fusion_obj, catalog=catalog, dataset=dataset, return_resp_obj=True)
    status_code = 204
    assert isinstance(resp, requests.Response)
    assert resp.status_code == status_code


def test_attribute_class_set_client(fusion_obj: Fusion) -> None:
    """Test attribute class set client method."""
    test_attribute = Attribute(
        title="Test Attribute",
        identifier="Test Attribute",
        index=0,
        is_dataset_key=True,
        data_type=cast(Types, "string"),
        available_from="May 5, 2020",
    )
    test_attribute.client = fusion_obj
    assert test_attribute.client is not None
    assert test_attribute.client == fusion_obj


def test_attribute_class_set_lineage(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test attribute class set lineage"""
    catalog = "my_catalog"

    test_attribute1 = Attribute(identifier="test_attribute1", index=0, application_id="12345")
    test_attribute2 = Attribute(identifier="test_attribute1", index=0, application_id="12345")
    test_attribute3 = Attribute(identifier="test_attribute1", index=0, application_id="12345")
    attributes = [test_attribute2, test_attribute3]

    url = f"{fusion_obj.root_url}catalogs/{catalog}/attributes/lineage"

    exp_data = [
        {
            "source": {
                "catalog": "my_catalog",
                "attribute": "test_attribute1",
                "applicationId": {"id": "12345", "type": "application"},
            },
            "targets": [
                {
                    "catalog": "my_catalog",
                    "attribute": "test_attribute2",
                    "applicationId": {"id": "12345", "type": "application"},
                },
                {
                    "catalog": "my_catalog",
                    "attribute": "test_attribute3",
                    "applicationId": {"id": "12345", "type": "application"},
                },
            ],
        }
    ]

    requests_mock.post(url, json=exp_data)

    resp = test_attribute1.set_lineage(client=fusion_obj, attributes=attributes, catalog=catalog, return_resp_obj=True)
    status_code = 200
    assert isinstance(resp, requests.Response)
    assert resp.status_code == status_code


def test_attribute_class_set_lineage_value_error(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test attribute class set lineage"""
    catalog = "my_catalog"

    test_attribute1 = Attribute(
        identifier="test_attribute1",
        index=0,
    )
    test_attribute2 = Attribute(identifier="test_attribute1", index=0, application_id="12345")
    test_attribute3 = Attribute(identifier="test_attribute1", index=0, application_id="12345")
    attributes = [test_attribute2, test_attribute3]

    url = f"{fusion_obj.root_url}catalogs/{catalog}/attributes/lineage"

    exp_data = [
        {
            "source": {
                "catalog": "my_catalog",
                "attribute": "test_attribute1",
                "applicationId": {"id": "12345", "type": "application"},
            },
            "targets": [
                {
                    "catalog": "my_catalog",
                    "attribute": "test_attribute2",
                    "applicationId": {"id": "12345", "type": "application"},
                },
                {
                    "catalog": "my_catalog",
                    "attribute": "test_attribute3",
                    "applicationId": {"id": "12345", "type": "application"},
                },
            ],
        }
    ]

    requests_mock.post(url, json=exp_data)

    with pytest.raises(ValueError, match="The 'application_id' attribute is required for setting lineage."):
        test_attribute1.set_lineage(client=fusion_obj, attributes=attributes, catalog=catalog, return_resp_obj=True)


def test_attribute_class_set_lineage_value_error2(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test attribute class set lineage"""
    catalog = "my_catalog"

    test_attribute1 = Attribute(identifier="test_attribute1", index=0, application_id="12345")
    test_attribute2 = Attribute(
        identifier="test_attribute1",
        index=0,
    )
    test_attribute3 = Attribute(identifier="test_attribute1", index=0, application_id="12345")
    attributes = [test_attribute2, test_attribute3]

    url = f"{fusion_obj.root_url}catalogs/{catalog}/attributes/lineage"

    exp_data = [
        {
            "source": {
                "catalog": "my_catalog",
                "attribute": "test_attribute1",
                "applicationId": {"id": "12345", "type": "application"},
            },
            "targets": [
                {
                    "catalog": "my_catalog",
                    "attribute": "test_attribute2",
                    "applicationId": {"id": "12345", "type": "application"},
                },
                {
                    "catalog": "my_catalog",
                    "attribute": "test_attribute3",
                    "applicationId": {"id": "12345", "type": "application"},
                },
            ],
        }
    ]

    requests_mock.post(url, json=exp_data)

    with pytest.raises(ValueError, match="The 'application_id' attribute is required for setting lineage."):
        test_attribute1.set_lineage(client=fusion_obj, attributes=attributes, catalog=catalog, return_resp_obj=True)


def test_attributes_class() -> None:
    """Test attributes class."""
    test_attribute = Attribute(
        title="Test Attribute",
        identifier="Test Attribute",
        index=0,
        is_dataset_key=True,
        data_type=cast(Types, "string"),
        available_from="May 5, 2020",
    )
    test_attributes = Attributes([test_attribute])
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


def test_attributes_class_set_client(fusion_obj: Fusion) -> None:
    """Test attribute class set client method."""
    test_attributes = Attributes(
        [
            Attribute(
                title="Test Attribute",
                identifier="Test Attribute",
                index=0,
                is_dataset_key=True,
                data_type=cast(Types, "string"),
                available_from="May 5, 2020",
            )
        ]
    )
    test_attributes.client = fusion_obj
    assert test_attributes.client is not None
    assert test_attributes.client == fusion_obj


def test_attributes_add_attribute() -> None:
    """Test attributes class add attribute method."""
    test_attributes = Attributes([])
    test_attribute = Attribute(
        title="Test Attribute",
        identifier="Test Attribute",
        index=0,
        is_dataset_key=True,
        data_type=cast(Types, "string"),
        available_from="May 5, 2020",
    )
    test_attributes.add_attribute(test_attribute)
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


def test_attributes_remove_attributes() -> None:
    """Test attributes class remove attributes method."""
    test_attributes = Attributes(
        [
            Attribute(
                title="Test Attribute",
                identifier="Test Attribute",
                index=0,
                is_dataset_key=True,
                data_type=cast(Types, "string"),
                available_from="May 5, 2020",
            )
        ]
    )
    test_attributes.remove_attribute("test_attribute")
    assert test_attributes.attributes == []


def test_attributes_remove_attributes_doesnt_exist() -> None:
    """Test attributes class remove attributes method."""
    test_attributes = Attributes(
        [
            Attribute(
                title="Test Attribute",
                identifier="Test Attribute",
                index=0,
                is_dataset_key=True,
                data_type=cast(Types, "string"),
                available_from="May 5, 2020",
            )
        ]
    )
    test_attributes.remove_attribute("test_attribute2")
    assert test_attributes.attributes[0].title == "Test Attribute"


def test_attributes_get_attribute() -> None:
    """Test attributes class get_attribute method."""
    test_attributes = Attributes(
        [
            Attribute(
                title="Test Attribute",
                identifier="Test Attribute",
                index=0,
                is_dataset_key=True,
                data_type=cast(Types, "string"),
                available_from="May 5, 2020",
            )
        ]
    )
    attr = test_attributes.get_attribute("test_attribute")
    assert attr is not None
    assert attr.title == "Test Attribute"


def test_attributes_get_attribute_doesnt_exist() -> None:
    """Test attributes class get_attribute method."""
    test_attributes = Attributes(
        [
            Attribute(
                title="Test Attribute",
                identifier="Test Attribute",
                index=0,
                is_dataset_key=True,
                data_type=cast(Types, "string"),
                available_from="May 5, 2020",
            )
        ]
    )
    attr = test_attributes.get_attribute("test_attribute2")
    assert attr is None


def test_attributes_to_dict() -> None:
    """Test attributes class to_dict method."""
    test_attributes = Attributes(
        [
            Attribute(
                title="Test Attribute",
                identifier="Test Attribute",
                index=0,
                is_dataset_key=True,
                data_type=cast(Types, "string"),
                available_from="May 5, 2020",
            )
        ]
    )
    dict_out = test_attributes.to_dict()

    exp_dict = {
        "attributes": [
            {
                "title": "Test Attribute",
                "identifier": "test_attribute",
                "index": 0,
                "isDatasetKey": True,
                "dataType": "String",
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
                "applicationId": None,
                "publisher": None,
                "isCriticalDataElement": None,
            }
        ]
    }
    assert dict_out == exp_dict


def test_attributes_from_dict_list() -> None:
    """Test attributes class from_dict_list method."""
    test_dict = {
        "attributes": [
            {
                "title": "Test Attribute",
                "identifier": "Test Attribute",
                "index": 0,
                "isDatasetKey": True,
                "dataType": "string",
                "availableFrom": "May 5, 2020",
            }
        ]
    }
    test_attributes = Attributes._from_dict_list(test_dict["attributes"])
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


def test_attributes_from_dataframe() -> None:
    """Test attributes class from_dataframe method."""
    test_df = pd.DataFrame(
        {
            "title": ["Test Attribute"],
            "identifier": ["Test Attribute"],
            "index": [0],
            "isDatasetKey": [True],
            "dataType": ["string"],
            "availableFrom": ["May 5, 2020"],
        }
    )
    test_attributes = Attributes._from_dataframe(test_df)
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


def test_attributes_from_object_list_dict() -> None:
    """Test attributes class from_object"""
    test_dict = {
        "attributes": [
            {
                "title": "Test Attribute",
                "identifier": "Test Attribute",
                "index": 0,
                "isDatasetKey": True,
                "dataType": "string",
                "availableFrom": "May 5, 2020",
            }
        ]
    }
    test_attributes = Attributes().from_object(test_dict["attributes"])
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


def test_attributes_from_object_list_attrs() -> None:
    """Test attributes class from_object"""
    test_attributes_input = [
        Attribute(
            title="Test Attribute",
            identifier="Test Attribute",
            index=0,
            is_dataset_key=True,
            data_type=cast(Types, "string"),
            available_from="May 5, 2020",
        )
    ]
    test_attributes = Attributes().from_object(test_attributes_input)
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


def test_attributes_from_object_value_error() -> None:
    """Test from object method for attributes with ValueError."""
    attr = "attributes"
    with pytest.raises(ValueError, match=f"Could not resolve the object provided: {attr}"):
        Attributes().from_object(attr)  # type: ignore


def test_attributes_from_object_dataframe() -> None:
    """Test attributes class from_dataframe method."""
    test_df = pd.DataFrame(
        {
            "title": ["Test Attribute"],
            "identifier": ["Test Attribute"],
            "index": [0],
            "isDatasetKey": [True],
            "dataType": ["string"],
            "availableFrom": ["May 5, 2020"],
        }
    )
    test_attributes = Attributes().from_object(test_df)
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


def test_from_object_attribute() -> None:
    """Test from object method for attribute."""
    test_attribute = Attribute(
        title="Test Attribute",
        identifier="Test Attribute",
        index=0,
        is_dataset_key=True,
        data_type=cast(Types, "string"),
        available_from="May 5, 2020",
    )

    test_attribute.from_object(test_attribute)
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


def test_from_object_value_error() -> None:
    """Test from object method for attribute with ValueError."""
    attribute_source = "test_attribute"

    with pytest.raises(ValueError, match=f"Could not resolve the object provided: {attribute_source}"):
        Attribute(identifier="test_attribute", index=0).from_object(attribute_source)  # type: ignore


def test_attributes_to_dataframe() -> None:
    """Test attributes class to_dataframe method."""
    test_attributes = Attributes(
        [
            Attribute(
                title="Test Attribute",
                identifier="Test Attribute",
                index=0,
                is_dataset_key=True,
                data_type=cast(Types, "string"),
                available_from="May 5, 2020",
            )
        ]
    )
    test_df = test_attributes.to_dataframe()
    assert test_df.shape == (1, 22)
    assert test_df["title"].iloc[0] == "Test Attribute"
    assert test_df["identifier"].iloc[0] == "test_attribute"
    assert test_df["index"].iloc[0] == 0
    assert test_df["isDatasetKey"].iloc[0]
    assert test_df["dataType"].iloc[0] == "String"
    assert test_df["description"].iloc[0] == "Test Attribute"
    assert test_df["source"].iloc[0] is None
    assert test_df["sourceFieldId"].iloc[0] == "test_attribute"
    assert test_df["isInternalDatasetKey"].iloc[0] is None
    assert test_df["isExternallyVisible"].iloc[0]
    assert test_df["unit"].iloc[0] is None
    assert test_df["multiplier"].iloc[0] == 1.0
    assert test_df["isMetric"].iloc[0] is None
    assert test_df["isPropagationEligible"].iloc[0] is None
    assert test_df["availableFrom"].iloc[0] == "2020-05-05"
    assert test_df["deprecatedFrom"].iloc[0] is None
    assert test_df["term"].iloc[0] == "bizterm1"
    assert test_df["dataset"].iloc[0] is None
    assert test_df["attributeType"].iloc[0] is None


def test_attributes_to_dataframe_empty() -> None:
    """Test attributes class to_dataframe method with empty attributes."""
    test_attributes = Attributes([])
    test_df = test_attributes.to_dataframe()
    assert test_df.shape == (1, 22)
    assert test_df["title"].iloc[0] == "Example Attribute"
    assert test_df["identifier"].iloc[0] == "example_attribute"
    assert test_df["index"].iloc[0] == 0
    assert not test_df["isDatasetKey"].iloc[0]
    assert test_df["dataType"].iloc[0] == "String"
    assert test_df["description"].iloc[0] == "Example Attribute"
    assert test_df["source"].iloc[0] is None
    assert test_df["sourceFieldId"].iloc[0] == "example_attribute"
    assert test_df["isInternalDatasetKey"].iloc[0] is None
    assert test_df["isExternallyVisible"].iloc[0]
    assert test_df["unit"].iloc[0] is None
    assert test_df["multiplier"].iloc[0] == 1.0
    assert test_df["isMetric"].iloc[0] is None
    assert test_df["isPropagationEligible"].iloc[0] is None
    assert test_df["availableFrom"].iloc[0] is None
    assert test_df["deprecatedFrom"].iloc[0] is None
    assert test_df["term"].iloc[0] == "bizterm1"
    assert test_df["dataset"].iloc[0] is None
    assert test_df["attributeType"].iloc[0] is None


def test_attributes_from_catalog(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test attributes class from_catalog method."""
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/attributes"
    expected_data = {
        "resources": [
            {
                "title": "Test Attribute",
                "identifier": "Test Attribute",
                "index": 0,
                "isDatasetKey": True,
                "dataType": "string",
                "availableFrom": "May 5, 2020",
            }
        ]
    }
    requests_mock.get(url, json=expected_data)

    test_attributes = Attributes().from_catalog(client=fusion_obj, catalog=catalog, dataset=dataset)
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


def test_attributes_create(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test creation of multiple attributes."""
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

    test_attributes = Attributes(
        [
            Attribute(
                title="Test Attribute",
                identifier="Test Attribute",
                index=0,
                is_dataset_key=True,
                data_type=cast(Types, "string"),
                available_from="May 5, 2020",
            )
        ]
    )
    resp = test_attributes.create(client=fusion_obj, catalog=catalog, dataset=dataset, return_resp_obj=True)
    status_code = 200
    assert isinstance(resp, requests.Response)
    assert resp.status_code == status_code


def test_catalog_attributes_create(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test creation of multiple attributes."""
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/attributes"

    expected_data = [
        {
            "title": "Test Attribute",
            "identifier": "test_attribute",
            "dataType": "string",
            "description": "Test Attribute",
            "publisher": "J.P. Morgan",
            "applicationId": {"id": "12345", "type": "Application (SEAL)"},
        }
    ]

    requests_mock.post(url, json=expected_data)

    test_attributes = Attributes(
        [
            Attribute(
                title="Test Attribute",
                identifier="test_attribute",
                index=0,
                data_type=cast(Types, "string"),
                publisher="J.P. Morgan",
                application_id="12345",
            )
        ]
    )
    resp = test_attributes.create(client=fusion_obj, catalog=catalog, return_resp_obj=True)
    status_code = 200
    assert isinstance(resp, requests.Response)
    assert resp.status_code == status_code


def test_catalog_attributes_create_no_publisher(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test creation of multiple attributes."""
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/attributes"

    expected_data = [
        {
            "title": "Test Attribute",
            "identifier": "test_attribute",
            "dataType": "string",
            "description": "Test Attribute",
            "applicationId": {"id": "12345", "type": "Application (SEAL)"},
        }
    ]

    requests_mock.post(url, json=expected_data)

    test_attributes = Attributes(
        [
            Attribute(
                title="Test Attribute",
                identifier="test_attribute",
                index=0,
                data_type=cast(Types, "string"),
                application_id="12345",
            )
        ]
    )
    with pytest.raises(ValueError, match="The 'publisher' attribute is required for catalog attributes."):
        test_attributes.create(client=fusion_obj, catalog=catalog, return_resp_obj=True)


def test_catalog_attributes_create_no_app_id(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test creation of multiple attributes."""
    catalog = "my_catalog"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/attributes"

    expected_data = [
        {
            "title": "Test Attribute",
            "identifier": "test_attribute",
            "dataType": "string",
            "description": "Test Attribute",
            "publisher": "J.P. Morgan",
        }
    ]

    requests_mock.post(url, json=expected_data)

    test_attributes = Attributes(
        [
            Attribute(
                title="Test Attribute",
                identifier="test_attribute",
                index=0,
                data_type=cast(Types, "string"),
                publisher="J.P. Morgan",
            )
        ]
    )
    with pytest.raises(ValueError, match="The 'application_id' attribute is required for catalog attributes."):
        test_attributes.create(client=fusion_obj, catalog=catalog, return_resp_obj=True)


def test_attributes_create_no_client() -> None:
    """Test create attribute without client."""
    test_attributes = Attributes(
        [
            Attribute(
                title="Test Attribute",
                identifier="Test Attribute",
                index=0,
                is_dataset_key=True,
                data_type=cast(Types, "string"),
                available_from="May 5, 2020",
            )
        ]
    )
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    with pytest.raises(ValueError, match="A Fusion client object is required."):
        test_attributes.create(catalog=catalog, dataset=dataset, return_resp_obj=True)


def test_attributes_from_catalog_no_client() -> None:
    """Test attributes class from_catalog method."""
    catalog = "my_catalog"
    dataset = "TEST_DATASET"

    with pytest.raises(ValueError, match="A Fusion client object is required."):
        Attributes().from_catalog(catalog=catalog, dataset=dataset)


def test_attribute_create_no_client() -> None:
    """Test create attribute without client."""
    test_attribute = Attribute(
        title="Test Attribute",
        identifier="Test Attribute",
        index=0,
        is_dataset_key=True,
        data_type=cast(Types, "string"),
        available_from="May 5, 2020",
    )

    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    with pytest.raises(ValueError, match="A Fusion client object is required."):
        test_attribute.create(catalog=catalog, dataset=dataset, return_resp_obj=True)


def test_attributes_delete(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test deletion of multiple attributes."""
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataset}/attributes/test_attribute"

    requests_mock.delete(url, status_code=204)

    test_attributes = Attributes(
        [
            Attribute(
                identifier="test_attribute",
                index=0,
            )
        ]
    )
    resp = test_attributes.delete(client=fusion_obj, catalog=catalog, dataset=dataset, return_resp_obj=True)
    status_code = 204
    assert resp is not None
    assert isinstance(resp[0], requests.Response)
    assert resp[0].status_code == status_code


def test_attributes_delete_no_client() -> None:
    """Test create attribute without client."""
    test_attributes = Attributes(
        [
            Attribute(
                title="Test Attribute",
                identifier="Test Attribute",
                index=0,
                is_dataset_key=True,
                data_type=cast(Types, "string"),
                available_from="May 5, 2020",
            )
        ]
    )
    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    with pytest.raises(ValueError, match="A Fusion client object is required."):
        test_attributes.delete(catalog=catalog, dataset=dataset, return_resp_obj=True)


def test_attribute_delete_no_client() -> None:
    """Test create attribute without client."""
    test_attribute = Attribute(
        title="Test Attribute",
        identifier="Test Attribute",
        index=0,
        is_dataset_key=True,
        data_type=cast(Types, "string"),
        available_from="May 5, 2020",
    )

    catalog = "my_catalog"
    dataset = "TEST_DATASET"
    with pytest.raises(ValueError, match="A Fusion client object is required."):
        test_attribute.delete(catalog=catalog, dataset=dataset, return_resp_obj=True)


def test_attribute_case_switching() -> None:
    """Test attribute class case switching."""
    my_attribute = Attribute(
        title="Test Attribute",
        identifier="Test Attribute",
        index=0,
        is_dataset_key=True,
        data_type="string",  # type: ignore
        available_from="May 5, 2020",
        is_internal_dataset_key=True,
        is_externally_visible=False,
        is_metric=True,
        is_propagation_eligible=True,
        deprecated_from="May 5, 2021",
    )

    my_attribute_dict = my_attribute.to_dict()

    assert my_attribute_dict == {
        "identifier": "test_attribute",
        "index": 0,
        "dataType": "String",
        "title": "Test Attribute",
        "description": "Test Attribute",
        "isDatasetKey": True,
        "source": None,
        "sourceFieldId": "test_attribute",
        "isInternalDatasetKey": True,
        "isExternallyVisible": False,
        "unit": None,
        "multiplier": 1.0,
        "isPropagationEligible": True,
        "isMetric": True,
        "availableFrom": "2020-05-05",
        "deprecatedFrom": "2021-05-05",
        "term": "bizterm1",
        "dataset": None,
        "attributeType": None,
        "applicationId": None,
        "publisher": None,
        "isCriticalDataElement": None,
    }

    attribute_from_camel_dict = Attribute("test_attribute", 0).from_object(my_attribute_dict)

    assert attribute_from_camel_dict == my_attribute

    assert attribute_from_camel_dict.dataType == attribute_from_camel_dict.data_type
    assert attribute_from_camel_dict.isDatasetKey == attribute_from_camel_dict.is_dataset_key
    assert attribute_from_camel_dict.isInternalDatasetKey == attribute_from_camel_dict.is_internal_dataset_key
    assert attribute_from_camel_dict.isExternallyVisible == attribute_from_camel_dict.is_externally_visible
    assert attribute_from_camel_dict.isMetric == attribute_from_camel_dict.is_metric
    assert attribute_from_camel_dict.isPropagationEligible == attribute_from_camel_dict.is_propagation_eligible
    assert attribute_from_camel_dict.deprecatedFrom == attribute_from_camel_dict.deprecated_from
    assert attribute_from_camel_dict.availableFrom == attribute_from_camel_dict.available_from
    assert attribute_from_camel_dict.sourceFieldId == attribute_from_camel_dict.source_field_id
    assert attribute_from_camel_dict.attributeType == attribute_from_camel_dict.attribute_type


def test_attribute_getattr() -> None:
    """Test __getattr__ method for Attribute class."""
    test_attribute = Attribute(
        title="Test Attribute",
        identifier="Test Attribute",
        index=0,
        is_dataset_key=True,
        data_type=cast(Types, "string"),
        available_from="May 5, 2020",
    )
    assert test_attribute.title == "Test Attribute"
    assert test_attribute.identifier == "test_attribute"
    assert test_attribute.index == 0
    assert test_attribute.isDatasetKey is True
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

    # Test accessing attributes using camelCase
    assert test_attribute.isDatasetKey == test_attribute.is_dataset_key
    assert test_attribute.dataType == test_attribute.data_type
    assert test_attribute.sourceFieldId == test_attribute.source_field_id
    assert test_attribute.isInternalDatasetKey == test_attribute.is_internal_dataset_key
    assert test_attribute.isExternallyVisible == test_attribute.is_externally_visible
    assert test_attribute.isPropagationEligible == test_attribute.is_propagation_eligible
    assert test_attribute.isMetric == test_attribute.is_metric
    assert test_attribute.availableFrom == test_attribute.available_from
    assert test_attribute.deprecatedFrom == test_attribute.deprecated_from

    # Test accessing non-existent attribute
    with pytest.raises(AttributeError) as error_info:
        _ = test_attribute.nonExistentAttribute
    assert str(error_info.value) == "'Attribute' object has no attribute 'nonExistentAttribute'"
