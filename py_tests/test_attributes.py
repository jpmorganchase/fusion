"""Test case for attributes module."""

import json
from collections.abc import Generator
from typing import Any, cast

from fusion.fusion_types import Types
import pandas as pd
import pytest
import requests
import requests_mock

from fusion import Fusion
from fusion.attributes import Attributes, Attribute

def test_attribute_class() -> None:
    """Test attribute class."""
    test_attribute = Attribute(
        title= "Test Attribute",
        identifier= "Test Attribute",
        index= 0,
        isDatatsetKey=True,
        dataType=cast(Types, "string"),
        availableFrom="May 5, 2020",
    )
    assert str(test_attribute)
    assert repr(test_attribute)
    assert test_attribute.title == "Test Attribute"
    assert test_attribute.identifier == "test_attribute"
    assert test_attribute.index == 0
    assert test_attribute.isDatatsetKey
    assert test_attribute.dataType == Types.String
    assert test_attribute.description == "Test Attribute"
    assert test_attribute.source is None
    assert test_attribute.sourceFieldId == "test_attribute"
    assert test_attribute.isInternalDatasetKey is None
    assert test_attribute.isExternallyVisible is True
    assert test_attribute.unit is None
    assert test_attribute.multiplier == 1.0
    assert test_attribute.isMetric is None
    assert test_attribute.isPropogationEligible is None
    assert test_attribute.availableFrom == "2020-05-05"
    assert test_attribute.deprecatedFrom is None
    assert test_attribute.term == "bizterm1"
    assert test_attribute.dataset is None
    assert test_attribute.attributeType is None


def test_attribute_class_from_series() -> None:
    """Test attribute class from series."""
    test_series = pd.Series(
        {
            "title": "Test Attribute",
            "identifier": "Test Attribute",
            "index": 0,
            "isDatatsetKey": True,
            "dataType": "string",
            "availableFrom": "May 5, 2020",
        }
    )
    test_attribute = Attribute.from_series(test_series)
    assert str(test_attribute)
    assert repr(test_attribute)
    assert test_attribute.title == "Test Attribute"
    assert test_attribute.identifier == "test_attribute"
    assert test_attribute.index == 0
    assert test_attribute.isDatatsetKey
    assert test_attribute.dataType == Types.String
    assert test_attribute.description == "Test Attribute"
    assert test_attribute.source is None
    assert test_attribute.sourceFieldId == "test_attribute"
    assert test_attribute.isInternalDatasetKey is None
    assert test_attribute.isExternallyVisible is True
    assert test_attribute.unit is None
    assert test_attribute.multiplier == 1.0
    assert test_attribute.isMetric is None
    assert test_attribute.isPropogationEligible is None
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
        "isDatatsetKey": True,
        "dataType": "string",
        "availableFrom": "May 5, 2020",
    }
    test_attribute = Attribute.from_dict(test_dict)
    assert str(test_attribute)
    assert repr(test_attribute)
    assert test_attribute.title == "Test Attribute"
    assert test_attribute.identifier == "test_attribute"
    assert test_attribute.index == 0
    assert test_attribute.isDatatsetKey
    assert test_attribute.dataType == Types.String
    assert test_attribute.description == "Test Attribute"
    assert test_attribute.source is None
    assert test_attribute.sourceFieldId == "test_attribute"
    assert test_attribute.isInternalDatasetKey is None
    assert test_attribute.isExternallyVisible is True
    assert test_attribute.unit is None
    assert test_attribute.multiplier == 1.0
    assert test_attribute.isMetric is None
    assert test_attribute.isPropogationEligible is None
    assert test_attribute.availableFrom == "2020-05-05"
    assert test_attribute.deprecatedFrom is None
    assert test_attribute.term == "bizterm1"
    assert test_attribute.dataset is None
    assert test_attribute.attributeType is None


def test_attribute_class_to_dict() -> None:
    """Test attribute class to dict method"""
    test_attribute = Attribute(
        title= "Test Attribute",
        identifier= "Test Attribute",
        index= 0,
        isDatatsetKey=True,
        dataType=cast(Types, "string"),
        availableFrom="May 5, 2020",
    )
    assert test_attribute.to_dict() == {
        "title": "Test Attribute",
        "identifier": "test_attribute",
        "index": 0,
        "isDatatsetKey": True,
        "dataType": "string",
        "description": "Test Attribute",
        "source": None,
        "sourceFieldId": "test_attribute",
        "isInternalDatasetKey": None,
        "isExternallyVisible": True,
        "unit": None,
        "multiplier": 1.0,
        "isMetric": None,
        "isPropogationEligible": None,
        "availableFrom": "2020-05-05",
        "deprecatedFrom": None,
        "term": "bizterm1",
        "dataset": None,
        "attributeType": None,
    }


def test_attribute_create() -> None:
    """Test creation of individual attribute."""
    


def test_attributes_class() -> None:
    """Test attributes class."""
    test_attribute = Attribute(
        title= "Test Attribute",
        identifier= "Test Attribute",
        index= 0,
        isDatatsetKey=True,
        dataType=cast(Types, "string"),
        availableFrom="May 5, 2020",
    )
    test_attributes = Attributes([test_attribute])
    assert str(test_attributes)
    assert repr(test_attributes)
    assert test_attributes[0].title == "Test Attribute"
    assert test_attributes[0].identifier == "test_attribute"
    assert test_attributes[0].index == 0
    assert test_attributes[0].isDatatsetKey
    assert test_attributes[0].dataType == Types.String
    assert test_attributes[0].description == "Test Attribute"
    assert test_attributes[0].source is None
    assert test_attributes[0].sourceFieldId == "test_attribute"
    assert test_attributes[0].isInternalDatasetKey is None
    assert test_attributes[0].isExternallyVisible is True
    assert test_attributes[0].unit is None
    assert test_attributes[0].multiplier == 1.0
    assert test_attributes[0].isMetric is None
    assert test_attributes[0].isPropogationEligible is None
    assert test_attributes[0].availableFrom == "2020-05-05"
    assert test_attributes[0].deprecatedFrom is None
    assert test_attributes[0].term == "bizterm1"
    assert test_attributes[0].dataset is None
    assert test_attributes[0].attributeType is None



