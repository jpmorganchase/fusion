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
