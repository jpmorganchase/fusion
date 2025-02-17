"""Test dataflow.py"""

import re

import pytest
import requests
import requests_mock

from fusion.dataflow import InputDataFlow, OutputDataFlow
from fusion.fusion import Fusion


def test_inputdataflow_class_object_representation() -> None:
    """Test the object representation of the Dataflow class."""
    dataflow = InputDataFlow(identifier="my_dataflow", flow_details={"key": "value"})
    assert repr(dataflow)


def test_outputdataflow_class_object_representation() -> None:
    """Test the object representation of the Dataflow class."""
    dataflow = OutputDataFlow(identifier="my_dataflow", flow_details={"key": "value"})
    assert repr(dataflow)


def test_add_registered_attribute(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test the add_registered_attribute method."""
    catalog = "my_catalog"
    dataflow = "TEST_DATAFLOW"
    attribute_identifier = "my_attribute"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{dataflow}/attributes/{attribute_identifier}/registration"

    requests_mock.post(url, json={"isCriticalDataElement": False})

    dataflow_obj = InputDataFlow(identifier="TEST_DATAFLOW")
    dataflow_obj.client = fusion_obj
    resp = dataflow_obj.add_registered_attribute(
        attribute_identifier="my_attribute", catalog=catalog, return_resp_obj=True
    )
    assert isinstance(resp, requests.Response)
    status_code = 200
    assert resp.status_code == status_code


def test_inputdataflow_post_init_with_application_id() -> None:
    """Test __post_init__ method of InputDataFlow with application_id."""
    dataflow = InputDataFlow(identifier="my_dataflow", application_id="app_id")
    assert dataflow.consumer_application_id == [{"id": "app_id", "type": "Application (SEAL)"}]


def test_inputdataflow_post_init_without_application_id() -> None:
    """Test __post_init__ method of InputDataFlow without application_id."""
    dataflow = InputDataFlow(identifier="my_dataflow")
    assert dataflow.consumer_application_id is None


def test_outputdataflow_producer_application_id_invalid_dict() -> None:
    """Test DataFlow with producer_application_id missing required keys."""
    with pytest.raises(
        ValueError,
        match=re.escape("producer_application_id must contain keys: {'id', 'type'}")
        + "|"
        + re.escape("producer_application_id must contain keys: {'type', 'id'}"),
    ):
        OutputDataFlow(identifier="Test DataFlow", producer_application_id={"id": "12345"})


def test_outputdataflow_producer_application_id_invalid_type() -> None:
    """Test DataFlow with producer_application_id having an invalid type."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid producer_application_id type: Invalid. Must be one of "
            "Application (SEAL), Intelligent Solution, User Tool"
        ),
    ):
        OutputDataFlow(
            identifier="Test DataFlow",
            producer_application_id={"id": "12345", "type": "Invalid"},
        )


def test_inputdataflow_class_consumer_application_id_invalid_dict() -> None:
    """Test DataFlow with consumer_application_id missing required keys."""
    with pytest.raises(
        ValueError,
        match=re.escape("application_id must contain keys: {'id', 'type'}")
        + "|"
        + re.escape("application_id must contain keys: {'type', 'id'}"),
    ):
        InputDataFlow(identifier="Test DataFlow", application_id={"id": "12345"})


def test_inputdataflow_consumer_application_id_invalid_type() -> None:
    """Test DataFlow with consumer_application_id having an invalid type."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid application_id type: Invalid. Must be one of Application (SEAL), Intelligent Solution, User Tool"
        ),
    ):
        InputDataFlow(
            identifier="Test DataFlow",
            application_id={"id": "12345", "type": "Invalid"},
        )


def test_dataflow_consumer_application_id_non_string_id() -> None:
    """Test DataFlow with consumer_application_id where 'id' is not a string."""
    dataflow = OutputDataFlow(
        identifier="Test DataFlow",
        consumer_application_id={"id": "98765", "type": "User Tool"},
    )
    if isinstance(dataflow.consumer_application_id, list):
        assert dataflow.consumer_application_id[0]["id"] == "98765"
        assert dataflow.consumer_application_id[0]["type"] == "User Tool"
    elif isinstance(dataflow.consumer_application_id, dict):
        assert dataflow.consumer_application_id["id"] == "98765"
        assert dataflow.consumer_application_id["type"] == "User Tool"
