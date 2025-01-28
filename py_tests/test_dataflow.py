"""Test dataflow.py"""

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
