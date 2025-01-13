"""Test file for report.py"""

import requests
import requests_mock

from fusion.fusion import Fusion
from fusion.report import Report


def test_report_class_object_representation() -> None:
    """Test the object representation of the Report class."""
    report = Report(identifier="my_report", report={"key": "value"})
    assert repr(report)

def test_add_registered_attribute(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test the add_registered_attribute method."""
    catalog = "my_catalog"
    report = "TEST_REPORT"
    attribute_identifier = "my_attribute"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{report}/attributes/{attribute_identifier}/registration"

    requests_mock.post(url, json={"isCriticalDataElement": True})

    report_obj = Report(identifier="TEST_REPORT")
    report_obj.client = fusion_obj
    resp = report_obj.add_registered_attribute(
        attribute_identifier="my_attribute",
        is_key_data_element=True,
        catalog=catalog,
        return_resp_obj=True
    )
    assert isinstance(resp, requests.Response)
    status_code = 200
    assert resp.status_code == status_code
