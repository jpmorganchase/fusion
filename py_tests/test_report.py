"""Test file for report.py"""

import pytest
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
        attribute_identifier="my_attribute", is_key_data_element=True, catalog=catalog, return_resp_obj=True
    )
    assert isinstance(resp, requests.Response)
    status_code = 200
    assert resp.status_code == status_code


def test_create_report_no_tier(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test creating a Report object without a tier."""
    catalog = "my_catalog"
    report = "REPORT"
    url = f"{fusion_obj.root_url}catalogs/{catalog}/datasets/{report}"

    exp_data = {
        "identifier": "REPORT",
        "title": "Report",
        "category": None,
        "description": "Report",
        "frequency": "Once",
        "isInternalOnlyDataset": False,
        "isThirdPartyData": True,
        "isRestricted": None,
        "isRawData": True,
        "maintainer": "J.P. Morgan Fusion",
        "source": None,
        "region": None,
        "publisher": "J.P. Morgan",
        "product": None,
        "subCategory": None,
        "tags": None,
        "createdDate": None,
        "modifiedDate": None,
        "deliveryChannel": ["API"],
        "language": "English",
        "status": "Available",
        "type": "Report",
        "containerType": "Snapshot-Full",
        "snowflake": None,
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
        "report": {"tier": ""},
    }

    requests_mock.post(url, json=exp_data)

    report_obj = Report(identifier="REPORT")
    report_obj.client = fusion_obj
    with pytest.raises(ValueError, match="Tier cannot be blank for reports."):
        report_obj.create(catalog=catalog)
