"""Test file for the updated reports.py"""

import pytest
import requests
import requests_mock

from fusion.fusion import Fusion
from fusion.report import Report


def test_report_object_representation() -> None:
    """Test that Report object is correctly instantiated and represented."""
    report = Report(
        name="MyReport",
        tier_type="Tier1",
        lob="Risk",
        data_node_id={"id": "node123"},
        alternative_id={"alt_id": "A1"},
        title="Test Report"
    )
    assert report.name == "MyReport"
    assert report.title == "Test Report"
    assert isinstance(repr(report), str)

def test_link_attributes_to_terms_success(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    # Setup dummy data using AttributeTermMapping
    report_id = "report-123"
    mappings: list[Report.AttributeTermMapping] = [
        {
            "attribute": {"id": "attr-1"},
            "term": {"id": "term-1"},
            "isKDE": True,
        },
        {
            "attribute": {"id": "attr-2"},
            "term": {"id": "term-2"},
            "isKDE": False,
        },
    ]

    # Mock URL
    base_url = fusion_obj._get_new_root_url()
    url = f"{base_url}/api/corelineage-service/v1/reports/{report_id}/reportElements/businessTerms"
    requests_mock.post(url, status_code=200, json={})

    # Create Report with client
    report = Report(
        name="test-report",
        title="Test Report",
        tier_type="Gold",
        lob="CIB",
        data_node_id={"id": "dn-123"},
        alternative_id={"id": "alt-123"},
    )
    report.client = fusion_obj

    # Call method
    resp = report.link_attributes_to_terms(
        report_id=report_id,
        mappings=mappings,
        return_resp_obj=True,
    )
    http_ok = 200
    assert isinstance(resp, requests.Response)
    assert resp.status_code == http_ok



def test_create_report_success(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test Report.create() with mocked Fusion API."""
    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/reports"
    expected_response = {"status": "ok"}
    http_ok = 200

    requests_mock.post(url, json=expected_response, status_code=http_ok)

    report = Report(
        name="AutoReport",
        tier_type="TierX",
        lob="Ops",
        data_node_id={"id": "node001"},
        alternative_id={"alt": "alt001"},
    )
    report.client = fusion_obj

    resp = report.create(return_resp_obj=True)

    assert resp is not None
    assert resp.status_code == http_ok
    assert resp.json() == expected_response


def test_create_without_client_raises() -> None:
    """Test that Report.create() raises if no Fusion client is set."""
    report = Report(
        name="BrokenReport",
        tier_type="TierX",
        lob="Ops",
        data_node_id={"id": "node001"},
        alternative_id={"alt": "alt001"},
    )

    with pytest.raises(ValueError, match="A Fusion client object is required."):
        report.create()
