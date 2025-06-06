"""Test file for the updated reports.py"""

import pytest
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


def test_report_to_dict() -> None:
    """Test the to_dict method to ensure camelCase conversion."""
    report = Report(
        name="TestReport",
        tier_type="Tier1",
        lob="Finance",
        data_node_id={"id": "node123"},
        alternative_id={"alt_id": "A1"},
        is_bcbs239_program=True,
    )
    report_dict = report.to_dict()
    assert "isBCBS239Program" in report_dict
    assert report_dict["isBCBS239Program"] is True
    assert "name" not in report_dict  # should be camelCased
    assert "name" in report.__dict__  # but exists internally


def test_from_dict_partial() -> None:
    """Test creation of Report from partial dictionary."""
    raw_data = {
        "name": "PartialReport",
        "tierType": "Tier2",
        "lob": "Compliance",
        "dataNodeId": {"id": "node321"},
        "alternativeId": {"id": "alt999"},
        "isBCBS239Program": "true"
    }

    report = Report.from_dict(raw_data)
    assert isinstance(report, Report)
    assert report.name == "PartialReport"
    assert report.is_bcbs239_program is True
    assert report.lob == "Compliance"

def test_link_attributes_to_terms(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    report_id = "report_abc123"
    base_url = fusion_obj._get_new_root_url()
    path = f"/api/corelineage-service/v1/reports/{report_id}/reportElements/businessTerms"
    url = f"{base_url}{path}"


    mock_response = {"status": "success"}
    requests_mock.post(url, json=mock_response, status_code=200)


    mappings: list[Fusion.AttributeTermMapping] = [
        {
            "attribute": {"id": "attr1"},
            "term": {"id": "term1"},
            "isKDE": True,
        },
        {
            "attribute": {"id": "attr2"},
            "term": {"id": "term2"},
            "isKDE": False,
        },
    ]

    response = fusion_obj.link_attributes_to_terms(
        report_id=report_id,
        mappings=mappings,
        return_resp_obj=True,
    )
    http_ok = 200

    assert response is not None


    assert response.status_code == http_ok
    assert response.json() == mock_response


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
