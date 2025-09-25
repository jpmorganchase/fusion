"""Test file for updated report.py and reports integration"""

from pathlib import Path

import pytest

from fusion.fusion import Fusion
from fusion.report import Report, Reports, ReportsWrapper


def test_report_basic_fields() -> None:
    report = Report(
        title="Quarterly Report",
        owner_node={"name": "Node1", "type": "User Tool"},
        publisher_node={"name": "Dash1", "type": "Intelligent Solutions"},
        description="Quarterly risk analysis",
        frequency="Quarterly",
        category="Risk",
        sub_category="Ops",
        business_domain="CDO",
        regulatory_related=True,
    )
    assert report.title == "Quarterly Report"
    assert report.owner_node["name"] == "Node1"
    assert report.business_domain == "CDO"


def test_report_to_dict() -> None:
    report = Report(
        title="Sample Report",
        owner_node={"name": "X", "type": "User Tool"},
        publisher_node={
            "name": "Y",
            "type": "Intelligent Solutions",
            "publisher_node_identifier": "seal:app:1",
        },
        description="Some desc",
        frequency="Monthly",
        category="Cat",
        sub_category="Sub",
        business_domain="CDO",
        regulatory_related=False,
    )
    result = report.to_dict()
    assert result["title"] == "Sample Report"
    assert result["businessDomain"] == "CDO"
    assert result["regulatoryRelated"] is False
    assert result["publisherNode"]["publisherNodeIdentifier"] == "seal:app:1"
    assert result["ownerNode"]["name"] == "X"
    assert result["publisherNode"]["name"] == "Y"


def test_report_validation_raises() -> None:
    report = Report(
        title="",
        description="",
        frequency="",
        category="",
        sub_category="",
        business_domain="",
        regulatory_related=True,
        # deliberately omit/leave invalid nodes to trigger validation
        owner_node=None,
        publisher_node=None,
    )
    with pytest.raises(ValueError, match="Missing required fields"):
        report.validate()


def test_report_from_dict() -> None:
    data = {
        "Title": "Dict Report",
        "Description": "Dict desc",
        "Frequency": "Daily",
        "Category": "Cat",
        "SubCategory": "Sub",
        "BusinessDomain": "CDO",
        "RegulatoryRelated": True,
        "OwnerNode": {"name": "OWN", "type": "User Tool"},
        "PublisherNode": {
            "name": "PUB",
            "type": "Intelligent Solutions",
            "publisherNodeIdentifier": "pid-1",
        },
    }
    report = Report.from_dict(data)
    assert isinstance(report, Report)
    assert report.title == "Dict Report"
    assert report.frequency == "Daily"
    assert report.publisher_node["publisher_node_identifier"] == "pid-1"

def test_reports_wrapper_from_csv(tmp_path: Path, fusion_obj: Fusion) -> None:
    csv_data = (
        "Report/Process Name,Report/Process Description,Frequency,Category,"
        "Sub Category,businessDomain,ownerNode_name,ownerNode_type,"
        "publisherNode_name,publisherNode_type,publisherNode_publisherNodeIdentifier,Regulatory Designated\n"
        "TestReport,Test description,Monthly,Risk,Ops,CDO,App1,User Tool,"
        "Dash1,Intelligent Solutions,pub-123,Yes"
    )
    file_path = tmp_path / "test_report.csv"
    file_path.write_text(csv_data)

    wrapper = ReportsWrapper(client=fusion_obj)
    reports = wrapper.from_csv(str(file_path))
    assert isinstance(reports, Reports)
    assert len(reports) == 1
    assert reports[0].title == "TestReport"
    assert reports[0].owner_node["name"] == "App1"
    assert reports[0].publisher_node["name"] == "Dash1"
    assert reports[0].publisher_node["publisher_node_identifier"] == "pub-123"

def test_reports_wrapper_from_object_dicts(fusion_obj: Fusion) -> None:
    source = [
        {
            "Report/Process Name": "ObjReport",
            "Report/Process Description": "Some desc",
            "Frequency": "Monthly",
            "Category": "Finance",
            "Sub Category": "Analysis",
            "businessDomain": "CDO",
            "ownerNode_name": "AppID",
            "ownerNode_type": "User Tool",
            "publisherNode_name": "DashX",
            "publisherNode_type": "Intelligent Solutions",
            "publisherNode_publisherNodeIdentifier": "pid-99",
            "Regulatory Designated": "No",
        }
    ]
    wrapper = ReportsWrapper(client=fusion_obj)
    reports = wrapper.from_object(source)
    assert isinstance(reports, Reports)
    assert reports[0].title == "ObjReport"
    assert reports[0].regulatory_related is False
    assert reports[0].owner_node["name"] == "AppID"
    assert reports[0].publisher_node["publisher_node_identifier"] == "pid-99"

