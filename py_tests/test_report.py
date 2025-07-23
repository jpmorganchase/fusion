"""Test file for updated report.py and reports integration"""

from pathlib import Path

import pytest

from fusion.fusion import Fusion
from fusion.report import Report, Reports, ReportsWrapper


def test_report_basic_fields() -> None:
    report = Report(
        title="Quarterly Report",
        data_node_id={"name": "Node1", "dataNodeType": "User Tool"},
        description="Quarterly risk analysis",
        frequency="Quarterly",
        category="Risk",
        sub_category="Ops",
        domain={"name": "CDO"},
        regulatory_related=True,
    )
    assert report.title == "Quarterly Report"
    assert report.data_node_id["name"] == "Node1"
    assert report.domain["name"] == "CDO"


def test_report_to_dict() -> None:
    report = Report(
        title="Sample Report",
        data_node_id={"name": "X", "dataNodeType": "Y"},
        description="Some desc",
        frequency="Monthly",
        category="Cat",
        sub_category="Sub",
        domain={"name": "CDO"},
        regulatory_related=False,
    )
    result = report.to_dict()
    assert result["title"] == "Sample Report"
    assert result["domain"]["name"] == "CDO"
    assert result["regulatoryRelated"] is False


def test_report_validation_raises() -> None:
    report = Report(
        title="",
        data_node_id={"name": "X", "dataNodeType": "Y"},
        description="",
        frequency="",
        category="",
        sub_category="",
        domain={"name": ""},
        regulatory_related=True,
    )
    with pytest.raises(ValueError, match="Missing required fields"):
        report.validate()


def test_report_from_dict() -> None:
    data = {
        "Title": "Dict Report",
        "DataNodeId": {"name": "X", "dataNodeType": "Y"},
        "Description": "Dict desc",
        "Frequency": "Daily",
        "Category": "Cat",
        "SubCategory": "Sub",
        "Domain": {"name": "CDO"},
        "RegulatoryRelated": True,
    }
    report = Report.from_dict(data)
    assert isinstance(report, Report)
    assert report.title == "Dict Report"
    assert report.frequency == "Daily"


def test_reports_wrapper_from_csv(tmp_path: Path, fusion_obj: Fusion) -> None:
    csv_data = (
        "Report/Process Name,Report/Process Description,Frequency,Category,"
        "Sub Category,CDO Office,Application ID,Application Type,Regulatory Designated\n"
        "TestReport,Test description,Monthly,Risk,Ops,CDO,App1,User Tool,Yes"
    )
    file_path = tmp_path / "test_report.csv"
    file_path.write_text(csv_data)

    wrapper = ReportsWrapper(client=fusion_obj)
    reports = wrapper.from_csv(str(file_path))
    assert isinstance(reports, Reports)
    assert len(reports) == 1
    assert reports[0].title == "TestReport"


def test_reports_wrapper_from_object_dicts(fusion_obj: Fusion) -> None:
    source = [
        {
            "Report/Process Name": "ObjReport",
            "Report/Process Description": "Some desc",
            "Frequency": "Monthly",
            "Category": "Finance",
            "Sub Category": "Analysis",
            "CDO Office": "CDO",
            "Application ID": "AppID",
            "Application Type": "User Tool",
            "Regulatory Designated": "No",
        }
    ]
    wrapper = ReportsWrapper(client=fusion_obj)
    reports = wrapper.from_object(source)
    assert isinstance(reports, Reports)
    assert reports[0].title == "ObjReport"
    assert reports[0].regulatory_related is False
