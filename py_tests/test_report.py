"""Test file for report.py and reports integration"""

from pathlib import Path
from typing import Any, Optional, cast

import pytest

from fusion.fusion import Fusion
from fusion.report import Report, Reports


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
    assert report.owner_node is not None
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
        owner_node=None,
        publisher_node=None,
    )
    with pytest.raises(ValueError, match="Missing required fields"):
        report.validate()


def test_report_from_dict() -> None:
    data = {
        "title": "Dict Report",
        "description": "Dict desc",
        "frequency": "Daily",
        "category": "Cat",
        "subCategory": "Sub",
        "businessDomain": "CDO",
        "regulatoryRelated": True,
        "ownerNode": {"name": "OWN", "type": "User Tool"},
        "publisherNode": {
            "name": "PUB",
            "type": "Intelligent Solutions",
            "publisherNodeIdentifier": "pid-1",
        },
    }
    report = Report.from_dict(data)
    assert isinstance(report, Report)
    assert report.title == "Dict Report"
    assert report.frequency == "Daily"
    assert report.publisher_node is not None
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

    reports = Reports.from_csv(str(file_path), client=fusion_obj)
    assert isinstance(reports, Reports)
    assert len(reports) == 1
    assert reports[0].title == "TestReport"
    assert reports[0].owner_node is not None
    assert reports[0].owner_node["name"] == "App1"
    assert reports[0].publisher_node is not None
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
    reports = Reports.from_object(source, client=fusion_obj)
    assert isinstance(reports, Reports)
    assert reports[0].title == "ObjReport"
    assert reports[0].regulatory_related is False
    assert reports[0].owner_node is not None
    assert reports[0].owner_node["name"] == "AppID"
    assert reports[0].publisher_node is not None
    assert reports[0].publisher_node["publisher_node_identifier"] == "pid-99"


def test_report_update_fields_excludes_id_and_uses_path() -> None:
    """PATCH should not send 'id' in body and must use /reports/{id} path."""
    report = Report(
        id="r-1",
        title="t",
        description="d",
        frequency="f",
        category="c",
        sub_category="s",
        business_domain="bd",
        regulatory_related=True,
        owner_node={"name": "own", "type": "User Tool"},
        publisher_node={"name": "pub", "type": "Intelligent Solutions"},
    )

    class _Resp:
        status_code = 200
        ok = True
        text = ""
        content = b""

        def raise_for_status(self) -> None:
            return None

    class _Sess:
        def __init__(self) -> None:
            self.last_url: Optional[str] = None
            self.last: Optional[dict[str, Any]] = None

        def patch(self, url: str, json: dict[str, Any]) -> _Resp:
            self.last_url = url
            self.last = json
            return _Resp()

    class _Fusion:
        def __init__(self) -> None:
            self.session = _Sess()

        def _get_new_root_url(self) -> str:
            return "http://unit.test"

    client = _Fusion()
    report.client = cast(Fusion, client)

    report.update_fields()

    assert client.session.last is not None
    assert "id" not in client.session.last  
    assert client.session.last_url is not None
    assert client.session.last_url.endswith("/api/corelineage-service/v1/reports/r-1")


def test_report_create_excludes_id_and_sets_id() -> None:
    """CREATE should not send 'id' in the POST body."""
    report = Report(
        id="pre-set",
        title="t",
        description="d",
        frequency="f",
        category="c",
        sub_category="s",
        business_domain="bd",
        regulatory_related=True,
        owner_node={"name": "own", "type": "User Tool"},
        publisher_node={"name": "pub", "type": "Intelligent Solutions"},
    )

    class _Resp:
        status_code = 200
        ok = True
        text = ""
        content = b""

        def raise_for_status(self) -> None:
            return None

       
        def json(self) -> dict[str, Any]:
            return {"id": "new-123"}

    class _Sess:
        def __init__(self) -> None:
            self.last_url: Optional[str] = None
            self.last: Optional[dict[str, Any]] = None

        def post(self, url: str, json: dict[str, Any]) -> _Resp:
            self.last_url = url
            self.last = json
            return _Resp()

    class _Fusion:
        def __init__(self) -> None:
            self.session = _Sess()

        def _get_new_root_url(self) -> str:
            return "http://unit.test"

    client = _Fusion()
    report.client = cast(Fusion, client)

    report.create()

    assert client.session.last is not None
    assert "id" not in client.session.last  
    assert client.session.last_url is not None
    assert client.session.last_url.endswith("/api/corelineage-service/v1/reports")



def test_report_update_excludes_id_in_body_and_uses_path() -> None:
    """UPDATE should not send 'id' in body and must use /reports/{id} path."""
    report = Report(
        id="abc-999",
        title="t",
        description="d",
        frequency="f",
        category="c",
        sub_category="s",
        business_domain="bd",
        regulatory_related=False,
        owner_node={"name": "own", "type": "User Tool"},
        publisher_node={"name": "pub", "type": "Intelligent Solutions"},
    )

    class _Resp:
        status_code = 200
        ok = True
        text = ""
        content = b""

        def raise_for_status(self) -> None:
            return None

    class _Sess:
        def __init__(self) -> None:
            self.last_url: Optional[str] = None
            self.last: Optional[dict[str, Any]] = None

        def put(self, url: str, json: dict[str, Any]) -> _Resp:
            self.last_url = url
            self.last = json
            return _Resp()

    class _Fusion:
        def __init__(self) -> None:
            self.session = _Sess()

        def _get_new_root_url(self) -> str:
            return "http://unit.test"

    client = _Fusion()
    report.client = cast(Fusion, client)

    report.update()

    assert client.session.last is not None
    assert "id" not in client.session.last 
    assert client.session.last_url is not None
    assert client.session.last_url.endswith("/api/corelineage-service/v1/reports/abc-999")