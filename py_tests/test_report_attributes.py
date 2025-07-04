from pathlib import Path

import pandas as pd
import pytest
import requests
import requests_mock

from fusion import Fusion
from fusion.report_attributes import ReportAttribute, ReportAttributes


def test_report_attribute_str_repr() -> None:
    attr = ReportAttribute(name="revenue", title="Revenue")
    assert isinstance(str(attr), str)
    assert isinstance(repr(attr), str)


def test_report_attribute_to_dict() -> None:
    attr = ReportAttribute(
        name="revenue",
        title="Revenue",
        description="Total revenue",
        technicalDataType="decimal",
        path="finance/metrics",
        dataPublisher="JPM",
    )
    expected = {
        "name": "revenue",
        "title": "Revenue",
        "description": "Total revenue",
        "technicalDataType": "decimal",
        "path": "finance/metrics",
        "dataPublisher": "JPM",
    }
    assert attr.to_dict() == expected


def test_report_attribute_client_get_set(fusion_obj: Fusion) -> None:
    attr = ReportAttribute(name="revenue", title="Revenue")
    attr.client = fusion_obj
    assert attr.client == fusion_obj


def test_report_attributes_add_get_remove() -> None:
    attr = ReportAttribute(name="revenue", title="Revenue")
    attrs = ReportAttributes()
    attrs.add_attribute(attr)
    assert attrs.get_attribute("revenue") == attr
    assert attrs.remove_attribute("revenue")
    assert attrs.get_attribute("revenue") is None


def test_report_attributes_to_and_from_dict_list() -> None:
    data = [{
        "name": "revenue",
        "title": "Revenue",
        "description": "Total revenue",
        "technicalDataType": "decimal",
        "path": "finance/metrics",
        "dataPublisher": "JPM",
    }]
    attrs_instance = ReportAttributes()
    attrs = attrs_instance.from_dict_list(data)
    assert isinstance(attrs, ReportAttributes)
    assert attrs.to_dict() == {"attributes": data}


def test_report_attributes_from_dataframe() -> None:
    test_df = pd.DataFrame([{
        "name": "revenue",
        "title": "Revenue",
        "description": "Total revenue",
        "technicalDataType": "decimal",
        "path": "finance/metrics",
        "dataPublisher": "JPM",
    }])
    attrs_instance = ReportAttributes()
    attrs = attrs_instance.from_dataframe(test_df)
    assert isinstance(attrs, ReportAttributes)
    assert attrs.attributes[0].name == "revenue"


def test_report_attributes_from_csv(tmp_path: Path) -> None:
    file_path = tmp_path / "test.csv"
    test_df = pd.DataFrame([{
        "name": "revenue",
        "title": "Revenue",
        "description": "Total revenue",
        "technicalDataType": "decimal",
        "path": "finance/metrics",
        "dataPublisher": "JPM",
    }])
    test_df.to_csv(file_path, index=False)
    attrs = ReportAttributes().from_csv(str(file_path))
    assert isinstance(attrs, ReportAttributes)
    assert attrs.attributes[0].name == "revenue"


def test_report_attributes_from_object_dict() -> None:
    dict_list = [{"name": "revenue", "title": "Revenue"}]
    attrs = ReportAttributes().from_object(dict_list)
    assert isinstance(attrs, ReportAttributes)
    assert attrs.attributes[0].name == "revenue"


def test_report_attributes_from_object_dataframe() -> None:
    test_df = pd.DataFrame([{"name": "revenue", "title": "Revenue"}])
    attrs = ReportAttributes().from_object(test_df)
    assert isinstance(attrs, ReportAttributes)
    assert attrs.attributes[0].name == "revenue"


def test_report_attributes_from_object_invalid_type() -> None:
    with pytest.raises(TypeError):
        ReportAttributes().from_object("invalid_input")  # type: ignore[arg-type]


def test_report_attributes_to_dataframe() -> None:
    attr = ReportAttribute(name="revenue", title="Revenue")
    test_df = ReportAttributes([attr]).to_dataframe()
    assert test_df.shape[0] == 1
    assert test_df["name"].iloc[0] == "revenue"


def test_report_attributes_use_client_value_error() -> None:
    with pytest.raises(ValueError, match="A Fusion client object is required."):
        ReportAttributes()._use_client(None)


def test_report_attributes_create(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    HTTP_OK = 200
    report_id = "report_123"
    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/reports/{report_id}/reportElements"

    expected_payload = [{
        "name": "revenue",
        "title": "Revenue",
        "description": None,
        "technicalDataType": None,
        "path": None,
        "dataPublisher": None,
    }]
    requests_mock.post(url, json=expected_payload, status_code=HTTP_OK)

    attr = ReportAttribute(name="revenue", title="Revenue")
    attrs = ReportAttributes(attributes=[attr])
    attrs.client = fusion_obj

    response = attrs.create(report_id=report_id, return_resp_obj=True)
    assert isinstance(response, requests.Response)
    assert response.status_code == HTTP_OK
