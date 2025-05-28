import pandas as pd
import pytest
import requests
import requests_mock

from fusion import Fusion
from fusion.report_attributes import ReportAttribute, ReportAttributes


def test_report_attribute_repr_str() -> None:
    attr = ReportAttribute(name="revenue", title="Revenue")
    assert str(attr)
    assert repr(attr)


def test_report_attribute_get_set_client(fusion_obj: Fusion) -> None:
    attr = ReportAttribute(name="revenue", title="Revenue")
    attr.client = fusion_obj
    assert attr.client == fusion_obj


def test_report_attribute_to_dict() -> None:
    attr = ReportAttribute(
        name="revenue",
        title="Revenue",
        description="Total revenue",
        technicalDataType="decimal",
        path="finance/metrics",
        dataPublisher="JPM",
    )
    result = attr.to_dict()
    expected = {
        "name": "revenue",
        "title": "Revenue",
        "description": "Total revenue",
        "technicalDataType": "decimal",
        "path": "finance/metrics",
        "dataPublisher": "JPM",
    }
    assert result == expected


def test_report_attributes_add_get_remove() -> None:
    attr = ReportAttribute(name="revenue", title="Revenue")
    attrs = ReportAttributes()
    attrs.add_attribute(attr)
    assert attrs.get_attribute("revenue") == attr
    assert attrs.remove_attribute("revenue")
    assert attrs.get_attribute("revenue") is None


def test_report_attributes_from_and_to_dict() -> None:
    data = [
        {
            "name": "revenue",
            "title": "Revenue",
            "description": "Total revenue",
            "technicalDataType": "decimal",
            "path": "finance/metrics",
            "dataPublisher": "JPM",
        }
    ]
    attrs = ReportAttributes._from_dict_list(data)
    assert isinstance(attrs, ReportAttributes)
    assert attrs.to_dict() == {"attributes": data}


def test_report_attributes_from_dataframe() -> None:
    test_df = pd.DataFrame(
        [
            {
                "name": "revenue",
                "title": "Revenue",
                "description": "Total revenue",
                "technicalDataType": "decimal",
                "path": "finance/metrics",
                "dataPublisher": "JPM",
            }
        ]
    )
    attrs = ReportAttributes._from_dataframe(test_df)
    assert isinstance(attrs, ReportAttributes)
    assert attrs.attributes[0].name == "revenue"


def test_report_attributes_from_object() -> None:
    dict_list = [
        {
            "name": "revenue",
            "title": "Revenue",
        }
    ]
    attrs = ReportAttributes().from_object(dict_list)
    assert isinstance(attrs, ReportAttributes)
    assert attrs.attributes[0].title == "Revenue"


def test_report_attributes_to_dataframe() -> None:
    attr = ReportAttribute(name="revenue", title="Revenue")
    test_df = ReportAttributes([attr]).to_dataframe()
    assert test_df.shape[0] == 1
    assert test_df["name"].iloc[0] == "revenue"


def test_report_attributes_use_client_value_error() -> None:
    with pytest.raises(ValueError, match="A Fusion client object is required."):
        ReportAttributes()._use_client(None)


def test_report_attributes_register(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    HTTP_OK = 200

    report_id = "report_123"
    url = f"{fusion_obj.root_url}metadata-lineage/report/{report_id}/attributes"
    expected_payload = [
        {
            "name": "revenue",
            "title": "Revenue",
            "description": None,
            "technicalDataType": None,
            "path": None,
            "dataPublisher": None,
        }
    ]
    requests_mock.post(url, json=expected_payload)

    test_attr = ReportAttribute(name="revenue", title="Revenue")
    test_attrs = ReportAttributes(attributes=[test_attr])
    test_attrs.client = fusion_obj

    resp = test_attrs.register(report_id=report_id, return_resp_obj=True)
    assert isinstance(resp, requests.Response)
    assert resp.status_code == HTTP_OK
