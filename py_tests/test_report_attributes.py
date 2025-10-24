from pathlib import Path

import pandas as pd
import pytest
import requests
import requests_mock

from fusion import Fusion
from fusion.report_attributes import ReportAttribute, ReportAttributes


def test_report_attribute_str_repr() -> None:
    attr = ReportAttribute(title="Revenue", source_identifier="rev-001")
    assert isinstance(str(attr), str)
    assert isinstance(repr(attr), str)


def test_report_attribute_to_dict() -> None:
    attr = ReportAttribute(
        title="Revenue",
        source_identifier="rev-001",
        description="Total revenue",
        technical_data_type="decimal",
        path="finance/metrics",
    )
    expected = {
        "id": None,
        "sourceIdentifier": "rev-001",
        "title": "Revenue",
        "description": "Total revenue",
        "technicalDataType": "decimal",
        "path": "finance/metrics",
    }
    assert attr.to_dict() == expected


def test_report_attribute_client_get_set(fusion_obj: Fusion) -> None:
    attr = ReportAttribute(title="Revenue", source_identifier="rev-001")
    attr.client = fusion_obj
    assert attr.client == fusion_obj


def test_report_attributes_add_get_remove() -> None:
    attr = ReportAttribute(title="Revenue", source_identifier="rev-001")
    attrs = ReportAttributes()
    attrs.add_attribute(attr)
    assert attrs.attributes[0].title == "Revenue"
    assert attrs.remove_attribute("Revenue") is True


def test_report_attributes_to_and_from_dict_list() -> None:
    data = [
        {
            "title": "Revenue",
            "sourceIdentifier": "rev-001",
            "description": "Total revenue",
            "technicalDataType": "decimal",
            "path": "finance/metrics",
        }
    ]
    expected_data = [
        {
            "id": None,
            "title": "Revenue",
            "sourceIdentifier": "rev-001",
            "description": "Total revenue",
            "technicalDataType": "decimal",
            "path": "finance/metrics",
        }
    ]
    attrs_instance = ReportAttributes()
    attrs = attrs_instance.from_dict_list(data)
    assert isinstance(attrs, ReportAttributes)
    assert attrs.to_dict() == {"attributes": expected_data}


def test_report_attributes_from_dataframe() -> None:
    test_df = pd.DataFrame(
        [
            {
                "title": "Revenue",
                "sourceIdentifier": "rev-001",
                "description": "Total revenue",
                "technicalDataType": "decimal",
                "path": "finance/metrics",
            }
        ]
    )
    attrs_instance = ReportAttributes()
    attrs = attrs_instance.from_dataframe(test_df)
    assert isinstance(attrs, ReportAttributes)
    assert attrs.attributes[0].title == "Revenue"


def test_report_attributes_from_csv(tmp_path: Path) -> None:
    file_path = tmp_path / "test.csv"
    test_df = pd.DataFrame(
        [
            {
                "Data Element Name": "Revenue",
                "Local Data Element Reference ID": "rev-001",
                "Data Element Description": "Total revenue",
            }
        ]
    )
    test_df.to_csv(file_path, index=False)
    attrs = ReportAttributes().from_csv(str(file_path))
    assert isinstance(attrs, ReportAttributes)
    assert attrs.attributes[0].title == "Revenue"


def test_report_attributes_from_object_dict() -> None:
    dict_list = [{"title": "Revenue", "sourceIdentifier": "rev-001"}]
    attrs = ReportAttributes().from_object(dict_list)
    assert isinstance(attrs, ReportAttributes)
    assert attrs.attributes[0].title == "Revenue"


def test_report_attributes_from_object_dataframe() -> None:
    test_df = pd.DataFrame([{"title": "Revenue", "sourceIdentifier": "rev-001"}])
    attrs = ReportAttributes().from_object(test_df)
    assert isinstance(attrs, ReportAttributes)
    assert attrs.attributes[0].title == "Revenue"


def test_report_attributes_from_object_csv(tmp_path: Path) -> None:
    file_path = tmp_path / "test.csv"
    test_df = pd.DataFrame(
        [
            {
                "Data Element Name": "Revenue",
                "Local Data Element Reference ID": "rev-001",
                "Data Element Description": "Total revenue",
            }
        ]
    )
    test_df.to_csv(file_path, index=False)
    attrs = ReportAttributes().from_object(str(file_path))
    assert isinstance(attrs, ReportAttributes)
    assert attrs.attributes[0].title == "Revenue"


def test_report_attributes_from_object_invalid_type() -> None:
    with pytest.raises(TypeError):
        ReportAttributes().from_object(123)  # type: ignore[arg-type]


def test_report_attributes_from_object_invalid_string() -> None:
    with pytest.raises(ValueError, match="String must be a .csv path or JSON array string."):
        ReportAttributes().from_object("invalid_string")


def test_report_attributes_to_dataframe() -> None:
    attr = ReportAttribute(title="Revenue", source_identifier="rev-001")
    test_df = ReportAttributes([attr]).to_dataframe()
    assert test_df.shape[0] == 1
    assert test_df["title"].iloc[0] == "Revenue"


def test_report_attributes_use_client_value_error() -> None:
    with pytest.raises(ValueError, match="A Fusion client object is required."):
        ReportAttributes()._use_client(None)


def test_report_attributes_create(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    HTTP_OK = 200
    report_id = "report_123"
    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/reports/{report_id}/attributes"

    expected_payload = [
        {
            "id": None,
            "sourceIdentifier": "rev-001",
            "title": "Revenue",
            "description": None,
            "technicalDataType": None,
            "path": None,
        }
    ]
    requests_mock.post(url, json=expected_payload, status_code=HTTP_OK)

    attr = ReportAttribute(title="Revenue", source_identifier="rev-001")
    attrs = ReportAttributes(attributes=[attr])
    attrs.client = fusion_obj

    response = attrs.create(report_id=report_id, return_resp_obj=True)
    assert isinstance(response, requests.Response)
    assert response.status_code == HTTP_OK


def test_report_attributes_update(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test the update method (PUT operation)."""
    HTTP_OK = 200
    report_id = "report_123"
    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/reports/{report_id}/attributes"

    expected_payload = [
        {
            "id": 456,
            "sourceIdentifier": "rev-001",
            "description": "Updated revenue field",
            "technicalDataType": "decimal",
            "path": "/data/revenue",
        }
    ]
    requests_mock.put(url, json=expected_payload, status_code=HTTP_OK)

    attr = ReportAttribute(
        id=456,
        title="Revenue",
        source_identifier="rev-001",
        description="Updated revenue field",
        technical_data_type="decimal",
        path="/data/revenue",
    )
    attrs = ReportAttributes(attributes=[attr])
    attrs.client = fusion_obj

    response = attrs.update(report_id=report_id, return_resp_obj=True)
    assert isinstance(response, requests.Response)
    assert response.status_code == HTTP_OK

    # Verify the request was made with correct data
    assert requests_mock.last_request is not None
    assert requests_mock.last_request.json() == expected_payload


def test_report_attributes_update_without_id_fails(fusion_obj: Fusion) -> None:
    """Test that update method fails when attribute doesn't have id."""
    attr = ReportAttribute(title="Revenue", source_identifier="rev-001")
    attrs = ReportAttributes(attributes=[attr])
    attrs.client = fusion_obj

    with pytest.raises(ValueError, match="must have an 'id' field for update"):
        attrs.update(report_id="report_123")


def test_report_attributes_update_fields(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test the update_fields method (PATCH operation)."""
    HTTP_OK = 200
    report_id = "report_123"
    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/reports/{report_id}/attributes"

    expected_payload = [
        {
            "id": 456,
            "description": "Updated revenue field",
            "technicalDataType": "decimal",
        }
    ]
    requests_mock.patch(url, json=expected_payload, status_code=HTTP_OK)

    # Create attribute with only id and fields to update (others are None)
    attr = ReportAttribute(
        id=456,
        title="Revenue",  # title is immutable, won't be in PATCH payload
        description="Updated revenue field",
        technical_data_type="decimal",
        # sourceIdentifier and path are None, so they are excluded from payload
    )
    attrs = ReportAttributes(attributes=[attr])
    attrs.client = fusion_obj

    response = attrs.update_fields(report_id=report_id, return_resp_obj=True)
    assert isinstance(response, requests.Response)
    assert response.status_code == HTTP_OK

    # Verify the request was made with correct data (only non-None updatable fields)
    assert requests_mock.last_request is not None
    assert requests_mock.last_request.json() == expected_payload


def test_report_attributes_update_fields_without_id_fails(fusion_obj: Fusion) -> None:
    """Test that update_fields method fails when attribute doesn't have id."""
    attr = ReportAttribute(title="Revenue", description="Some description")
    attrs = ReportAttributes(attributes=[attr])
    attrs.client = fusion_obj

    with pytest.raises(ValueError, match="must have an 'id' field for update_fields"):
        attrs.update_fields(report_id="report_123")


def test_report_attributes_delete(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test the delete method (DELETE operation)."""
    HTTP_OK = 200
    report_id = "report_123"
    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/reports/{report_id}/attributes"

    expected_payload = [{"id": 456}]
    requests_mock.delete(url, json=expected_payload, status_code=HTTP_OK)

    attr = ReportAttribute(id=456, title="Revenue")  # Only id is needed for deletion
    attrs = ReportAttributes(attributes=[attr])
    attrs.client = fusion_obj

    response = attrs.delete(report_id=report_id, return_resp_obj=True)
    assert isinstance(response, requests.Response)
    assert response.status_code == HTTP_OK

    # Verify the request was made with correct data (only id)
    assert requests_mock.last_request is not None
    assert requests_mock.last_request.json() == expected_payload


def test_report_attributes_delete_multiple(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test the delete method with multiple attributes."""
    HTTP_OK = 200
    report_id = "report_123"
    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/reports/{report_id}/attributes"

    expected_payload = [{"id": 456}, {"id": 789}]
    requests_mock.delete(url, json=expected_payload, status_code=HTTP_OK)

    attrs = ReportAttributes(
        attributes=[ReportAttribute(id=456, title="Revenue"), ReportAttribute(id=789, title="Profit")]
    )
    attrs.client = fusion_obj

    response = attrs.delete(report_id=report_id, return_resp_obj=True)
    assert isinstance(response, requests.Response)
    assert response.status_code == HTTP_OK

    # Verify the request was made with correct data
    assert requests_mock.last_request is not None
    assert requests_mock.last_request.json() == expected_payload


def test_report_attributes_delete_without_id_fails(fusion_obj: Fusion) -> None:
    """Test that delete method fails when attribute doesn't have id."""
    attr = ReportAttribute(title="Revenue", source_identifier="rev-001")
    attrs = ReportAttributes(attributes=[attr])
    attrs.client = fusion_obj

    with pytest.raises(ValueError, match="must have an 'id' field for deletion"):
        attrs.delete(report_id="report_123")


def test_report_attributes_methods_without_client_fails() -> None:
    """Test that all methods fail when no client is provided."""
    attr = ReportAttribute(id=456, title="Revenue")
    attrs = ReportAttributes(attributes=[attr])
    # No client set

    with pytest.raises(ValueError, match="A Fusion client object is required"):
        attrs.update(report_id="report_123")

    with pytest.raises(ValueError, match="A Fusion client object is required"):
        attrs.update_fields(report_id="report_123")

    with pytest.raises(ValueError, match="A Fusion client object is required"):
        attrs.delete(report_id="report_123")
