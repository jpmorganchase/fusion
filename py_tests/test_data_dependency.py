"""Tests for Fusion Data Dependency module."""

import pytest
import requests
import requests_mock

from fusion import Fusion
from fusion.data_dependency import (
    AttributeTermMapping,
    DataDependency,
    DataMapping,
    DependencyAttribute,
    DependencyMapping,
)


def test_dependency_attribute_to_dict_and_validation() -> None:
    """Test DependencyAttribute creation, validation, and to_dict conversion."""
    attr = DependencyAttribute("Dataset", "dataset1", "colA", data_space="Finance")
    expected_dict = {
        "entityType": "Dataset",
        "entityIdentifier": "dataset1",
        "attributeIdentifier": "colA",
        "dataSpace": "Finance",
    }
    assert attr.to_dict() == expected_dict

    # Valid Report without data_space
    report_attr = DependencyAttribute("Report", "report1", "fieldX")
    expected_report_dict = {
        "entityType": "Report",
        "entityIdentifier": "report1",
        "attributeIdentifier": "fieldX",
    }
    assert report_attr.to_dict() == expected_report_dict

    # Invalid Dataset without data_space should raise ValueError
    with pytest.raises(ValueError, match="`data_space` is required when entity_type is 'Dataset'"):
        DependencyAttribute("Dataset", "dataset1", "colA")


def test_dependency_mapping_to_dict() -> None:
    """Test DependencyMapping payload structure."""
    src = DependencyAttribute("Dataset", "src_dataset", "src_col", data_space="Finance")
    tgt = DependencyAttribute("Dataset", "tgt_dataset", "tgt_col", data_space="Finance")
    mapping = DependencyMapping([src], tgt)

    mapping_dict = mapping.to_dict()
    assert "sourceAttributes" in mapping_dict
    assert "targetAttribute" in mapping_dict
    assert mapping_dict["sourceAttributes"][0]["entityIdentifier"] == "src_dataset"
    assert mapping_dict["targetAttribute"]["entityIdentifier"] == "tgt_dataset"


def test_link_attributes(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test DataDependency.link_attributes with mocked POST call."""
    data_dep = DataDependency()
    data_dep.client = fusion_obj

    src_attr = DependencyAttribute("Dataset", "dataset1", "colA", data_space="Finance")
    tgt_attr = DependencyAttribute("Dataset", "dataset2", "colB", data_space="Finance")
    mapping = DependencyMapping([src_attr], tgt_attr)

    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/data-dependencies/attributes"
    requests_mock.post(url, status_code=200)

    resp = data_dep.link_attributes([mapping], return_resp_obj=True)
    assert isinstance(resp, requests.Response)
    HTTP_STATUS_OK = 200
    assert resp.status_code == HTTP_STATUS_OK


def test_unlink_attributes(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test DataDependency.unlink_attributes with mocked DELETE call."""
    data_dep = DataDependency()
    data_dep.client = fusion_obj

    src_attr = DependencyAttribute("Dataset", "dataset1", "colA", data_space="Finance")
    tgt_attr = DependencyAttribute("Dataset", "dataset2", "colB", data_space="Finance")
    mapping = DependencyMapping([src_attr], tgt_attr)

    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/data-dependencies/attributes"
    requests_mock.delete(url, status_code=200)

    resp = data_dep.unlink_attributes([mapping], return_resp_obj=True)
    assert isinstance(resp, requests.Response)
    HTTP_STATUS_OK = 200
    assert resp.status_code == HTTP_STATUS_OK


def test_attribute_term_mapping_link_payload() -> None:
    """Test AttributeTermMapping link payload generation."""
    attr = DependencyAttribute("Dataset", "dataset1", "colA", data_space="Finance")
    term = {"id": "term_123"}
    mapping = AttributeTermMapping(attr, term, is_kde=True)

    payload = mapping.to_link_payload()
    assert payload["attribute"]["entityIdentifier"] == "dataset1"
    assert payload["term"]["id"] == "term_123"
    assert payload["isKDE"] is True


def test_attribute_term_mapping_unlink_payload() -> None:
    """Test AttributeTermMapping unlink payload generation."""
    attr = DependencyAttribute("Dataset", "dataset1", "colA", data_space="Finance")
    term = {"id": "term_123"}
    mapping = AttributeTermMapping(attr, term)

    payload = mapping.to_unlink_payload()
    assert "isKDE" not in payload
    assert payload["attribute"]["entityIdentifier"] == "dataset1"


def test_link_attribute_to_term(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test DataMapping.link_attribute_to_term with mocked POST call."""
    data_map = DataMapping()
    data_map.client = fusion_obj

    attr = DependencyAttribute("Dataset", "dataset1", "colA", data_space="Finance")
    term = {"id": "term_123"}
    mapping = AttributeTermMapping(attr, term, is_kde=True)

    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/data-mapping/attributes/terms"
    requests_mock.post(url, status_code=200)

    resp = data_map.link_attribute_to_term([mapping], return_resp_obj=True)
    assert isinstance(resp, requests.Response)
    HTTP_STATUS_OK = 200
    assert resp.status_code == HTTP_STATUS_OK


def test_update_attribute_to_term_kde_status(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test DataMapping.update_attribute_to_term_kde_status with mocked PATCH call."""
    data_map = DataMapping()
    data_map.client = fusion_obj

    attr = DependencyAttribute("Dataset", "dataset1", "colA", data_space="Finance")
    term = {"id": "term_123"}
    mapping = AttributeTermMapping(attr, term, is_kde=False)

    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/data-mapping/attributes/terms"
    requests_mock.patch(url, status_code=200)

    resp = data_map.update_attribute_to_term_kde_status([mapping], return_resp_obj=True)
    assert isinstance(resp, requests.Response)
    HTTP_STATUS_OK = 200
    assert resp.status_code == HTTP_STATUS_OK


def test_unlink_attribute_from_term(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test DataMapping.unlink_attribute_from_term with mocked DELETE call."""
    data_map = DataMapping()
    data_map.client = fusion_obj

    attr = DependencyAttribute("Dataset", "dataset1", "colA", data_space="Finance")
    term = {"id": "term_123"}
    mapping = AttributeTermMapping(attr, term)

    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/data-mapping/attributes/terms"
    requests_mock.delete(url, status_code=200)

    resp = data_map.unlink_attribute_from_term([mapping], return_resp_obj=True)
    assert isinstance(resp, requests.Response)    
    HTTP_STATUS_OK = 200
    assert resp.status_code == HTTP_STATUS_OK