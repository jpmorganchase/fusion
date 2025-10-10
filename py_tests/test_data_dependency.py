"""Test file for Fusion Data Dependency module following dataset.py style."""

import pytest
import requests
import requests_mock

from fusion import Fusion
from fusion.data_dependency import DataDependency, DataMapping, DependencyAttribute


def test_dependency_attribute_to_dict_and_validation() -> None:
    """Test DependencyAttribute creation, validation, and to_dict conversion."""
    # Valid Dataset with data_space
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
    with pytest.raises(ValueError, match="data_space is required when entity_type is 'Dataset'"):
        DependencyAttribute("Dataset", "dataset1", "colA")

def test_link_attributes(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test DataDependency link_attributes method."""
    data_dep = DataDependency(client=fusion_obj)

    src_attr = DependencyAttribute("Dataset", "dataset1", "colA", data_space="Finance")
    tgt_attr = DependencyAttribute("Dataset", "dataset2", "colB", data_space="Finance")

    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/data-dependencies/attributes"
    requests_mock.post(url, status_code=200)

    resp = data_dep.link_attributes([src_attr], tgt_attr, return_resp_obj=True)
    assert isinstance(resp, requests.Response)
    HTTP_OK = 200
    assert resp.status_code == HTTP_OK


def test_unlink_attributes(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test DataDependency unlink_attributes method."""
    data_dep = DataDependency(client=fusion_obj)

    src_attr = DependencyAttribute("Dataset", "dataset1", "colA", data_space="Finance")
    tgt_attr = DependencyAttribute("Dataset", "dataset2", "colB", data_space="Finance")

    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/data-dependencies/attributes"
    requests_mock.delete(url, status_code=200)

    resp = data_dep.unlink_attributes([src_attr], tgt_attr, return_resp_obj=True)
    assert isinstance(resp, requests.Response)
    HTTP_OK = 200
    assert resp.status_code == HTTP_OK

def test_link_attribute_to_term(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test DataMapping link_attribute_to_term method."""
    data_map = DataMapping(client=fusion_obj)
    attr = DependencyAttribute("Dataset", "dataset1", "colA", data_space="Finance")
    term = {"id": "term_123"}

    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/data-mapping/attributes/terms"
    requests_mock.post(url, status_code=200)

    resp = data_map.link_attribute_to_term(attr, term, is_kde=True, return_resp_obj=True)
    assert isinstance(resp, requests.Response)
    HTTP_OK = 200
    assert resp.status_code == HTTP_OK


def test_update_attribute_to_term_kde_status(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test DataMapping update_attribute_to_term_kde_status method."""
    data_map = DataMapping(client=fusion_obj)
    attr = DependencyAttribute("Dataset", "dataset1", "colA", data_space="Finance")
    term = {"id": "term_123"}

    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/data-mapping/attributes/terms"
    requests_mock.patch(url, status_code=200)

    resp = data_map.update_attribute_to_term_kde_status(attr, term, is_kde=False, return_resp_obj=True)
    assert isinstance(resp, requests.Response)
    HTTP_OK = 200
    assert resp.status_code == HTTP_OK


def test_unlink_attribute_from_term(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test DataMapping unlink_attribute_from_term method."""
    data_map = DataMapping(client=fusion_obj)
    attr = DependencyAttribute("Dataset", "dataset1", "colA", data_space="Finance")
    term = {"id": "term_123"}

    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/data-mapping/attributes/terms"
    requests_mock.delete(url, status_code=200)

    resp = data_map.unlink_attribute_from_term(attr, term, return_resp_obj=True)
    assert isinstance(resp, requests.Response)
    HTTP_OK = 200
    assert resp.status_code == HTTP_OK
