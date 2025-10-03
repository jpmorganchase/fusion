"""Test file for Data Dependency integration."""

import pytest
from fusion.data_dependency import DataDependency, DataElement
import requests
import requests_mock
from fusion import Fusion


def test_data_element_valid_and_to_dict():
    """Test DataElement creation, validation, and to_dict conversion."""
    # Valid Dataset with data_space
    element = DataElement("Dataset", "dataset1", "colA", data_space="Finance")
    expected_dict = {
        "entityType": "Dataset",
        "entityIdentifier": "dataset1",
        "elementIdentifier": "colA",
        "dataSpace": "Finance",
    }
    assert element.to_dict() == expected_dict

    # Valid Report without data_space
    report_element = DataElement("Report", "report1", "fieldX")
    expected_report_dict = {
        "entityType": "Report",
        "entityIdentifier": "report1",
        "elementIdentifier": "fieldX",
    }
    assert report_element.to_dict() == expected_report_dict

    # Invalid Dataset without data_space should raise ValueError
    with pytest.raises(ValueError, match="data_space is required when entity_type is 'Dataset'"):
        DataElement("Dataset", "dataset1", "colA")


def test_create_logical_data_elements(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test create LogicalDataElements method."""
    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/datadependencies/elements"
    requests_mock.post(url, status_code=200)

    source = DataElement("Dataset", "dataset1", "colA", data_space="Finance")
    target = DataElement("Dataset", "dataset2", "colB", data_space="Finance")

    resp = fusion_obj.data_dependency().logical_data_elements.create(
        source_element=source, target_element=target, return_resp_obj=True
    )
    assert isinstance(resp, requests.Response)
    assert resp.status_code == 200


def test_delete_logical_data_elements(requests_mock: requests_mock.Mocker, fusion_obj: Fusion) -> None:
    """Test delete LogicalDataElements method."""
    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/datadependencies/elements"
    requests_mock.delete(url, status_code=200)

    source = DataElement("Dataset", "dataset1", "colA", data_space="Finance")
    target = DataElement("Dataset", "dataset2", "colB", data_space="Finance")

    resp = fusion_obj.data_dependency().logical_data_elements.delete(
        source_element=source, target_element=target
    )
    assert isinstance(resp, requests.Response)
    assert resp.status_code == 200


def test_create_logical_data_element_to_glossary_term(
    requests_mock: requests_mock.Mocker, fusion_obj: Fusion
) -> None:
    """Test create LogicalDataElementToGlossaryTerm method."""
    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/datadependencies/elements/terms"
    requests_mock.post(url, status_code=200)

    element = DataElement("Dataset", "dataset1", "colA", data_space="Finance")
    term = {"id": "term_id"}

    resp = fusion_obj.data_dependency().logical_data_element_to_glossary_term.create(
        logical_data_element=element, term=term, is_kde=True
    )
    assert isinstance(resp, requests.Response)
    assert resp.status_code == 200


def test_update_logical_data_element_to_glossary_term(
    requests_mock: requests_mock.Mocker, fusion_obj: Fusion
) -> None:
    """Test update LogicalDataElementToGlossaryTerm method."""
    url = f"{fusion_obj._get_new_root_url()}/api/corelineage-service/v1/datadependencies/elements/terms"
    requests_mock.patch(url, status_code=200)

    element = DataElement("Dataset", "dataset1", "colA", data_space="Finance")
    term = {"id": "term_id"}

    resp = fusion_obj.data_dependency().logical_data_element_to_glossary_term.update(
        logical_data_element=element, term=term, is_kde=False
    )
    assert isinstance(resp, requests.Response)
    assert resp.status_code == 200


def test_delete_logical_data_element_to_glossary_term(
    requests_mock: requests_mock.Mocker, fusion_obj: Fusion
) -> None:
    """Test delete LogicalDataElementToGlossaryTerm method."""
    url = f"{fusion_obj.root_url}/api/corelineage-service/v1/datadependencies/elements/terms"
    requests_mock.delete(url, status_code=200)

    element = DataElement("Dataset", "dataset1", "colA", data_space="Finance")
    term = {"id": "term_id"}

    resp = fusion_obj.data_dependency().logical_data_element_to_glossary_term.delete(
        logical_data_element=element, term=term, is_kde=True
    )
    assert isinstance(resp, requests.Response)
    assert resp.status_code == 200


def test_data_dependency_instantiation(fusion_obj):
    """Test DataDependency class initialization and attribute access."""
    data_dep = fusion_obj.data_dependency()
    assert data_dep.logical_data_elements is not None
    assert data_dep.logical_data_element_to_glossary_term is not None
    assert data_dep._client == fusion_obj
