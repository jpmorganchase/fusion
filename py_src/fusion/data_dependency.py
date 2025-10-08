"""Fusion Data Dependency module.

This module defines classes to manage logical dependencies between metadata objects
like Datasets, Reports, and Business Terms in Fusion.

Classes:
    DataElement: Represents a single source or target element in a data dependency.
    DataDependency: Entry point for managing logical data dependencies.
    LogicalDataElements: Manages element-to-element dependencies.
    LogicalDataElementToGlossaryTerm: Manages element-to-term dependencies.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Any

from fusion.utils import snake_to_camel

if TYPE_CHECKING:
    import requests
    from fusion import Fusion


@dataclass
class DataAttribute:
    """Represents a single source or target Attribute in a data dependency.

    Attributes:
        entity_type (str): Type of entity, e.g., "Dataset" or "Report".
        entity_identifier (str): Identifier of the entity.
        attribute_identifier (str): Identifier of the attribute.
        data_space (Optional[str]): Required if entity_type is "Dataset".
        _client (Optional[Fusion]): Optional Fusion client.

    Raises:
        ValueError: If entity_type is "Dataset" and data_space is not provided.

    Examples:
        >>> from fusion import Fusion
        >>> from fusion.data_dependency import DataAttribute
        >>> fusion = Fusion()

        # Dataset
        >>> de = DataAttribute("Dataset", "dataset1", "colA", data_space="test")
        >>> de.to_dict()
        {'entityType': 'Dataset', 'entityIdentifier': 'dataset1', 'attributeIdentifier': 'colA', 'dataSpace': 'test1'}

        # Report
        >>> de2 = DataAttribute("Report", "report1", "fieldX")
        >>> de2.to_dict()
        {'entityType': 'Report', 'entityIdentifier': 'report1', 'attributeIdentifier': 'fieldX'}
    """

    entity_type: str
    entity_identifier: str
    attribute_identifier: str
    data_space: Optional[str] = None
    _client: Fusion | None = None

    def __post_init__(self) -> None:
        """Validate required attributes."""
        if self.entity_type.lower() == "dataset" and not self.data_space:
            raise ValueError("data_space is required when entity_type is 'Dataset'")

    def to_dict(self) -> dict[str, Any]:
        """Convert DataAttribute instance to dictionary.

        Returns:
            dict[str, Any]: Dictionary representation of the Data Attribute.
        """
        attribute_dict = {
            snake_to_camel(k): v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and v is not None
        }
        return attribute_dict

    @property
    def client(self) -> Fusion | None:
        """Return the Fusion client."""
        return self._client

    @client.setter
    def client(self, client: Fusion) -> None:
        """Set the Fusion client."""
        self._client = client


class LogicalDataElements:
    """Manager for logical Attribute-to-Attribute dependencies."""

    def __init__(self, client: Fusion):
        self._client = client

    def create(
    self,
    dependencies: list[dict[str, list[DataAttribute] | DataAttribute]],
    return_resp_obj: bool = False
) -> requests.Response | None:
        """Create logical Element-to-Element dependencies (supports multiple source elements).

        Args:
            dependencies (list[dict]): List of dependency mappings. Each dict should have:
                - "sourceAttributes": list of DataAttribute instances
                - "targetAttribute": a single DataAttribute instance
            return_resp_obj (bool): If True, returns the requests.Response object.

        Returns:
            requests.Response | None: Response from the Fusion API if return_resp_obj=True, else None.

        Examples:
            >>> from fusion import Fusion
            >>> from fusion.data_dependency import DataElement
            >>> fusion = Fusion()
            >>> logical_deps = fusion.data_dependency().logical_data_elements
            >>> deps = [
            ...     {
            ...         "sourceElements": [DataElement("Dataset", "dataset1", "colA", data_space="Finance")],
            ...         "targetElement": DataElement("Dataset", "dataset2", "colB", data_space="Finance")
            ...     }
            ... ]
            >>> logical_deps.create(deps)
        """
        url = f"{self._client._get_new_root_url()}/api/corelineage-service/v1/data-dependencies/attributes"

        # Convert DataElement objects to dictionaries
        payload = []
        for dep in dependencies:
            payload.append({
                "sourceElements": [el.to_dict() for el in dep["sourceElements"]],
                "targetElement": dep["targetElement"].to_dict()
            })

        resp = self._client.session.post(url, json=payload)
        return resp if return_resp_obj else None

    def delete(self, source_element: DataElement, target_element: DataElement) -> dict[str, Any]:
        """Delete a logical Element-to-Element dependency.

        Args:
            source_element (DataElement): Source element.
            target_element (DataElement): Target element.

        Returns:
            dict[str, Any]: Response from the Fusion API.

        Examples:
            >>> from fusion import Fusion
            >>> from fusion.data_dependency import DataElement
            >>> fusion = Fusion()
            >>> logical_deps = fusion.data_dependency().logical_data_elements
            >>> src = DataElement("Dataset", "dataset1", "colA", data_space="Finance")
            >>> tgt = DataElement("Dataset", "dataset2", "colB", data_space="Finance")
            >>> logical_deps.delete(src, tgt)
        """
        url = f"{self._client._get_new_root_url()}/api/corelineage-service/v1/data-dependencies/attributes"
        payload = {"sourceElement": source_element.to_dict(), "targetElement": target_element.to_dict()}
        return self._client.session.delete(url, json=payload)


class LogicalDataElementToGlossaryTerm:
    """Manager for logical element-to-business-term dependencies."""

    def __init__(self, client: Fusion):
        self._client = client

    def create(self, logical_data_element: DataElement, term: dict[str, str], is_kde: bool) -> dict[str, Any]:
        """Create a logical Element-to-Glossary-Term dependency.

        Args:
            logical_data_element (DataElement): Logical data element.
            term (dict[str, str]): Business term dictionary with 'id'.
            is_kde (bool): Whether this is a Key Data Element.

        Returns:
            dict[str, Any]: Response from the Fusion API.

        Examples:
            >>> from fusion import Fusion
            >>> from fusion.data_dependency import DataElement
            >>> fusion = Fusion()
            >>> element = DataElement("Dataset", "dataset1", "colA", data_space="Finance")
            >>> term_dict = {"id": "Revenue"}
            >>> fusion.data_dependency().logical_data_element_to_glossary_term.create(element, term_dict, True)
        """
        url = f"{self._client._get_new_root_url()}/api/corelineage-service/v1/datadependencies/elements/terms"
        payload = {"logicalDataElement": logical_data_element.to_dict(), "term": term, "isKDE": is_kde}
        return self._client.session.post(url, json=payload)

    def update(self, logical_data_element: DataElement, term: dict[str, str], is_kde: bool) -> dict[str, Any]:
        """Update a logical Element-to-Glossary-Term dependency.

        Args:
            logical_data_element (DataElement): Logical data element.
            term (dict[str, str]): Business term dictionary with 'id'.
            is_kde (bool): Whether this is a Key Data Element.

        Returns:
            dict[str, Any]: Response from the Fusion API.

        Examples:
            >>> from fusion import Fusion
            >>> from fusion.data_dependency import DataElement
            >>> fusion = Fusion()
            >>> element = DataElement("Dataset", "dataset1", "colA", data_space="Finance")
            >>> term_dict = {"id": "Revenue"}
            >>> fusion.data_dependency().logical_data_element_to_glossary_term.update(element, term_dict, False)
        """
        url = f"{self._client._get_new_root_url()}/api/corelineage-service/v1/datadependencies/elements/terms"
        payload = {"logicalDataElement": logical_data_element.to_dict(), "term": term, "isKDE": is_kde}
        return self._client.session.patch(url, json=payload)

    def delete(self, logical_data_element: DataElement, term: dict[str, str]) -> dict[str, Any]:
        """Delete a logical Element-to-Glossary-Term dependency.

        Args:
            logical_data_element (DataElement): Logical data element.
            term (dict[str, str]): Business term dictionary with 'id'.

        Returns:
            dict[str, Any]: Response from the Fusion API.

        Examples:
            >>> from fusion import Fusion
            >>> from fusion.data_dependency import DataElement
            >>> fusion = Fusion()
            >>> element = DataElement("Dataset", "dataset1", "colA", data_space="Finance")
            >>> term_dict = {"id": "Revenue"}
            >>> fusion.data_dependency().logical_data_element_to_glossary_term.delete(element, term_dict)
        """
        url = f"{self._client._get_new_root_url()}/api/corelineage-service/v1/datadependencies/elements/terms"
        payload = {"logicalDataElement": logical_data_element.to_dict(), "term": term}
        return self._client.session.delete(url, json=payload)


class DataDependency:
    """Fusion Data Dependency class for managing logical dependencies.

    Provides interfaces to manage:
        - Logical Data Element to Data Element dependencies
        - Logical Data Element to Business Term dependencies

    Accessed via:
        >>> from fusion import Fusion
        >>> fusion = Fusion()
        >>> data_dependency = fusion.data_dependency()
    """

    def __init__(self, client: Fusion):
        self._client = client
        self.logical_data_elements = LogicalDataElements(client)
        self.logical_data_element_to_glossary_term = LogicalDataElementToGlossaryTerm(client)
