"""Fusion Data Dependency module.

This module defines classes to manage dependencies and mappings between
attributes and business terms in Fusion.

Classes:
    DependencyAttribute: Represents an attribute participating in dependency or mapping relationships.
    DataDependency: Manages attribute-to-attribute dependencies.
    DataMapping: Manages attribute-to-term mappings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fusion.utils import snake_to_camel

if TYPE_CHECKING:
    import requests

    from fusion import Fusion


@dataclass
class DependencyAttribute:
    """Represents a source or target Attribute in a dependency or mapping relationship.

    Attributes:
        entity_type (str): The type of entity (e.g., "Dataset" or "Report").
        entity_identifier (str): Identifier of the entity (e.g., dataset name or report ID).
        attribute_identifier (str): Identifier of the specific attribute.
        data_space (Optional[str]): Data space for the attribute. Required if entity_type is "Dataset".
        _client (Optional[Fusion]): Optional Fusion client for API calls.

    Raises:
        ValueError: If `entity_type` is "Dataset" and `data_space` is not provided.

    Examples:
        >>> from fusion import Fusion
        >>> fusion = Fusion()
        >>> src = fusion.dependency_attribute(
        ...     entity_type="Dataset",
        ...     entity_identifier="dataset1",
        ...     attribute_identifier="columnA",
        ...     data_space="Finance"
        ... )
        >>> src.to_dict()
        {'entityType': 'Dataset', 'entityIdentifier': 'dataset1',
         'attributeIdentifier': 'columnA', 'dataSpace': 'Finance'}
    """

    entity_type: str
    entity_identifier: str
    attribute_identifier: str
    data_space: str | None = None
    _client: Fusion | None = None

    def __post_init__(self) -> None:
        """Validate required attributes."""
        if self.entity_type.lower() == "dataset" and not self.data_space:
            raise ValueError("data_space is required when entity_type is 'Dataset'")

    def to_dict(self) -> dict[str, Any]:
        """Convert DependencyAttribute instance to dictionary.

        Returns:
            dict[str, Any]: Dictionary representation of the attribute, with keys in camelCase.

        Examples:
            >>> attr = fusion.dependency_attribute("Dataset", "dataset1", "colA", "Finance")
            >>> attr.to_dict()
            {'entityType': 'Dataset', 'entityIdentifier': 'dataset1',
             'attributeIdentifier': 'colA', 'dataSpace': 'Finance'}
        """
        return {
            snake_to_camel(k): v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and v is not None
        }

    @property
    def client(self) -> Fusion | None:
        """Return the Fusion client.

        Returns:
            Fusion | None: The Fusion client associated with this attribute.
        """
        return self._client

    @client.setter
    def client(self, client: Fusion) -> None:
        """Set the Fusion client.

        Args:
            client (Fusion): The Fusion client instance to associate with this attribute.

        Examples:
            >>> attr = DependencyAttribute("Dataset", "dataset1", "colA", "Finance")
            >>> attr.client = fusion
        """
        self._client = client

class DataDependency:
    """Manages attribute-to-attribute dependencies.

    This class provides methods to:
        - Link source attributes to a target attribute
        - Unlink source attributes from a target attribute

    Examples:
        >>> from fusion import Fusion
        >>> fusion = Fusion()
        >>> src = fusion.dependency_attribute("Dataset", "dataset1", "colA", "Finance")
        >>> tgt = fusion.dependency_attribute("Dataset", "dataset2", "colB", "Finance")
        >>> fusion.data_dependency().link_attributes([src], tgt)
        >>> fusion.data_dependency().unlink_attributes([src], tgt)
    """

    def __init__(self, client: Fusion) -> None:
        """Initialize DataDependency with a Fusion client.

        Args:
            client (Fusion): The Fusion client instance.
        """
        self._client = client

    def link_attributes(
        self,
        source_attributes: list[DependencyAttribute],
        target_attribute: DependencyAttribute,
        return_resp_obj: bool = False
    ) -> requests.Response | None:
        """Link one or more source attributes to a target attribute.

        Args:
            source_attributes (list[DependencyAttribute]): List of source attributes to be linked.
            target_attribute (DependencyAttribute): The target attribute to which the sources will be linked.
            return_resp_obj (bool, optional): If True, returns the `requests.Response` object. Defaults to False.

        Returns:
            requests.Response | None: The HTTP response object if `return_resp_obj=True`, else None.

        Examples:
            >>> src = fusion.dependency_attribute("Dataset", "dataset1", "colA", "Finance")
            >>> tgt = fusion.dependency_attribute("Dataset", "dataset2", "colB", "Finance")
            >>> fusion.data_dependency().link_attributes([src], tgt)
        """
        url = f"{self._client._get_new_root_url()}/api/corelineage-service/v1/data-dependencies/attributes"
        payload = [
            {
                "sourceAttributes": [attr.to_dict() for attr in source_attributes],
                "targetAttribute": target_attribute.to_dict(),
            }
        ]
        resp = self._client.session.post(url, json=payload)
        return resp if return_resp_obj else None

    def unlink_attributes(
        self,
        source_attributes: list[DependencyAttribute],
        target_attribute: DependencyAttribute,
        return_resp_obj: bool = False
    ) -> requests.Response | None:
        """Unlink one or more source attributes from a target attribute.

        Args:
            source_attributes (list[DependencyAttribute]): List of source attributes to be unlinked.
            target_attribute (DependencyAttribute): The target attribute from which the sources will be unlinked.
            return_resp_obj (bool, optional): If True, returns the `requests.Response` object. Defaults to False.

        Returns:
            requests.Response | None: The HTTP response object if `return_resp_obj=True`, else None.

        Examples:
            >>> src = fusion.dependency_attribute("Dataset", "dataset1", "colA", "Finance")
            >>> tgt = fusion.dependency_attribute("Dataset", "dataset2", "colB", "Finance")
            >>> fusion.data_dependency().unlink_attributes([src], tgt)
        """
        url = f"{self._client._get_new_root_url()}/api/corelineage-service/v1/data-dependencies/attributes"
        payload = [
            {
                "sourceAttributes": [attr.to_dict() for attr in source_attributes],
                "targetAttribute": target_attribute.to_dict(),
            }
        ]
        resp = self._client.session.delete(url, json=payload)
        return resp if return_resp_obj else None

class DataMapping:
    """Manages attribute-to-term mappings.

    This class provides methods to:
        - Link an attribute to a business term
        - Unlink an attribute from a business term
        - Update KDE (Key Data Element) status for a mapping

    Examples:
        >>> from fusion import Fusion
        >>> fusion = Fusion()
        >>> attr = fusion.dependency_attribute("Dataset", "dataset1", "colA", "Finance")
        >>> term = {"id": "term_123"}
        >>> fusion.data_mapping().link_attribute_to_term(attr, term, True)
        >>> fusion.data_mapping().update_attribute_to_term_kde_status(attr, term, False)
        >>> fusion.data_mapping().unlink_attribute_from_term(attr, term)
    """

    def __init__(self, client: Fusion) -> None:
        """Initialize DataMapping with a Fusion client.

        Args:
            client (Fusion): The Fusion client instance.
        """
        self._client = client

    def link_attribute_to_term(
        self,
        attribute: DependencyAttribute,
        term: dict[str, str],
        is_kde: bool,
        return_resp_obj: bool = False
    ) -> requests.Response | None:
        """Link an attribute to a business term.

        Args:
            attribute (DependencyAttribute): The attribute object to link.
            term (dict[str, str]): The business term dictionary containing the term `id`.
            is_kde (bool): Whether the attribute is designated as a Key Data Element (KDE).
            return_resp_obj (bool, optional): If True, returns the `requests.Response` object. Defaults to False.

        Returns:
            requests.Response | None: The HTTP response object if `return_resp_obj=True`, else None.

        Examples:
            >>> attr = fusion.dependency_attribute("Dataset", "dataset1", "colA", "Finance")
            >>> term = {"id": "term_123"}
            >>> fusion.data_mapping().link_attribute_to_term(attr, term, True)
        """
        url = f"{self._client._get_new_root_url()}/api/corelineage-service/v1/data-mapping/attributes/terms"
        payload = [
            {
                "attribute": attribute.to_dict(),
                "term": term,
                "isKDE": is_kde
            }
        ]
        resp = self._client.session.post(url, json=payload)
        return resp if return_resp_obj else None

    def unlink_attribute_from_term(
        self,
        attribute: DependencyAttribute,
        term: dict[str, str],
        return_resp_obj: bool = False
    ) -> requests.Response | None:
        """Unlink an attribute from a business term.

        Args:
            attribute (DependencyAttribute): The attribute object to unlink.
            term (dict[str, str]): The business term dictionary containing the term `id`.
            return_resp_obj (bool, optional): If True, returns the `requests.Response` object. Defaults to False.

        Returns:
            requests.Response | None: The HTTP response object if `return_resp_obj=True`, else None.

        Examples:
            >>> attr = fusion.dependency_attribute("Dataset", "dataset1", "colA", "Finance")
            >>> term = {"id": "term_123"}
            >>> fusion.data_mapping().unlink_attribute_from_term(attr, term)
        """
        url = f"{self._client._get_new_root_url()}/api/corelineage-service/v1/data-mapping/attributes/terms"
        payload = [
            {
                "attribute": attribute.to_dict(),
                "term": term
            }
        ]
        resp = self._client.session.delete(url, json=payload)
        return resp if return_resp_obj else None

    def update_attribute_to_term_kde_status(
        self,
        attribute: DependencyAttribute,
        term: dict[str, str],
        is_kde: bool,
        return_resp_obj: bool = False
    ) -> requests.Response | None:
        """Update the KDE (Key Data Element) status for an attribute-to-term mapping.

        Args:
            attribute (DependencyAttribute): The attribute object.
            term (dict[str, str]): The business term dictionary containing the term `id`.
            is_kde (bool): The KDE status to set (True for KDE, False for non-KDE).
            return_resp_obj (bool, optional): If True, returns the `requests.Response` object. Defaults to False.

        Returns:
            requests.Response | None: The HTTP response object if `return_resp_obj=True`, else None.

        Examples:
            >>> attr = fusion.dependency_attribute("Dataset", "dataset1", "colA", "Finance")
            >>> term = {"id": "term_123"}
            >>> fusion.data_mapping().update_attribute_to_term_kde_status(attr, term, False)
        """
        url = f"{self._client._get_new_root_url()}/api/corelineage-service/v1/data-mapping/attributes/terms"
        payload = [
            {
                "attribute": attribute.to_dict(),
                "term": term,
                "isKDE": is_kde
            }
        ]
        resp = self._client.session.patch(url, json=payload)
        return resp if return_resp_obj else None
