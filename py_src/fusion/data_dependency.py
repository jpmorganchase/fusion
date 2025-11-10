"""
Fusion Data Dependency module.

This module defines classes to manage dependencies and mappings between
attributes and business terms in Fusion.

Classes:
    DependencyAttribute: Represents an attribute participating in dependency or mapping relationships.
    DependencyMapping: Represents a mapping between source and target attributes.
    AttributeTermMapping: Represents a mapping between an attribute and a business term.
    DataDependency: Manages attribute-to-attribute dependencies.
    DataMapping: Manages attribute-to-term mappings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fusion.utils import requests_raise_for_status

if TYPE_CHECKING:
    import requests

    from fusion import Fusion


@dataclass
class DependencyAttribute:
    """Represents an attribute participating in dependency relationships.

    Args:
        entity_type (str): The type of entity, e.g., "Dataset".
        entity_identifier (str): Identifier of the entity.
        attribute_identifier (str): Identifier of the attribute.
        data_space (str | None, optional): Required if entity_type is "Dataset".

    Raises:
        ValueError: If `entity_type` is "Dataset" and `data_space` is not provided.

    Example:
        >>> from fusion import Fusion
        >>> fusion = Fusion()
        >>> attr = fusion.dependency_attribute("Dataset", "dataset1", "colA", "Finance")
    """

    entity_type: str
    entity_identifier: str
    attribute_identifier: str
    data_space: str | None = None

    def __post_init__(self) -> None:
        """Validate required attributes."""
        if self.entity_type.lower() == "dataset" and not self.data_space:
            raise ValueError("`data_space` is required when entity_type is 'Dataset'.")

    def to_dict(self) -> dict[str, Any]:
        """Convert the attribute to dictionary representation."""
        result = {
            "entityType": self.entity_type,
            "entityIdentifier": self.entity_identifier,
            "attributeIdentifier": self.attribute_identifier,
        }
        if self.entity_type.lower() == "dataset" and self.data_space is not None:
            result["dataSpace"] = self.data_space
        return result


@dataclass
class DependencyMapping:
    """Represents a dependency mapping between source and target attributes.

    Args:
        source_attributes (List[DependencyAttribute]): List of source dependency attributes.
        target_attribute (DependencyAttribute): Target dependency attribute.

    Example:
        >>> src_attr = DependencyAttribute(
        ...     dataspace="677834y",
        ...     entity_type="Dataset",
        ...     entity_identifier="data_asset_1",
        ...     attribute_identifier="attribute1"
        ... )
        >>> tgt_attr = DependencyAttribute(
        ...     dataspace="677834y",
        ...     entity_type="Dataset",
        ...     entity_identifier="data_asset_2",
        ...     attribute_identifier="attribute2"
        ... )
        >>> DependencyMapping(
        ...     source_attributes=[src_attr],
        ...     target_attribute=tgt_attr
        ... )
    """

    source_attributes: list[DependencyAttribute]
    target_attribute: DependencyAttribute

    def to_dict(self) -> dict[str, Any]:
        """Convert the DependencyMapping to a dictionary for API payload."""
        return {
            "sourceAttributes": [attr.to_dict() for attr in self.source_attributes],
            "targetAttribute": self.target_attribute.to_dict(),
        }


@dataclass
class AttributeTermMapping:
    """Represents a mapping between a single attribute and a term, with optional KDE flag.

    Attributes:
        attribute (DependencyAttribute): The attribute to link or unlink.
        term (dict[str, str]): Term information. Must include `"id"`.
        is_kde (bool | None): KDE status. Required for link and update operations.

    Example:
        >>> attr = fusion.dependency_attribute("Dataset", "dataset1", "colA", "Finance")
        >>> term = {"id": "term_123"}
        >>> mapping = AttributeTermMapping(attribute=attr, term=term, is_kde=True)
    """

    attribute: DependencyAttribute
    term: dict[str, str]
    is_kde: bool | None = None

    def to_link_payload(self) -> dict[str, Any]:
        """Convert mapping to payload for link or update operations.

        Raises:
            ValueError: If `is_kde` is not provided.

        Returns:
            dict[str, Any]: Payload for linking or updating the KDE status.
        """
        if self.is_kde is None:
            raise ValueError("`is_kde` must be provided for link or update operations.")
        return {
            "attribute": self.attribute.to_dict(),
            "term": self.term,
            "isKDE": self.is_kde,
        }

    def to_unlink_payload(self) -> dict[str, Any]:
        """Convert mapping to payload for unlink operation.

        Returns:
            dict[str, Any]: Payload for unlinking a term.
        """
        return {
            "attribute": self.attribute.to_dict(),
            "term": self.term,
        }


class DataDependency:
    """Manages attribute-to-attribute dependencies in Fusion.

    This class provides methods to:
        - Link source attributes to a target attribute
        - Unlink source attributes from a target attribute

    Attributes:
        client (Fusion | None): The Fusion client object. Automatically set when
            instantiated via ``fusion.data_dependency()``.

    Examples:
        >>> from fusion import Fusion
        >>> fusion = Fusion()
        >>> data_dep = fusion.data_dependency()
        >>> src = fusion.dependency_attribute("Dataset", "dataset1", "colA", "Finance")
        >>> tgt = fusion.dependency_attribute("Dataset", "dataset2", "colB", "Finance")
        >>> mapping = fusion.dependency_mapping([src], tgt)
        >>> data_dep.link_attributes([mapping])
    """

    _client: Fusion | None = field(init=False, repr=False, compare=False, default=None)

    @property
    def client(self) -> Fusion | None:
        """Return the Fusion client object."""
        return self._client

    @client.setter
    def client(self, client: Fusion | None) -> None:
        """Set the Fusion client object.

        Args:
            client (Fusion | None): Fusion client instance.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> data_dep = fusion.data_dependency()
            >>> data_dep.client = fusion
        """
        self._client = client

    def _use_client(self, client: Fusion | None) -> Fusion:
        """Resolve the client to use for API operations.

        Args:
            client (Fusion | None): Optional Fusion client override.

        Returns:
            Fusion: Resolved Fusion client.

        Raises:
            ValueError: If no client is provided or set on the object.
        """
        res = self._client if client is None else client
        if res is None:
            raise ValueError("A Fusion client object is required.")
        return res

    def link_attributes(
        self, mappings: list[DependencyMapping], client: Fusion | None = None, return_resp_obj: bool = False
    ) -> requests.Response | None:
        """Link source attributes to a target attribute.

        Args:
            mappings (list[DependencyMapping]): List of dependency mappings to link.
            client (Fusion | None, optional): Fusion client. Defaults to the instance's client.
            return_resp_obj (bool, optional): If True, returns the response object. Defaults to False.

        Returns:
            requests.Response | None: The HTTP response if ``return_resp_obj`` is True, else None.

        Raises:
            ValueError: If no client is provided or set on the object.
            requests.HTTPError: If the HTTP request fails.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> src = fusion.dependency_attribute("Dataset", "dataset1", "colA", "Finance")
            >>> tgt = fusion.dependency_attribute("Dataset", "dataset2", "colB", "Finance")
            >>> mapping = fusion.dependency_mapping([src], tgt)
            >>> fusion.data_dependency().link_attributes([mapping])
        """
        client = self._use_client(client)
        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/data-dependencies/attributes"
        payload = [m.to_dict() for m in mappings]

        resp = client.session.post(url, json=payload)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def unlink_attributes(
        self, mappings: list[DependencyMapping], client: Fusion | None = None, return_resp_obj: bool = False
    ) -> requests.Response | None:
        """Unlink source attributes from a target attribute.

        Args:
            mappings (list[DependencyMapping]): List of dependency mappings to unlink.
            client (Fusion | None, optional): Fusion client. Defaults to the instance's client.
            return_resp_obj (bool, optional): If True, returns the response object. Defaults to False.

        Returns:
            requests.Response | None: The HTTP response if ``return_resp_obj`` is True, else None.

        Raises:
            ValueError: If no client is provided or set on the object.
            requests.HTTPError: If the HTTP request fails.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> src = fusion.dependency_attribute("Dataset", "dataset1", "colA", "Finance")
            >>> tgt = fusion.dependency_attribute("Dataset", "dataset2", "colB", "Finance")
            >>> mapping = fusion.dependency_mapping([src], tgt)
            >>> fusion.data_dependency().unlink_attributes([mapping])
        """
        client = self._use_client(client)
        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/data-dependencies/attributes"
        payload = [m.to_dict() for m in mappings]

        resp = client.session.delete(url, json=payload)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None


class DataMapping:
    """Manages attribute-to-term mappings in Fusion.

    This class provides methods to:
        - Link attributes to business terms
        - Unlink attributes from business terms
        - Update KDE (Key Data Element) status

    Attributes:
        client (Fusion | None): The Fusion client object. Automatically set when
            instantiated via ``fusion.data_mapping()``.

    Examples:
        >>> from fusion import Fusion
        >>> fusion = Fusion()
        >>> data_map = fusion.data_mapping()
        >>> attr = fusion.dependency_attribute("Dataset", "dataset1", "colA", "Finance")
        >>> term = {"id": "term_123"}
        >>> mapping = fusion.attribute_term_mapping(attr, term, is_kde=True)
        >>> data_map.link_attribute_to_term([mapping])
    """

    _client: Fusion | None = field(init=False, repr=False, compare=False, default=None)

    @property
    def client(self) -> Fusion | None:
        """Return the Fusion client object."""
        return self._client

    @client.setter
    def client(self, client: Fusion | None) -> None:
        """Set the Fusion client object.

        Args:
            client (Fusion | None): Fusion client instance.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> data_map = fusion.data_mapping()
            >>> data_map.client = fusion
        """
        self._client = client

    def _use_client(self, client: Fusion | None) -> Fusion:
        """Resolve the client to use for API operations.

        Args:
            client (Fusion | None): Optional Fusion client override.

        Returns:
            Fusion: Resolved Fusion client.

        Raises:
            ValueError: If no client is provided or set on the object.
        """
        res = self._client if client is None else client
        if res is None:
            raise ValueError("A Fusion client object is required.")
        return res

    def link_attribute_to_term(
        self, mappings: list[AttributeTermMapping], client: Fusion | None = None, return_resp_obj: bool = False
    ) -> requests.Response | None:
        """Link attributes to business terms.

        Args:
            mappings (list[AttributeTermMapping]): List of attribute-term mappings with KDE status set.
            client (Fusion | None, optional): Fusion client. Defaults to the instance's client.
            return_resp_obj (bool, optional): If True, returns the response object. Defaults to False.

        Returns:
            requests.Response | None: The HTTP response if ``return_resp_obj`` is True, else None.

        Raises:
            ValueError: If no client is provided or set on the object.
            requests.HTTPError: If the HTTP request fails.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> attr = fusion.dependency_attribute("Dataset", "dataset1", "colA", "Finance")
            >>> term = {"id": "term_123"}
            >>> mapping = fusion.attribute_term_mapping(attr, term, is_kde=True)
            >>> fusion.data_mapping().link_attribute_to_term([mapping])
        """
        client = self._use_client(client)
        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/data-mapping/attributes/terms"
        payload = [m.to_link_payload() for m in mappings]

        resp = client.session.post(url, json=payload)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def unlink_attribute_from_term(
        self, mappings: list[AttributeTermMapping], client: Fusion | None = None, return_resp_obj: bool = False
    ) -> requests.Response | None:
        """Unlink attributes from business terms.

        Args:
            mappings (list[AttributeTermMapping]): List of attribute-term mappings without KDE status.
            client (Fusion | None, optional): Fusion client. Defaults to the instance's client.
            return_resp_obj (bool, optional): If True, returns the response object. Defaults to False.

        Returns:
            requests.Response | None: The HTTP response if ``return_resp_obj`` is True, else None.

        Raises:
            ValueError: If no client is provided or set on the object.
            requests.HTTPError: If the HTTP request fails.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> attr = fusion.dependency_attribute("Dataset", "dataset1", "colA", "Finance")
            >>> term = {"id": "term_123"}
            >>> mapping = fusion.attribute_term_mapping(attr, term)
            >>> fusion.data_mapping().unlink_attribute_from_term([mapping])
        """
        client = self._use_client(client)
        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/data-mapping/attributes/terms"
        payload = [m.to_unlink_payload() for m in mappings]

        resp = client.session.delete(url, json=payload)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def update_attribute_to_term_kde_status(
        self, mappings: list[AttributeTermMapping], client: Fusion | None = None, return_resp_obj: bool = False
    ) -> requests.Response | None:
        """Update KDE status for attribute-term mappings.

        Args:
            mappings (list[AttributeTermMapping]): List of attribute-term mappings with updated KDE status.
            client (Fusion | None, optional): Fusion client. Defaults to the instance's client.
            return_resp_obj (bool, optional): If True, returns the response object. Defaults to False.

        Returns:
            requests.Response | None: The HTTP response if ``return_resp_obj`` is True, else None.

        Raises:
            ValueError: If no client is provided or set on the object.
            requests.HTTPError: If the HTTP request fails.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> attr = fusion.dependency_attribute("Dataset", "dataset1", "colA", "Finance")
            >>> term = {"id": "term_123"}
            >>> mapping = fusion.attribute_term_mapping(attr, term, is_kde=False)
            >>> fusion.data_mapping().update_attribute_to_term_kde_status([mapping])
        """
        client = self._use_client(client)
        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/data-mapping/attributes/terms"
        payload = [m.to_link_payload() for m in mappings]

        resp = client.session.patch(url, json=payload)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None
