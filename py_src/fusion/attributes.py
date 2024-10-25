"""Fusion Product class and functions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from fusion.fusion_types import Types
from fusion.utils import convert_date_format, make_bool, tidy_string

if TYPE_CHECKING:
    import requests

    from fusion import Fusion


@dataclass
class Attribute:
    """Fusion Attribute class for managing attributes metadata in a Fusion catalog.

    Attributes:
        identifier (str): The unique identifier for the attribute.
        index (int): Attribute index.
        dataType (str | Types, optional): Datatype of attribute. Defaults to "String".
        title (str, optional): Attribute title. If not provided, defaults to identifier.
        description (str, optional): Attribute description. If not provided, defaults to identifier.
        isDatasetKey (bool, optional): Flag for primary keys. Defaults to False.
        source (str | None, optional): Name of data vendor which provided the data. Defaults to None.
        sourceFieldId (str | None, optional): Original identifier of attribute, if attribute has been renamed.
            If not provided, defaults to identifier.
        isInternalDatasetKey (bool | None, optional): Flag for internal primary keys. Defaults to None.
        isExternallyVisible (bool | None, optional): Flag for externally visible attributes. Defaults to True.
        unit (Any | None, optional): Unit of attribute. Defaults to None.
        multiplier (float, optional): Multiplier for unit. Defaults to 1.0.
        isPropogationEligible (bool | None, optional): Flag for propogation eligibility. Defaults to None.
        isMetric (bool | None, optional): Flag for attributes that are metrics. Defaults to None.
        availableFrom (str | None, optional): Date from which the attribute is available. Defaults to None.
        deprecatedFrom (str | None, optional): Date from which the attribute is deprecated. Defaults to None.
        term (str, optional): Term. Defaults to "bizterm1".
        dataset (int | None, optional): Dataset. Defaults to None.
        attributeType (str | None, optional): Attribute type. Defaults to None.
        _client (Fusion | None, optional): Fusion client object. Defaults to None.

    """

    identifier: str
    index: int
    dataType: Types = cast(Types, Types.String)
    title: str = ""
    description: str = ""
    isDatasetKey: bool = False
    source: str | None = None
    sourceFieldId: str | None = None
    isInternalDatasetKey: bool | None = None
    isExternallyVisible: bool | None = True
    unit: Any | None = None  # add units handling
    multiplier: float = 1.0
    isPropogationEligible: bool | None = None
    isMetric: bool | None = None
    availableFrom: str | None = None
    deprecatedFrom: str | None = None
    term: str = "bizterm1"
    dataset: int | None = None
    attributeType: str | None = None

    _client: Fusion | None = None

    def set_client(self, client: Any) -> None:
        """Set the client for the Attribute. Set automatically, if the Attribute is instantiated from a Fusion object.

        Args:
            client (Any): Fusion client object.

        Examples:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> attribute = fusion.attribute(identifier="my_attribute", index=0)
            >>> attribute.set_client(fusion)

        """
        self._client = client

    def __str__(self: Attribute) -> str:
        """Format string representation."""
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"Attribute(\n" + ",\n ".join(f"{k}={v!r}" for k, v in attrs.items()) + "\n)"

    def __repr__(self: Attribute) -> str:
        """Format object representation."""
        s = ", ".join(f"{getattr(self, f.name)!r}" for f in fields(self) if not f.name.startswith("_"))
        return "(" + s + ")"

    def __post_init__(self: Attribute) -> None:
        """Format Attribute metadata fields after object initialization."""
        self.isDatasetKey = make_bool(self.isDatasetKey)
        self.identifier = tidy_string(self.identifier).lower().replace(" ", "_")
        self.title = tidy_string(self.title) if self.title != "" else self.identifier.replace("_", " ").title()
        self.description = tidy_string(self.description) if self.description and self.description != "" else self.title
        self.sourceFieldId = (
            tidy_string(self.sourceFieldId).lower().replace(" ", "_") if self.sourceFieldId else self.identifier
        )
        self.availableFrom = convert_date_format(self.availableFrom) if self.availableFrom else None
        self.deprecatedFrom = convert_date_format(self.deprecatedFrom) if self.deprecatedFrom else None
        self.dataType = Types[str(self.dataType).strip().rsplit(".", maxsplit=1)[-1].title()]

    @classmethod
    def from_series(
        cls: type[Attribute],
        series: pd.Series[Any],
    ) -> Attribute:
        """Instantiate an Attribute object from a pandas Series.

        Args:
            series (pd.Series[Any]): Attribute metadata as a pandas Series.

        Returns:
            Attribute: Attribute object.

        Examples:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> import pandas as pd
            >>> series = pd.Series({
            ...     "identifier": "my_attribute",
            ...     "index": 0,
            ...     "dataType": "String",
            ...     "title": "My Attribute",
            ...     "description": "My attribute description"
            ... })
            >>> attribute = fusion.attribute(identifier="my_attribute", index=0).from_series(series)

        """
        series = series.rename(lambda x: x.replace(" ", "").replace("_", "").lower()).replace(
            to_replace=np.nan, value=None
        )
        dataType = series.get("datatype", cast(Types, Types.String))
        if dataType is None:
            dataType = series.get("type", cast(Types, Types.String))

        source = series.get("source", None)
        source = source.strip() if isinstance(source, str) else source

        isPropogationEligible = series.get("ispropogationeligible", None)
        isPropogationEligible = (
            make_bool(isPropogationEligible) if isPropogationEligible is not None else isPropogationEligible
        )
        isMetric = series.get("ismetric", None)
        isMetric = make_bool(isMetric) if isMetric is not None else isMetric
        isInternalDatasetKey = series.get("isinternaldatasetkey", None)
        isInternalDatasetKey = (
            make_bool(isInternalDatasetKey) if isInternalDatasetKey is not None else isInternalDatasetKey
        )
        isExternallyVisible = series.get("isexternallyvisible", True)
        isExternallyVisible = make_bool(isExternallyVisible) if isExternallyVisible is not None else isExternallyVisible

        return cls(
            identifier=series.get("identifier", "").strip(),
            index=series.get("index", -1),
            dataType=Types[dataType.strip().split(".")[-1].title()],
            title=series.get("title", ""),
            description=series.get("description", ""),
            isDatasetKey=series.get("isdatasetkey", False),
            source=source,
            sourceFieldId=series.get("sourcefieldid", None),
            isInternalDatasetKey=isInternalDatasetKey,
            isExternallyVisible=isExternallyVisible,
            unit=series.get("unit", None),
            multiplier=series.get("multiplier", 1.0),
            isPropogationEligible=isPropogationEligible,
            isMetric=isMetric,
            availableFrom=series.get("availablefrom", None),
            deprecatedFrom=series.get("deprecatedfrom", None),
            term=series.get("term", "bizterm1"),
            dataset=series.get("dataset", None),
            attributeType=series.get("attributetype", None),
        )

    @classmethod
    def from_dict(cls: type[Attribute], data: dict[str, Any]) -> Attribute:
        """Instantiate an Attribute object from a dictionary.

        Args:
            data (dict[str, Any]): Attribute metadata as a dictionary.

        Returns:
            Attribute: Attribute object.

        Examples:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> data = {
            ...     "identifier": "my_attribute",
            ...     "index": 0,
            ...     "dataType": "String",
            ...     "title": "My Attribute",
            ...     "description": "My attribute description"
            ... }
            >>> attribute = fusion.attribute(identifier="my_attribute", index=0).from_dict(data)

        """
        keys = [f.name for f in fields(cls)]
        data = {k: (None if pd.isna(v) else v) for k, v in data.items() if k in keys}
        if "dataType" in data:
            data["dataType"] = Types[data["dataType"].strip().rsplit(".", maxsplit=1)[-1].title()]
        return cls(**data)

    def to_dict(self: Attribute) -> dict[str, Any]:
        """Convert object to dictionary.

        Returns:
            dict[str, Any]: Attribute metadata as a dictionary.

        Examples:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> attribute = fusion.attribute(identifier="my_attribute", index=0)
            >>> attribute_dict = attribute.to_dict()
    
        """
        result = asdict(self)
        result["unit"] = str(self.unit) if self.unit is not None else None
        result["dataType"] = self.dataType.name
        result.pop("_client")
        return result

    def create(
        self,
        dataset: str,
        catalog: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Upload a new attribute to a Fusion catalog.

        Args:
            dataset (str): Dataset identifier.
            client (Fusion, optional): A Fusion client object. Defaults to the instance's _client.
                If instantiated from a Fusion object, then the client is set automatically.
            catalog (str, optional): A catalog identifier. Defaults to None.
            return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

        Returns:
            requests.Response | None: The response object from the API call if return_resp_obj is True, otherwise None.

        Examples:

            Individually, from scratch:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> attribute0 = fusion.attribute(identifier="my_attribute_0", index=0)
            >>> attribute.create(dataset="my_dataset", catalog="my_catalog")
            >>> attribute1 = fusion.attribute(identifier="my_attribute_1", index=1)
            >>> attribute.create(dataset="my_dataset", catalog="my_catalog")

            Individually, from a dictionary:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> data = {
            ...     "identifier": "my_attribute",
            ...     "index": 0,
            ...     "dataType": "String",
            ...     "title": "My Attribute",
            ...     "description": "My attribute description"
            ...    }
            >>> attribute = fusion.attribute(identifier="my_attribute", index=0).from_dict(data)
            >>> attribute.create(dataset="my_dataset", catalog="my_catalog")

            Individually, from a pandas Series:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> import pandas as pd
            >>> series = pd.Series({
            ...     "identifier": "my_attribute",
            ...     "index": 0,
            ...     "dataType": "String",
            ...     "title": "My Attribute",
            ...     "description": "My attribute description"
            ... })
            >>> attribute = fusion.attribute(identifier="my_attribute", index=0).from_series(series)
            >>> attribute.create(dataset="my_dataset", catalog="my_catalog")

        """
        client = self._client if client is None else client

        if client is None:
            raise ValueError("Client must be provided")
        catalog = client._use_catalog(catalog)
        data = self.to_dict()
        url = f"{client.root_url}catalogs/{catalog}/datasets/{dataset}/attributes/{self.identifier}"
        resp = client.session.put(url, json=data)
        resp.raise_for_status()
        return resp if return_resp_obj else None

    def delete(
        self,
        dataset: str,
        catalog: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Delete an Attribute from a Fusion catalog.

        Args:
            dataset (str): Dataset identifier.
            client (Fusion, optional): A Fusion client object. Defaults to the instance's _client.
            catalog (str, optional): A catalog identifier. Defaults to None.
            return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

        Returns:
            requests.Response | None: The response object from the API call if return_resp_obj is True, otherwise None.

        Examples:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> fusion.attribute(identifier="my_attribute", index=0).delete(dataset="my_dataset", catalog="my_catalog")

        """
        client = self._client if client is None else client
        if client is None:
            raise ValueError("Client must be provided")
        catalog = client._use_catalog(catalog)
        url = f"{client.root_url}catalogs/{catalog}/datasets/{dataset}/attributes/{self.identifier}"
        resp = client.session.delete(url)
        resp.raise_for_status()
        return resp if return_resp_obj else None


@dataclass
class Attributes:
    """Class representing a collection of Attribute instances for managing atrribute metadata in a Fusion catalog.
    
    Attributes:
        attributes (list[Attribute]): List of Attribute instances.
        _client (Fusion | None): Fusion client object.
        
    """

    attributes: list[Attribute] = field(default_factory=list)

    _client: Fusion | None = None

    def __str__(self) -> str:
        """String representation of the Attributes collection."""
        return (
            f"[\n" + ",\n ".join(f"{attr.__repr__()}" for attr in self.attributes) + "\n]" if self.attributes else "[]"
        )

    def __repr__(self) -> str:
        """Object representation of the Attributes collection."""
        return self.__str__()

    def set_client(self, client: Any) -> None:
        """Set the client for the Attributes object. Set automatically,
            if the Attributes object is instantiated from a Fusion object.

        Args:
            client (Any): Fusion client object.

        Examples:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> attributes = fusion.attributes()
            >>> attributes.set_client(fusion)

        """
        self._client = client

    def add_attribute(self, attribute: Attribute) -> None:
        """Add an Attribute instance to the collection.

        Args:
            attribute (Attribute): Attribute instance to add.

        Examples:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> attribute = fusion.attribute(identifier="my_attribute", index=0)
            >>> attributes = fusion.attributes()
            >>> attributes.add_attribute(attribute)

        """
        self.attributes.append(attribute)

    def remove_attribute(self, identifier: str) -> bool:
        """Remove an Attribute instance from the collection by identifier.

        Args:
            identifier (str): Identifier of the Attribute to remove.

        Returns:
            bool: True if the Attribute was removed, False otherwise.

        Examples:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> attribute = fusion.attribute(identifier="my_attribute", index=0)
            >>> attributes = fusion.attributes(attributes=[attribute])
            >>> attributes.remove_attribute("my_attribute")

        """
        for attr in self.attributes:
            if attr.identifier == identifier:
                self.attributes.remove(attr)
                return True
        return False

    def get_attribute(self, identifier: str) -> Attribute | None:
        """Get an Attribute instance from the collection by identifier.

        Args:
            identifier (str): Identifier of the Attribute to retrieve.

        Returns:
            Attribute | None: The Attribute instance if found, None otherwise.

        Examples:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> attribute = fusion.attribute(identifier="my_attribute", index=0)
            >>> attributes =fusion.attributes(attributes=[attribute])
            >>> retrieved_attribute = attributes.get_attribute("my_attribute")

        """
        for attr in self.attributes:
            if attr.identifier == identifier:
                return attr
        return None

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Convert the collection of Attribute instances to a list of dictionaries.

        Returns:
            dict[str, list[dict[str, Any]]]: Collection of Attribute instances as a dictionary.

        Examples:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> attribute = fusion.attribute(identifier="my_attribute", index=0)
            >>> attributes = fusion.attributes(attributes=[attribute])
            >>> attributes_dict = attributes.to_dict()

        """
        dict_out = {"attributes": [attr.to_dict() for attr in self.attributes]}
        return dict_out

    def from_dict_list(self, data: list[dict[str, Any]]) -> Attributes:
        """Create an Attributes instance from a list of dictionaries.

        Args:
            data (list[dict[str, Any]]): List of dictionaries representing Attribute instances.

        Returns:
            Attributes: Attributes instance.

        Examples:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> data = [
            ...     {
            ...         "identifier": "my_attribute",
            ...         "index": 0,
            ...         "dataType": "String",
            ...         "title": "My Attribute",
            ...         "description": "My attribute description"
            ...     }
            ... ]
            >>> attributes = fusion.attributes().from_dict_list(data)

        """
        attributes = [Attribute.from_dict(attr_data) for attr_data in data]
        attrs_obj = Attributes(attributes=attributes)

        attrs_obj.set_client(self._client)
        return attrs_obj

    def from_dataframe(self, data: pd.DataFrame) -> Attributes:
        """Create an Attributes instance from a pandas DataFrame.

        Args:
            data (pd.DataFrame): DataFrame representing Attribute instances.

        Returns:
            Attributes: Attributes instance.

        Examples:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> import pandas as pd
            >>> data = pd.DataFrame([
            ...     {
            ...         "identifier": "my_attribute",
            ...         "index": 0,
            ...         "dataType": "String",
            ...         "title": "My Attribute",
            ...         "description": "My attribute description"
            ...     }
            ... ])
            >>> attributes = fusion.attributes().from_dataframe(data)

        """
        data = data.replace(to_replace=np.nan, value=None)
        data = data.reset_index() if "index" not in data.columns else data
        attributes = [Attribute.from_series(series) for _, series in data.iterrows()]
        attrs_obj = Attributes(attributes=attributes)

        attrs_obj.set_client(self._client)
        return attrs_obj

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the collection of Attribute instances to a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame representing the collection of Attribute instances.

        Examples:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> import pandas as pd
            >>> attribute = fusion.attribute(identifier="my_attribute", index=0)
            >>> attributes = fusion.attributes(attributes=[attribute])
            >>> attributes_df = attributes.to_dataframe()

        """
        if len(self.attributes) == 0:
            self.attributes = [Attribute(identifier="example_attribute", index=0)]
        data = [attr.to_dict() for attr in self.attributes]
        return pd.DataFrame(data)

    def from_catalog(self, dataset: str, catalog: str | None = None, client: Fusion | None = None) -> Attributes:
        """Instatiate an Attributes object from a dataset's attributes in a Fusion catalog.

        Args:
            dataset (str): The dataset identifier.
            catalog (str | None, optional): The catalog identifier. Defaults to None.
            client (Fusion | None, optional): Fusion session. Defaults to None.
                If instantiated from a Fusion object, then the client is set automatically.

        Returns:
            Attributes: An instance of the Attributes class with the attributes from the catalog.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> attributes = fusion.attributes().from_catalog(dataset="my_dataset", catalog="my_catalog")

        """
        client = self._client if client is None else client
        if client is None:
            raise ValueError("Client must be provided")
        catalog = client._use_catalog(catalog)
        url = f"{client.root_url}/catalogs/{catalog}/datasets/{dataset}/attributes"
        response = client.session.get(url)
        response.raise_for_status()
        list_attributes = response.json()["resources"]
        list_attributes = sorted(list_attributes, key=lambda x: x["index"])

        self.attributes = [Attribute.from_dict(attr_data) for attr_data in list_attributes]
        return self

    def create(
        self,
        dataset: str,
        catalog: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Upload the Attributes  to a dataset in a Fusion catalog.

        Args:
            dataset (str): Dataset identifier.
            client (Fusion, optional): A Fusion client object. Defaults to the instance's _client.
                If instantiated from a Fusion object, then the client is set automatically.
            catalog (str, optional): A catalog identifier. Defaults to None.
            return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

        Returns:
            requests.Response | None: The response object from the API call if return_resp_obj is True, otherwise None.

        Examples:

            From scratch:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> attribute = fusion.attribute(identifier="my_attribute", index=0)
            >>> attributes = fusion.attributes(attributes=[attribute])
            >>> attributes.create(dataset="my_dataset", catalog="my_catalog")

            From a list of dictionaries:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> data = [
            ...     {
            ...         "identifier": "my_attribute",
            ...         "index": 0,
            ...         "dataType": "String",
            ...         "title": "My Attribute",
            ...         "description": "My attribute description"
            ...     }
            ... ]
            >>> attributes = fusion.attributes().from_dict_list(data)
            >>> attributes.create(dataset="my_dataset", catalog="my_catalog")

            From a pandas DataFrame:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> import pandas as pd
            >>> data = pd.DataFrame([
            ...     {
            ...         "identifier": "my_attribute",
            ...         "index": 0,
            ...         "dataType": "String",
            ...         "title": "My Attribute",
            ...         "description": "My attribute description"
            ...     }
            ... ])
            >>> attributes = fusion.attributes().from_dataframe(data)
            >>> attributes.create(dataset="my_dataset", catalog="my_catalog")

            From existing dataset's attributes in a Fusion catalog:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> attributes = fusion.attributes().from_catalog(dataset="my_dataset", catalog="my_catalog")
            >>> attributes.create(dataset="my_new_dataset", catalog="my_catalog")

        """
        client = self._client if client is None else client
        if client is None:
            raise ValueError("Client must be provided")
        catalog = client._use_catalog(catalog)
        data = self.to_dict()
        url = f"{client.root_url}/catalogs/{catalog}/datasets/{dataset}/attributes"
        resp = client.session.put(url, json=data)
        resp.raise_for_status()
        return resp if return_resp_obj else None

    def delete(
        self,
        dataset: str,
        catalog: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> list[requests.Response] | None:
        """Delete the Attributes from a Fusion catalog.

        Args:
            dataset (str): Dataset identifier.
            client (Fusion, optional): A Fusion client object. Defaults to the instance's _client.
            catalog (str, optional): A catalog identifier. Defaults to None.
            return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

        Returns:
            list[requests.Response] | None: List of response objects from the API calls if return_resp_obj is True,
             otherwise None.

        Examples:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> attributes = fusion.attributes().from_catalog(dataset="my_dataset", catalog="my_catalog")
            >>> attributes.delete(dataset="my_dataset", catalog="my_catalog")

        """
        client = self._client if client is None else client
        if client is None:
            raise ValueError("Client must be provided")
        catalog = client._use_catalog(catalog)

        resp = [
            client.session.delete(
                f"{client.root_url}/catalogs{catalog}/datasets/{dataset}/attributes/{attr.identifier}"
            )
            for attr in self.attributes
        ]
        return resp if return_resp_obj else None