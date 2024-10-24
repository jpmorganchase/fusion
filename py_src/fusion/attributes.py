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
    """Attribute class."""

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
        """Set the client for the Product."""
        self._client = client

    def __str__(self: Attribute) -> str:
        """Format object representation."""
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"Attribute(\n" + ",\n ".join(f"{k}={v!r}" for k, v in attrs.items()) + "\n)"

    def __repr__(self: Attribute) -> str:
        """Format object representation."""
        s = ", ".join(f"{getattr(self, f.name)!r}" for f in fields(self) if not f.name.startswith("_"))
        return "(" + s + ")"

    def __post_init__(self: Attribute) -> None:
        """Post-initialization steps."""
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
        """Create an Attribute object from a pandas Series."""
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
        """Create an Attribute object from a dictionary."""
        keys = [f.name for f in fields(cls)]
        data = {k: (None if pd.isna(v) else v) for k, v in data.items() if k in keys}
        if "dataType" in data:
            data["dataType"] = Types[data["dataType"].strip().rsplit(".", maxsplit=1)[-1].title()]
        return cls(**data)

    def to_dict(self: Attribute) -> dict[str, Any]:
        """Convert object to dictionary."""
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
        """Create an Attribute in a Fusion catalog.

        Args:
            dataset (str): Dataset identifier.
            client (Fusion, optional): A Fusion client object. Defaults to the instance's _client.
            catalog (str, optional): A catalog identifier. Defaults to None.
            return_resp_obj (bool, optional): _description_. Defaults to False.

        Returns:
            requests.Response | None: The response object from the API call if return_resp_obj is True, otherwise None.
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
    """Class representing a collection of Attribute instances."""

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
        """Set the client for the Product."""
        self._client = client

    def add_attribute(self, attribute: Attribute) -> None:
        """Add an Attribute instance to the collection."""
        self.attributes.append(attribute)

    def remove_attribute(self, identifier: str) -> bool:
        """Remove an Attribute instance from the collection by identifier."""
        for attr in self.attributes:
            if attr.identifier == identifier:
                self.attributes.remove(attr)
                return True
        return False

    def get_attribute(self, identifier: str) -> Attribute | None:
        """Get an Attribute instance from the collection by identifier."""
        for attr in self.attributes:
            if attr.identifier == identifier:
                return attr
        return None

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Convert the collection of Attribute instances to a list of dictionaries."""
        dict_out = {"attributes": [attr.to_dict() for attr in self.attributes]}
        return dict_out

    def from_dict_list(self, data: list[dict[str, Any]]) -> Attributes:
        """Create an Attributes instance from a list of dictionaries."""
        attributes = [Attribute.from_dict(attr_data) for attr_data in data]
        attrs_obj = Attributes(attributes=attributes)

        attrs_obj.set_client(self._client)
        return attrs_obj

    def from_dataframe(self, data: pd.DataFrame) -> Attributes:
        """Create an Attributes instance from a pandas DataFrame."""
        data = data.replace(to_replace=np.nan, value=None)
        data = data.reset_index() if "index" not in data.columns else data
        attributes = [Attribute.from_series(series) for _, series in data.iterrows()]
        attrs_obj = Attributes(attributes=attributes)

        attrs_obj.set_client(self._client)
        return attrs_obj

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the collection of Attribute instances to a pandas DataFrame."""
        if len(self.attributes) == 0:
            self.attributes = [Attribute(identifier="example_attribute", index=0)]
        data = [attr.to_dict() for attr in self.attributes]
        return pd.DataFrame(data)

    def from_catalog(self, dataset: str, catalog: str | None = None, client: Fusion | None = None) -> Attributes:
        """Get the Attributes from a Fusion catalog."""
        client = self._client if client is None else client
        if client is None:
            raise ValueError("Client must be provided")
        catalog = client._use_catalog(catalog)
        url = f"{client.root_url}catalogs/{catalog}/datasets/{dataset}/attributes"
        response = client.session.get(url)
        response.raise_for_status()
        list_attributes = response.json()["resources"]
        list_attributes = sorted(list_attributes, key=lambda x: x["index"])

        self.attributes = [Attribute.from_dict(attr_data) for attr_data in list_attributes]

        attributes_obj = Attributes(attributes=self.attributes)

        return attributes_obj

    def create(
        self,
        dataset: str,
        catalog: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Create the Attributes in a Fusion catalog.

        Args:
            dataset (str): Dataset identifier.
            client (Fusion, optional): A Fusion client object. Defaults to the instance's _client.
            catalog (str, optional): A catalog identifier. Defaults to None.
            return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

        Returns:
            requests.Response | None: The response object from the API call if return_resp_obj is True, otherwise None.
        """
        client = self._client if client is None else client
        if client is None:
            raise ValueError("Client must be provided")
        catalog = client._use_catalog(catalog)
        data = self.to_dict()
        url = f"{client.root_url}catalogs/{catalog}/datasets/{dataset}/attributes"
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
        """
        client = self._client if client is None else client
        if client is None:
            raise ValueError("Client must be provided")
        catalog = client._use_catalog(catalog)

        resp = [
            client.session.delete(
                f"{client.root_url}catalogs/{catalog}/datasets/{dataset}/attributes/{attr.identifier}"
            )
            for attr in self.attributes
        ]
        return resp if return_resp_obj else None
