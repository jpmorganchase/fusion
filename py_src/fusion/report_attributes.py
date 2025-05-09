from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from fusion.utils import (
    CamelCaseMeta,
    camel_to_snake,
    requests_raise_for_status,
)

if TYPE_CHECKING:
    import requests

    from fusion import Fusion


@dataclass
class ReportAttribute(metaclass=CamelCaseMeta):
    """Fusion ReportAttribute class for managing attribute metadata in a Fusion catalog.

    Attributes:
        name (str): The unique name of the attribute. Mandatory.
        title (str): The display title of the attribute. Mandatory.
        description (str, optional): A description of the attribute. Defaults to None.
        technicalDataType (str, optional): The technical data type of the attribute. Defaults to None.
        path (str, optional): The hierarchical path for the attribute. Defaults to None.
        dataPublisher (str, optional): The publisher of the data. Defaults to None.
    """

    name: str
    title: str
    description: str | None = None
    technicalDataType: str | None = None
    path: str | None = None
    dataPublisher: str | None = None

    _client: Fusion | None = field(init=False, repr=False, compare=False, default=None)

    def __str__(self) -> str:
        """String representation of ReportAttribute."""
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"ReportAttribute(\n" + ",\n ".join(f"{k}={v!r}" for k, v in attrs.items()) + "\n)"

    def __repr__(self) -> str:
        """Object representation."""
        return self.__str__()

    def __getattr__(self, name: str) -> Any:
        """Redirect attribute access to the snake_case version."""
        snake_name = camel_to_snake(name)
        if snake_name in self.__dict__:
            return self.__dict__[snake_name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute with camelCase/snake_case support."""
        if name == "client":
            object.__setattr__(self, name, value)
        else:
            snake_name = camel_to_snake(name)
            self.__dict__[snake_name] = value

    @property
    def client(self) -> Fusion | None:
        """Return the Fusion client."""
        return self._client

    @client.setter
    def client(self, client: Fusion | None) -> None:
        """Set the Fusion client."""
        self._client = client

    def to_dict(self) -> dict[str, Any]:
        """Convert the object to a dictionary for API calls."""
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "technicalDataType": self.technicalDataType,
            "path": self.path,
            "dataPublisher": self.dataPublisher,
        }

    def create(
        self,
        dataset: str,
        catalog: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Upload a new ReportAttribute to a Fusion catalog."""
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)
        data = self.to_dict()
        url = f"{client.root_url}catalogs/{catalog}/datasets/{dataset}/attributes/{self.name}"
        resp = client.session.put(url, json=data)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def delete(
        self,
        dataset: str,
        catalog: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Delete a ReportAttribute from a Fusion catalog."""
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)
        url = f"{client.root_url}catalogs/{catalog}/datasets/{dataset}/attributes/{self.name}"
        resp = client.session.delete(url)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def _use_client(self, client: Fusion | None) -> Fusion:
        """Get the Fusion client (internal helper)."""
        res = self._client if client is None else client
        if res is None:
            raise ValueError("A Fusion client object is required.")
        return res


@dataclass
class ReportAttributes:
    """Class representing a collection of ReportAttribute instances for managing attribute metadata."""

    attributes: list[ReportAttribute] = field(default_factory=list)
    _client: Fusion | None = None

    def __str__(self) -> str:
        """String representation of the ReportAttributes collection."""
        return (
            f"[\n" + ",\n ".join(f"{attr.__repr__()}" for attr in self.attributes) + "\n]"
            if self.attributes else "[]"
        )

    def __repr__(self) -> str:
        """Object representation of the ReportAttributes collection."""
        return self.__str__()

    @property
    def client(self) -> Fusion | None:
        """Return the Fusion client."""
        return self._client

    @client.setter
    def client(self, client: Fusion | None) -> None:
        """Set the Fusion client."""
        self._client = client

    def _use_client(self, client: Fusion | None) -> Fusion:
        """Determine client."""
        res = self._client if client is None else client
        if res is None:
            raise ValueError("A Fusion client object is required.")
        return res

    def add_attribute(self, attribute: ReportAttribute) -> None:
        """Add a ReportAttribute instance to the collection."""
        self.attributes.append(attribute)

    def remove_attribute(self, name: str) -> bool:
        """Remove a ReportAttribute instance from the collection by name."""
        for attr in self.attributes:
            if attr.name == name:
                self.attributes.remove(attr)
                return True
        return False

    def get_attribute(self, name: str) -> ReportAttribute | None:
        """Get a ReportAttribute instance from the collection by name."""
        for attr in self.attributes:
            if attr.name == name:
                return attr
        return None

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Convert the collection of ReportAttribute instances to a list of dictionaries."""
        return {"attributes": [attr.to_dict() for attr in self.attributes]}

    @classmethod
    def _from_dict_list(cls, data: list[dict[str, Any]]) -> ReportAttributes:
        """Create a ReportAttributes instance from a list of dictionaries."""
        attributes = [ReportAttribute(**attr_data) for attr_data in data]
        return ReportAttributes(attributes=attributes)

    @classmethod
    def _from_dataframe(cls, data: pd.DataFrame) -> ReportAttributes:
        """Create a ReportAttributes instance from a pandas DataFrame."""
        data = data.where(data.notna(), None)
        attributes = [
            ReportAttribute(**series.dropna().to_dict())
            for _, series in data.iterrows()
        ]
        return ReportAttributes(attributes=attributes)

    def from_object(
        self,
        attributes_source: list[ReportAttribute]
        | list[dict[str, Any]]
        | pd.DataFrame,
    ) -> ReportAttributes:
        """Instantiate a ReportAttributes object from a list of objects, dicts, or a DataFrame."""
        if isinstance(attributes_source, list):
            if all(isinstance(attr, ReportAttribute) for attr in attributes_source):
                attributes = ReportAttributes(attributes=attributes_source)
            elif all(isinstance(attr, dict) for attr in attributes_source):
                attributes = ReportAttributes._from_dict_list(attributes_source)
            else:
                raise TypeError("List must contain either ReportAttribute instances or dicts.")
        elif isinstance(attributes_source, pd.DataFrame):
            attributes = ReportAttributes._from_dataframe(attributes_source)
        else:
            raise TypeError("Unsupported type for attributes_source.")
        attributes.client = self._client
        return attributes

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the collection of ReportAttribute instances to a pandas DataFrame."""
        data = [attr.to_dict() for attr in self.attributes]
        return pd.DataFrame(data)

    def from_catalog(
        self,
        dataset: str,
        catalog: str | None = None,
        client: Fusion | None = None,
    ) -> ReportAttributes:
        """Instantiate a ReportAttributes object from a dataset's attributes in a Fusion catalog."""
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)
        url = f"{client.root_url}catalogs/{catalog}/datasets/{dataset}/attributes"
        response = client.session.get(url)
        requests_raise_for_status(response)
        list_attributes = response.json().get("resources", [])
        self.attributes = [ReportAttribute(**attr_data) for attr_data in list_attributes]
        return self

    def create(
        self,
        dataset: str,
        catalog: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Upload the ReportAttributes to a dataset in a Fusion catalog."""
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)
        data = self.to_dict()
        url = f"{client.root_url}catalogs/{catalog}/datasets/{dataset}/attributes"
        resp = client.session.put(url, json=data)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def delete(
        self,
        dataset: str,
        catalog: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> list[requests.Response] | None:
        """Delete the ReportAttributes from a Fusion catalog."""
        responses = []
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)
        for attr in self.attributes:
            resp = client.session.delete(
                f"{client.root_url}catalogs/{catalog}/datasets/{dataset}/attributes/{attr.name}"
            )
            requests_raise_for_status(resp)
            responses.append(resp)
        return responses if return_resp_obj else None
