from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union

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
    name: str
    title: str
    description: str | None = None
    technicalDataType: str | None = None
    path: str | None = None
    dataPublisher: str | None = None

    _client: Fusion | None = field(init=False, repr=False, compare=False, default=None)

    def __str__(self) -> str:
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"ReportAttribute(\n" + ",\n ".join(f"{k}={v!r}" for k, v in attrs.items()) + "\n)"

    def __repr__(self) -> str:
        return self.__str__()

    def __getattr__(self, name: str) -> Any:
        snake_name = camel_to_snake(name)
        if snake_name in self.__dict__:
            return self.__dict__[snake_name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "client":
            object.__setattr__(self, name, value)
        else:
            snake_name = camel_to_snake(name)
            self.__dict__[snake_name] = value

    @property
    def client(self) -> Fusion | None:
        return self._client

    @client.setter
    def client(self, client: Fusion | None) -> None:
        self._client = client

    def to_dict(self) -> dict[str, Any]:
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
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)
        url = f"{client.root_url}catalogs/{catalog}/datasets/{dataset}/attributes/{self.name}"
        resp = client.session.delete(url)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def _use_client(self, client: Fusion | None) -> Fusion:
        res = self._client if client is None else client
        if res is None:
            raise ValueError("A Fusion client object is required.")
        return res


@dataclass
class ReportAttributes:
    attributes: list[ReportAttribute] = field(default_factory=list)
    _client: Fusion | None = None

    def __str__(self) -> str:
        return (
            f"[\n" + ",\n ".join(f"{attr.__repr__()}" for attr in self.attributes) + "\n]" if self.attributes else "[]"
        )

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def client(self) -> Fusion | None:
        return self._client

    @client.setter
    def client(self, client: Fusion | None) -> None:
        self._client = client

    def _use_client(self, client: Fusion | None) -> Fusion:
        res = self._client if client is None else client
        if res is None:
            raise ValueError("A Fusion client object is required.")
        return res

    def add_attribute(self, attribute: ReportAttribute) -> None:
        self.attributes.append(attribute)

    def remove_attribute(self, name: str) -> bool:
        for attr in self.attributes:
            if attr.name == name:
                self.attributes.remove(attr)
                return True
        return False

    def get_attribute(self, name: str) -> ReportAttribute | None:
        for attr in self.attributes:
            if attr.name == name:
                return attr
        return None

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {"attributes": [attr.to_dict() for attr in self.attributes]}

    @classmethod
    def _from_dict_list(cls, data: list[dict[str, Any]]) -> ReportAttributes:
        attributes = [ReportAttribute(**attr_data) for attr_data in data]
        return cls(attributes=attributes)

    @classmethod
    def _from_dataframe(cls, data: pd.DataFrame) -> ReportAttributes:
        data = data.where(data.notna(), None)
        attributes = [ReportAttribute(**series.dropna().to_dict()) for _, series in data.iterrows()]
        return cls(attributes=attributes)

    @classmethod
    def _from_csv(cls: type[ReportAttributes], file_path: str) -> ReportAttributes:
        data = pd.read_csv(file_path)
        return cls._from_dataframe(data)

    def from_object(
        self,
        attributes_source: Union[list[ReportAttribute], list[dict[str, Any]], pd.DataFrame],  # noqa
    ) -> ReportAttributes:
        if isinstance(attributes_source, list):
            if all(isinstance(attr, ReportAttribute) for attr in attributes_source):
                attributes_obj = ReportAttributes(attributes=attributes_source)  # ✅ safe
            elif all(isinstance(attr, dict) for attr in attributes_source):
                attributes_obj = ReportAttributes._from_dict_list(attributes_source)  # ✅ safe
            else:
                raise TypeError("List must contain either ReportAttribute instances or dicts.")
        elif isinstance(attributes_source, pd.DataFrame):
            attributes_obj = ReportAttributes._from_dataframe(attributes_source)
        else:
            raise TypeError("Unsupported type for attributes_source.")
        attributes_obj.client = self._client
        return attributes_obj

    def to_dataframe(self) -> pd.DataFrame:
        data = [attr.to_dict() for attr in self.attributes]
        return pd.DataFrame(data)

    def from_catalog(
        self,
        dataset: str,
        catalog: str | None = None,
        client: Fusion | None = None,
    ) -> ReportAttributes:
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)
        url = f"{client.root_url}catalogs/{catalog}/datasets/{dataset}/attributes"
        response = client.session.get(url)
        requests_raise_for_status(response)
        list_attributes = response.json().get("resources", [])
        self.attributes = [ReportAttribute(**attr_data) for attr_data in list_attributes]
        return self


    def register(
        self,
        report_id: str,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """
        Register the ReportAttributes to the metadata-lineage/report API.

        Args:
            report_id (str): The identifier of the report.
            client (Fusion, optional): Fusion client, for auth and config. Uses self._client if not passed.
            return_resp_obj (bool, optional): If True, returns the response object. Otherwise, returns None.

        Returns:
            requests.Response | None: API response object if return_resp_obj is True.
        """
        client = self._use_client(client)

        url = f"{client.root_url}metadata-lineage/report/{report_id}/attributes"  
        # again replace with f"https...reportEleemts whole url

        payload = [attr.to_dict() for attr in self.attributes]

        resp = client.session.post(url, json=payload)
        requests_raise_for_status(resp)

        return resp if return_resp_obj else None

    def create(
        self,
        dataset: str,
        catalog: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
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
