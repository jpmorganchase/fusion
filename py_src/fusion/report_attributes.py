from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union, cast

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
    title: str | None = None
    id: int | None = None
    source_identifier: str | None = None
    description: str | None = None
    technical_data_type: str | None = None
    path: str | None = None

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
            "id": self.id,
            "sourceIdentifier": self.source_identifier,
            "title": self.title,
            "description": self.description,
            "technicalDataType": self.technical_data_type,
            "path": self.path,
        }

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
            if attr.title == name:
                self.attributes.remove(attr)
                return True
        return False

    def get_attribute(self, name: str) -> ReportAttribute | None:
        for attr in self.attributes:
            if attr.title == name:
                return attr
        return None

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {"attributes": [attr.to_dict() for attr in self.attributes]}

    def from_dict_list(self, data: list[dict[str, Any]]) -> ReportAttributes:
        attributes = [ReportAttribute(**attr_data) for attr_data in data]
        result = ReportAttributes(attributes=attributes)
        result.client = self._client
        return result

    def from_dataframe(self, data: pd.DataFrame) -> ReportAttributes:
        data = data.where(data.notna(), None)
        attributes = [ReportAttribute(**series.dropna().to_dict()) for _, series in data.iterrows()]
        result = ReportAttributes(attributes=attributes)
        result.client = self._client
        return result

    def from_csv(self, file_path: str) -> ReportAttributes:
        """Load ReportAttributes from a CSV file with custom column mappings and ignore irrelevant columns."""
        df = pd.read_csv(file_path)  # noqa

        # Only keep relevant columns
        column_map = {
            "Local Data Element Reference ID": "source_identifier",
            "Data Element Name": "title",
            "Data Element Description": "description",
        }

        # Filter to only needed columns (drop all others)
        df = df[[col for col in column_map if col in df.columns]]  # noqa

        # Rename to match ReportAttribute fields
        df = df.rename(columns=column_map)  # noqa

        # Add any missing required fields with default None
        for col in ["technical_data_type", "path"]:
            if col not in df:
                df[col] = None

        # Replace NaN/missing values with None
        df = df.where(pd.notna(df), None)  # noqa

        return self.from_dataframe(df)

    def from_object(
        self,
        attributes_source: Union[  # noqa
            list[ReportAttribute], list[dict[str, Any]], pd.DataFrame, str
        ],
    ) -> ReportAttributes:
        """
        Load ReportAttributes from various sources: list, DataFrame, .csv path, or JSON string.
        """
        import json

        if isinstance(attributes_source, list):
            if all(isinstance(attr, ReportAttribute) for attr in attributes_source):
                attributes_obj = ReportAttributes(attributes=cast(list[ReportAttribute], attributes_source))
            elif all(isinstance(attr, dict) for attr in attributes_source):
                attributes_obj = self.from_dict_list(cast(list[dict[str, Any]], attributes_source))
            else:
                raise TypeError("List must contain either ReportAttribute instances or dicts.")
        elif isinstance(attributes_source, pd.DataFrame):
            attributes_obj = self.from_dataframe(attributes_source)
        elif isinstance(attributes_source, str):
            if attributes_source.strip().endswith(".csv"):
                attributes_obj = self.from_csv(attributes_source)
            elif attributes_source.strip().startswith("[{"):
                dict_list = json.loads(attributes_source)
                attributes_obj = self.from_dict_list(dict_list)
            else:
                raise ValueError("String must be a .csv path or JSON array string.")
        else:
            raise TypeError("Unsupported type for attributes_source.")

        attributes_obj.client = self._client
        return attributes_obj

    def to_dataframe(self) -> pd.DataFrame:
        data = [attr.to_dict() for attr in self.attributes]
        return pd.DataFrame(data)

    def _build_api_url(self, client: Fusion, report_id: str) -> str:
        """This is a private method, use it to build the API URL for report attributes operations."""
        return f"{client._get_new_root_url()}/api/corelineage-service/v1/reports/{report_id}/attributes"

    def create(
        self,
        report_id: str,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Create the ReportAttributes.

        Args:
            report_id (str): The identifier of the report.
            client (Fusion, optional): Fusion client, for auth and config. Uses self._client incase not passed.
            return_resp_obj (bool, optional): If True, returns the response object. Otherwise, returns None.

        Returns:
            requests.Response | None: API response object if return_resp_obj is True.

        Example:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> attr = fusion.report_attribute(
            ...     title="Revenue",
            ...     source_identifier="rev_001",
            ...     description="Revenue field for reporting",
            ...     technical_data_type="decimal",
            ...     path="/data/revenue"
            ... )
            >>> report_attrs = fusion.report_attributes([attr])
            >>> report_attrs.create(report_id="report_1")
        """
        client = self._use_client(client)
        url = self._build_api_url(client, report_id)
        payload = []
        for attr in self.attributes:
            attr_dict = {
                field_name: field_value
                for field_name, field_value in attr.to_dict().items()
                if field_value is not None and field_name != "id"
            }
            payload.append(attr_dict)

        resp = client.session.post(url, json=payload)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def update(
        self,
        report_id: str,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Replace report attributes.

        This method performs a complete replacement of each report attribute.
        Any properties not specified will be assigned null or default values.
        Note: title is immutable and cannot be modified.

        Args:
            report_id (str): The identifier of the report.
            client (Fusion, optional): Fusion client for auth and config. Uses self._client if not passed.
            return_resp_obj (bool, optional): If True, returns the response object. Otherwise, returns None.

        Returns:
            requests.Response | None: API response object if return_resp_obj is True.

        Raises:
            ValueError: If required fields are missing or invalid.

        Example:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> attr = fusion.report_attribute(
            ...     id=456,
            ...     source_identifier="rev_001",
            ...     description="Updated revenue field",
            ...     technical_data_type="decimal"
            ... )
            >>> report_attrs = fusion.report_attributes([attr])
            >>> report_attrs.update(report_id="report_1")

        """
        client = self._use_client(client)
        for attr in self.attributes:
            if attr.id is None:
                raise ValueError(f"ReportAttribute must have an 'id' field for update")

        url = self._build_api_url(client, report_id)
        payload = []
        for attr in self.attributes:
            attr_dict = attr.to_dict()
            attr_dict.pop("title", None)
            payload.append(attr_dict)

        resp = client.session.put(url, json=payload)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def update_fields(
        self,
        report_id: str,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Update specific fields of report attributes.

        This method performs a partial update of each report attribute.
        Only the specified properties will be updated while all other properties remain unchanged.
        Note: title is immutable and cannot be modified. At least one property must be provided.

        Args:
            report_id (str): The identifier of the report.
            client (Fusion, optional): Fusion client for auth and config. Uses self._client if not passed.
            return_resp_obj (bool, optional): If True, returns the response object. Otherwise, returns None.

        Returns:
            requests.Response | None: API response object if return_resp_obj is True.

        Raises:
            ValueError: If required fields are missing or invalid, or if no updatable fields are provided.

        Example:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> attr = fusion.report_attribute(
            ...     id=456,
            ...     description="Updated revenue field"  # only this will be updated
            ... )
            >>> report_attrs = fusion.report_attributes([attr])
            >>> report_attrs.update_fields(report_id="report_1")

        """
        client = self._use_client(client)
        for attr in self.attributes:
            if attr.id is None:
                raise ValueError(f"ReportAttribute must have an 'id' field for update_fields")

        url = self._build_api_url(client, report_id)
        payload = []
        for attr in self.attributes:
            attr_dict = {
                field_name: field_value
                for field_name, field_value in attr.to_dict().items()
                if field_value is not None and field_name != "title"
            }
            payload.append(attr_dict)

        resp = client.session.patch(url, json=payload)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def delete(
        self,
        report_id: str,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Soft delete report attributes.

        This method performs a soft delete of each report attribute.
        Once soft deleted, the report attributes can still be viewed but cannot be modified.
        Note: Throws an error if report attribute is already deleted.

        Args:
            report_id (str): The identifier of the report.
            client (Fusion, optional): Fusion client for auth and config. Uses self._client if not passed.
            return_resp_obj (bool, optional): If True, returns the response object. Otherwise, returns None.

        Returns:
            requests.Response | None: API response object if return_resp_obj is True.

        Raises:
            ValueError: If required fields are missing or if attributes are already deleted.

        Example:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> attr = fusion.report_attribute(
            ...     id=456  # only id is needed for deletion
            ... )
            >>> report_attrs = fusion.report_attributes([attr])
            >>> report_attrs.delete(report_id="report_1")

        """
        client = self._use_client(client)
        for attr in self.attributes:
            if attr.id is None:
                raise ValueError(f"ReportAttribute must have an 'id' field for deletion")

        url = self._build_api_url(client, report_id)
        payload = [{"id": attr.id} for attr in self.attributes]
        resp = client.session.delete(url, json=payload)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None
