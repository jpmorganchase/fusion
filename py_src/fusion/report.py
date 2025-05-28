"""Fusion Report class and functions."""

from __future__ import annotations

import json as js
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import pandas as pd

from .utils import (
    CamelCaseMeta,
    _is_json,
    camel_to_snake,
    make_bool,
    requests_raise_for_status,
    snake_to_camel,
    tidy_string,
)

if TYPE_CHECKING:
    import requests

    from fusion import Fusion


@dataclass
class Report(metaclass=CamelCaseMeta):
    """Fusion Report class for managing report metadata.

    Attributes:
        name (str): A unique name for the report.
        tier_type (str): The tier classification of the report.
        lob (str): The line of business associated with the report.
        data_node_id (dict[str, str]): Identifier of the associated data node.
        alternative_id (dict[str, str]): Alternate identifiers for the report.

        title (str, optional): A title for the report. Defaults to None.
        alternate_id (str, optional): A alternate identifier for the report. Defaults to None.
        description (str, optional): A description of the report. Defaults to None.
        frequency (str, optional): The frequency with which the report is generated. Defaults to None.
        category (str, optional): The primary category of the report. Defaults to None.
        sub_category (str, optional): A more specific classification under the main category. Defaults to None.
        report_inventory_name (str, optional): Name of the report in the report inventory. Defaults to None.
        report_inventory_id (str, optional): Identifier of the report in the inventory. Defaults to None.
        report_owner (str, optional): Owner responsible for the report. Defaults to None.
        sub_lob (str, optional): Subdivision of the line of business. Defaults to None.
        is_bcbs239_program (bool, optional): Indicates if the report is part of the BCBS 239 program. Defaults to None.
        risk_area (str, optional): The area of risk the report addresses. Defaults to None.
        risk_stripe (str, optional): A specific risk category or stripe. Defaults to None.
        sap_code (str, optional): Associated SAP code for financial tracking. Defaults to None.
        sourced_object (str, optional): The original source object for the report. Defaults to None.
        domain (dict[str, str | bool], optional): Domain information related to the report. Defaults to None.
        data_model_id (dict[str, str], optional): Identifier of the data model used. Defaults to None.
        _client (Any, optional): A Fusion client object. Defaults to None.
    """

    name: str
    tier_type: str
    lob: str
    data_node_id: dict[str, str]
    alternative_id: dict[str, str]

    # Optional fields
    title: str | None = None
    alternate_id: str | None = None
    description: str | None = None
    frequency: str | None = None
    category: str | None = None
    sub_category: str | None = None
    report_inventory_name: str | None = None
    report_inventory_id: str | None = None
    report_owner: str | None = None
    sub_lob: str | None = None
    is_bcbs239_program: bool | None = None
    risk_area: str | None = None
    risk_stripe: str | None = None
    sap_code: str | None = None
    sourced_object: str | None = None
    domain: dict[str, str | bool] | None = None
    data_model_id: dict[str, str] | None = None

    _client: Fusion | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        self.name = tidy_string(self.name)
        self.title = tidy_string(self.title) if self.title else None
        self.description = tidy_string(self.description) if self.description else None

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

    def _use_client(self, client: Fusion | None) -> Fusion:
        res = self._client if client is None else client
        if res is None:
            raise ValueError("A Fusion client object is required.")
        return res

    @classmethod
    def _from_series(cls: type[Report], series:  pd.Series[Any]) -> Report:
        """Instantiate a Report object from a pandas Series."""
        series = series.rename(lambda x: x.replace(" ", "").replace("_", "").lower())

        # Normalize booleans
        is_bcbs239_program = series.get("isbcbs239program", None)
        is_bcbs239_program = make_bool(is_bcbs239_program) if is_bcbs239_program is not None else False

        return cls(
            name=series.get("name", ""),
            tier_type=series.get("tiertype", ""),
            lob=series.get("lob", ""),
            data_node_id=series.get("datanodeid", {"id": "", "name": "", "dataNodeType": ""}),
            alternative_id=series.get("alternativeid", {"domain": "", "value": ""}),
            title=series.get("title", None),
            alternate_id=series.get("alternateid", None),
            description=series.get("description", None),
            frequency=series.get("frequency", None),
            category=series.get("category", None),
            sub_category=series.get("subcategory", None),
            report_inventory_name=series.get("reportinventoryname", None),
            report_inventory_id=series.get("reportinventoryid", None),
            report_owner=series.get("reportowner", None),
            sub_lob=series.get("sublob", None),
            is_bcbs239_program=is_bcbs239_program,
            risk_area=series.get("riskarea", None),
            risk_stripe=series.get("riskstripe", None),
            sap_code=series.get("sapcode", None),
            sourced_object=series.get("sourcedobject", None),
            domain=series.get("domain", None),
            data_model_id=series.get("datamodelid", None),
        )

    @classmethod
    def _from_dict(cls: type[Report], data: dict[str, Any]) -> Report:
        """Instantiate a Report object from a dictionary.

        Args:
            data (dict[str, Any]): Report metadata as a dictionary.

        Returns:
            Report: Report object.
        """
        # Override camelCase keys to snake_case that are not automatically converted
        field_name_overrides = {"isBCBSProgram": "is_bcbs239_program"}

        def convert_keys(d: dict[str, Any]) -> dict[str, Any]:
            """Recursively convert camelCase keys in nested dicts to snake_case."""
            result = {}
            for key, value in d.items():
                new_key = camel_to_snake(key)
                if isinstance(value, dict):
                    result[new_key] = convert_keys(value)
                else:
                    result[new_key] = value
            return result

        # Convert all keys and nested dicts
        converted_data = {}
        for k, v in data.items():
            snake_key = field_name_overrides.get(k, camel_to_snake(k))
            if isinstance(v, dict) and not isinstance(v, str):
                converted_data[snake_key] = convert_keys(v)
            else:
                converted_data[snake_key] = v

        # Convert specific fields
        if "is_bcbs239_program" in converted_data:
            converted_data["is_bcbs239_program"] = make_bool(converted_data["is_bcbs239_program"])

        # Only use fields defined in the class
        keys = [f.name for f in fields(cls)]
        filtered_data = {k: v for k, v in converted_data.items() if k in keys}

        return cls(**filtered_data)

    @classmethod
    def _from_csv(cls: type[Report], file_path: str, name: str | None = None) -> Report:
        """Instantiate a Report object from a CSV file.

        Args:
            file_path (str): Path to the CSV file.
            name (str | None, optional): Report name for filtering if multiple reports are defined in the CSV.

        Returns:
            Report: Report object.
        """
        data = pd.read_csv(file_path)
        return (
            cls._from_series(data[data["name"] == name].reset_index(drop=True).iloc[0])
            if name
            else cls._from_series(data.reset_index(drop=True).iloc[0])
        )

    def from_object(
        self,
        report_source: Report | dict[str, Any] | str | pd.Series[Any],
    ) -> Report:
        """Instantiate a Report object from various sources.

        Args:
            report_source (Report | dict[str, Any] | str | pd.Series[Any]): Report metadata source.

        Raises:
            TypeError: If the source is unsupported.

        Returns:
            Report: Report object.
        """
        if isinstance(report_source, Report):
            report = report_source
        elif isinstance(report_source, dict):
            report = self._from_dict(report_source)
        elif isinstance(report_source, str):
            if _is_json(report_source):
                report = self._from_dict(js.loads(report_source))
            else:
                report = self._from_csv(report_source)
        elif isinstance(report_source, pd.Series):
            report = self._from_series(report_source)
        else:
            raise TypeError(f"Could not resolve the object provided: {report_source}")

        report.client = self._client
        return report

    def to_dict(self) -> dict[str, Any]:
        """Convert the Report instance to a dictionary.

        Returns:
            dict[str, Any]: Report metadata as a dictionary.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> report = fusion.report("report")
            >>> report_dict = report.to_dict()

        """

        report_dict = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if k == "is_bcbs239_program":
                report_dict["isBCBSProgram"] = v
            else:
                report_dict[snake_to_camel(k)] = v
        return report_dict

    def create(
        self,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Upload a new report to a Fusion catalog.

        Args:
            client (Fusion, optional): A Fusion client object. Defaults to the instance's _client.
                If instantiated from a Fusion object, then the client is set automatically.
            return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

        Returns:
            requests.Response | None: The response object from the API call if return_resp_obj is True, otherwise None.

        Examples:

            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> report = fusion.report(
            ...     name="report_1",
            ...     title="Quarterly Risk Report",
            ...     category="Risk",
            ...     frequency="Quarterly",
            ... )
            >>> report.create()

        """
        client = self._use_client(client)

        data = self.to_dict()

        url = f"{client.get_new_root_url()}/api/corelineage-service/v1/reports"
        resp: requests.Response = client.session.post(url, json=data)
        requests_raise_for_status(resp)

        return resp if return_resp_obj else None
