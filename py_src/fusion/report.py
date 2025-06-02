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
    def from_dict(cls: type[Report], data: dict[str, Any]) -> Report:
        """Instantiate a Report object from a partial API-style dictionary.
        
        Only sets fields that are present in the input.
        Missing fields are completely omitted from the instance.
        Accessing them later will raise AttributeError.
        """

        def normalize_value(val: Any) -> Any:
            if isinstance(val, str) and val.strip() == "":
                return None
            return val

        def convert_keys(d: dict[str, Any]) -> dict[str, Any]:
            converted = {}
            for k, v in d.items():
                key = k if k == "isBCBS239Program" else camel_to_snake(k)
                if isinstance(v, dict) and not isinstance(v, str):
                    converted[key] = convert_keys(v)
                else:
                    converted[key] = normalize_value(v)
            return converted

        converted_data = convert_keys(data)

                # convert to field name used in class
        if "isBCBS239Program" in converted_data:
            converted_data["is_bcbs239_program"] = make_bool(converted_data.pop("isBCBS239Program"))


        # Filter keys that are valid fields in the class
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in converted_data.items() if k in valid_fields}

        # Create instance without calling __init__
        report = cls.__new__(cls)

        # Only set attributes that were in the input
        for key, value in filtered_data.items():
            setattr(report, key, value) 

        # Optionally call __post_init__ only if you know its fields were set
        if hasattr(report, "name") and hasattr(report, "title") and hasattr(report, "description"):
            report.__post_init__()

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
                report_dict["isBCBS239Program"] = v
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
