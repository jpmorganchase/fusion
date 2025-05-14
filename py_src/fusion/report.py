"""Fusion Report class and functions."""

from __future__ import annotations

import json as js
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import pandas as pd
import requests

from .utils import (
    CamelCaseMeta,
    _is_json,
    camel_to_snake,
    make_bool,
    requests_raise_for_status,
    tidy_string,
)

if TYPE_CHECKING:
    from fusion import Fusion


@dataclass
class Report(metaclass=CamelCaseMeta):
    """Fusion Report class for managing report metadata.

    Attributes:
        name (str): A unique name for the report.
        tier_type (str): The tier classification of the report.
        alternate_id (str): An alternate identifier for the report.
        data_node_id (dict[str, str]): Identifier of the associated data node.
        title (str, optional): A title for the report. Defaults to "".
        frequency (str, optional): The frequency of the report. Defaults to "".
        category (str, optional): Business category. Defaults to "".
        sub_category (str, optional): Sub-category for classification. Defaults to "".
        report_inventory_name (str, optional): Internal inventory name. Defaults to "".
        report_owner (str, optional): Owner responsible for the report. Defaults to "".
        lob (str, optional): Line of business. Defaults to "".
        sub_lob (str, optional): Sub-line of business. Defaults to "".
        is_bcbs239_program (bool, optional): Part of BCBS239 program. Defaults to False.
        risk_area (str, optional): Risk area the report covers. Defaults to "".
        riskstripe (str, optional): Risk classification. Defaults to "".
        sap_code (str, optional): SAP code. Defaults to "".
        domain (str, optional): Business or data domain. Defaults to "".
        sourced_object (str, optional): Source object. Defaults to "".
        alternative_id (dict[str, str], optional): Additional alternate IDs. Defaults to empty dict.
        data_model_id (dict[str, str], optional): Associated data model. Defaults to empty dict.
        id (str, optional): Unique identifier. Defaults to "".
        description (str, optional): Description of the report. Defaults to "".
        report_inventory_id (str, optional): Inventory ID. Defaults to "".
        created_service (str, optional): Service that created the report. Defaults to "".
        originator_firm_id (str, optional): Originator firm ID. Defaults to "".
        is_instance (bool, optional): Indicates if this is an instance. Defaults to False.
        version (str, optional): Report version. Defaults to "".
        status (str, optional): Current status. Defaults to "".
        created_by (str, optional): Creator. Defaults to "".
        created_datetime (str, optional): Creation timestamp. Defaults to "".
        modified_by (str, optional): Last modifier. Defaults to "".
        modified_datetime (str, optional): Last modified timestamp. Defaults to "".
        approved_by (str, optional): Approver. Defaults to "".
        approved_datetime (str, optional): Approval timestamp. Defaults to "".
    """

    name: str
    tier_type: str
    alternate_id: str
    data_node_id: dict[str, str]
    title: str = ""
    frequency: str = ""
    category: str = ""
    sub_category: str = ""
    report_inventory_name: str = ""
    report_owner: str = ""
    lob: str = ""
    sub_lob: str = ""
    is_bcbs239_program: bool = False
    risk_area: str = ""
    riskstripe: str = ""
    sap_code: str = ""
    domain: str = ""
    sourced_object: str = ""
    alternative_id: dict[str, str]
    data_model_id: dict[str, str]
    id: str = ""
    description: str = ""
    report_inventory_id: str = ""
    created_service: str = ""
    originator_firm_id: str = ""
    is_instance: bool = False
    version: str = ""
    status: str = ""
    created_by: str = ""
    created_datetime: str = ""
    modified_by: str = ""
    modified_datetime: str = ""
    approved_by: str = ""
    approved_datetime: str = ""

    _client: Fusion | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        self.name = tidy_string(self.name).upper().replace(" ", "_")
        self.title = tidy_string(self.title)
        self.description = tidy_string(self.title)
        self.is_bcbs239_program = make_bool(self.is_bcbs239_program)

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
    def _from_series(cls: type[Report], series: pd.Series) -> Report:
        """Instantiate a Report object from a pandas Series."""
        series = series.rename(lambda x: x.replace(" ", "").replace("_", "").lower())

        # Normalize booleans
        is_bcbs239_program = series.get("isbcbs239program", None)
        is_bcbs239_program = make_bool(is_bcbs239_program) if is_bcbs239_program is not None else False

        return cls(
            name=series.get("name", ""),
            title=series.get("title", ""),
            alternate_id=series.get("alternateid", ""),
            tier_type=series.get("tiertype", ""),
            frequency=series.get("frequency", ""),
            category=series.get("category", ""),
            sub_category=series.get("subcategory", ""),
            report_inventory_name=series.get("reportinventoryname", ""),
            report_owner=series.get("reportowner", ""),
            lob=series.get("lob", ""),
            sub_lob=series.get("sublob", ""),
            is_bcbs239_program=is_bcbs239_program,
            risk_area=series.get("riskarea", ""),
            riskstripe=series.get("riskstripe", ""),
            sap_code=series.get("sapcode", ""),
            domain=series.get("domain", ""),
            sourced_object=series.get("sourcedobject", ""),
            alternative_id=series.get("alternativeid", {"domain": "", "Value": ""}),
            data_model_id=series.get("datamodelid", {"domain": "", "Value": ""}),
            data_node_id=series.get("datanodeid", {"Id": "", "dataNodeType": ""}),
        )

    @classmethod
    def _from_dict(cls: type[Report], data: dict[str, Any]) -> Report:
        """Instantiate a Report object from a dictionary.

        Args:
            data (dict[str, Any]): Report metadata as a dictionary.

        Returns:
            Report: Report object.
        """
        keys = [f.name for f in fields(cls)]
        data = {camel_to_snake(k): v for k, v in data.items()}
        data = {k: v for k, v in data.items() if k in keys}
        return cls(**data)

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
        
        # Set default dates if not provided
        # self.created_date = self.created_date or pd.Timestamp("today").strftime("%Y-%m-%d")
        # self.modified_date = self.modified_date or pd.Timestamp("today").strftime("%Y-%m-%d")

        data = self.to_dict()

        url = f"{client.root_url}/corelineage-service/api/v1/reports"
        resp: requests.Response = client.session.post(url, json=data)
        requests_raise_for_status(resp)

        return resp if return_resp_obj else None

