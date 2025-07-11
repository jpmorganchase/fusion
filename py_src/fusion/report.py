"""Fusion Report class and functions."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, TypedDict

import pandas as pd 

from .utils import (
    CamelCaseMeta,
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
    title: str
    description: str
    frequency: str
    category: str
    sub_category: str
    domain: dict[str, str]  # Dictionary with "name" key populated from "CDO Office"
    regulatory_related: str

    # Optional fields
    sub_lob: str | None = None
    is_bcbs239_program: bool | None = None
    risk_stripe: str | None = None
    tier_designation: str | None = None
    region: str | None = None
    mnpi_indicator: bool | None = None
    country_of_reporting_obligation: str | None = None
    primary_regulator: str | None = None


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
        """Instantiate a Report object from a dictionary.

        All fields defined in the class will be set.
        If a field is missing from input, it will be set to None.
        """

        def normalize_value(val: Any) -> Any:
            if isinstance(val, str) and val.strip() == "":
                return None
            return val

        def convert_keys(d: dict[str, Any]) -> dict[str, Any]:
            converted = {}
            for k, v in d.items():
                # Special case: keep as-is if already matches the field name (e.g., isBCBS239Program)
                key = k if k == "isBCBS239Program" else camel_to_snake(k)
                if isinstance(v, dict) and not isinstance(v, str):
                    converted[key] = convert_keys(v)
                else:
                    converted[key] = normalize_value(v)
            return converted

        converted_data = convert_keys(data)

        if "isBCBS239Program" in converted_data:
            converted_data["isBCBS239Program"] = make_bool(converted_data["isBCBS239Program"])

        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in converted_data.items() if k in valid_fields}

        report = cls.__new__(cls)

        for fieldsingle in fields(cls):
            if fieldsingle.name in filtered_data:
                setattr(report, fieldsingle.name, filtered_data[fieldsingle.name])
            else:
                setattr(report, fieldsingle.name, None)

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
    @classmethod
    def map_application_type(cls, app_type: str) -> str:
        """Map application types to enum values."""
        mapping = {
            "Application (SEAL)": "Application (SEAL)",
            "Intelligent Solutions": "Intelligent Solutions",
            "User Tool": "User Tool"
        }
        return mapping.get(app_type, None)

    @classmethod
    def map_tier_type(cls, tier_type: str) -> str:
        """Map tier types to enum values."""
        tier_mapping = {
            "Tier 1": "Tier 1",
            "Non Tier 1": "Non Tier 1"
        }
        return tier_mapping.get(tier_type, None)
        

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame) -> list[Report]:
        """
        Create a list of Report objects from a DataFrame, applying permanent column mapping.

        Args:
            data (pd.DataFrame): DataFrame containing report data.

        Returns:
            list[Report]: List of Report objects.
        """
        # Apply permanent column mapping
        data = data.rename(columns=Report.COLUMN_MAPPING)

        # Replace NaN with None
        data = data.where(data.notna(), None)

        # Process each row and create Report objects
        reports = []
        for _, row in data.iterrows():
            report_data = row.to_dict()

            # Handle nested fields like domain and data_node_id
            report_data["domain"] = {"name": report_data.pop("domain_name", None)}  # Populate "name" inside "domain"
            report_data["data_node_id"] = {
                "name": report_data.pop("data_node_name", None),
                "dataNodeType": cls.map_application_type(report_data.pop("data_node_type", None)),
            }
            # Convert boolean fields
            is_bcbs = report_data.get("is_bcbs239_program")
            report_data["is_bcbs239_program"] = is_bcbs == "Yes" if is_bcbs else None

            mnpi = report_data.get("mnpi_indicator")
            report_data["mnpi_indicator"] = mnpi == "Yes" if mnpi else None

            reg_related = report_data.get("regulatory_related")
            report_data["regulatory_related"] = reg_related == "Yes" if reg_related else None


            # Map tier designation
            tier_val = report_data.get("tier_designation")
            report_data["tier_designation"] = cls.map_tier_type(tier_val) if tier_val else None


            reports.append(cls(**report_data))

        return reports

    @classmethod
    def from_csv(cls, file_path: str) -> list[Report]:
        """
        Create a list of Report objects from a CSV file, applying permanent column mapping.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            list[Report]: List of Report objects.
        """
        data = pd.read_csv(file_path)
        return cls.from_dataframe(data)

   

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

        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/reports"
        resp: requests.Response = client.session.post(url, json=data)
        requests_raise_for_status(resp)

        return resp if return_resp_obj else None
    
    class AttributeTermMapping(TypedDict):
        attribute: dict[str, str]
        term: dict[str, str]
        isKDE: bool


    def link_attributes_to_terms(
        self,
        report_id: str,
        mappings: list[AttributeTermMapping],
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """
        Links attributes to business terms for a report using pre-formatted mappings.

        Args:
            report_id (str): Report identifier.
            mappings (list[dict]): List of mappings in the format:
                {
                    "attribute": {"id": str},
                    "term": {"id": str},
                    "isKDE": bool
                }
            return_resp_obj (bool, optional): If True, returns the response object. Defaults to False.

        Returns:
            requests.Response | None: Response object from the API if return_resp_obj is True, otherwise None.
        """
        # Validate mappings structure
        for i, m in enumerate(mappings):
            if not isinstance(m, dict):
                raise ValueError(f"Mapping at index {i} is not a dictionary.")
            if not ("attribute" in m and "term" in m and "isKDE" in m):
                raise ValueError(f"Mapping at index {i} must include 'attribute', 'term', and 'isKDE'.")

        # Get the Fusion client
        client = self._use_client(None)
        base = client._get_new_root_url()
        url = f"{base}/api/corelineage-service/v1/reports/{report_id}/reportElements/businessTerms"


        # Make the request
        response = client.session.post(url, json=mappings)
        requests_raise_for_status(response)

        return response if return_resp_obj else None


Report.COLUMN_MAPPING = {
            "Report/Process Name": "title",
            "Report/Process Description": "description",
            "Activity Type": "tier_type",
            "Frequency": "frequency",
            "Category": "category",
            "Report/Process Owner SID": "report_owner",
            "Sub Category": "sub_category",
            "LOB": "lob",
            "Sub-LOB": "sub_lob",
            "JPMSE BCBS Related": "is_bcbs239_program",
            "Report Type": "risk_stripe",
            "Tier Type": "tier_designation",
            "Region": "region",
            "MNPI Indicator": "mnpi_indicator",
            "Country of Reporting Obligation": "country_of_reporting_obligation",
            "Regulatory Designated": "regulatory_related",
            "Primary Regulator": "primary_regulator",
            "CDO Office": "domain_name",  # Map to "name" inside "domain"
            "Application ID": "data_node_name",
            "Application Type": "data_node_type",
        }