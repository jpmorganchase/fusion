"""Fusion Report class and functions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
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
    from collections.abc import Iterator

    import requests

    from fusion import Fusion


logger = logging.getLogger(__name__)


@dataclass
class Report(metaclass=CamelCaseMeta):
    """
    Fusion Report class for managing report metadata.

    Attributes:
        title (str): Title of the report or process.
        data_node_id (dict[str, str]): Identifier of the associated data node (e.g., name, dataNodeType).
        description (str): Description of the report.
        frequency (str): Reporting frequency (e.g., Monthly, Quarterly).
        category (str): Primary category classification.
        sub_category (str): Sub-category under the main category.
        domain (dict[str, str]): Domain metadata (typically with a "name" key).
        regulatory_related (bool): Whether the report is regulatory-related.

        lob (str, optional): Line of Business.
        sub_lob (str, optional): Subdivision of the Line of Business.
        tier_type (str, optional): Report's tier classification.
        is_bcbs239_program (bool, optional): Flag indicating BCBS 239 program inclusion.
        risk_stripe (str, optional): Stripe under risk category.
        risk_area (str, optional): The area of risk addressed.
        sap_code (str, optional): Associated SAP cost code.
        tier_designation (str, optional): Tier designation (e.g., Tier 1, Non Tier 1).
        alternative_id (dict[str, str], optional): Alternate report identifiers.
        region (str, optional): Associated region.
        mnpi_indicator (bool, optional): Whether report contains MNPI.
        country_of_reporting_obligation (str, optional): Country of regulatory obligation.
        primary_regulator (str, optional): Main regulatory authority.

        _client (Fusion, optional): Fusion client for making API calls (injected automatically).
    """

    title: str
    data_node_id: dict[str, str]
    description: str
    frequency: str
    category: str
    sub_category: str
    domain: dict[str, str]
    regulatory_related: bool

    # Optional fields
    lob: str | None = None
    sub_lob: str | None = None
    tier_type: str | None = None
    is_bcbs239_program: bool | None = None
    risk_stripe: str | None = None
    risk_area: str | None = None
    sap_code: str | None = None
    tier_designation: str | None = None
    alternative_id: dict[str, str] | None = None
    region: str | None = None
    mnpi_indicator: bool | None = None
    country_of_reporting_obligation: str | None = None
    primary_regulator: str | None = None

    _client: Fusion | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        self.title = tidy_string(self.title or "")
        self.description = tidy_string(self.description or "")

    def __getattr__(self, name: str) -> Any:
        snake_name = camel_to_snake(name)
        return self.__dict__.get(snake_name, None)

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

    def validate(self) -> None:
        required_fields = ["title", "data_node_id", "category", "frequency", "description", "sub_category", "domain"]
        missing = [f for f in required_fields if getattr(self, f, None) in [None, ""]]
        if missing:
            raise ValueError(f"Missing required fields in Report: {', '.join(missing)}")

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
            elif k == "regulatory_related":
                report_dict["regulatoryRelated"] = v
            else:
                report_dict[snake_to_camel(k)] = v

        return report_dict

    @classmethod
    def map_application_type(cls, app_type: str) -> str | None:
        """Map application types to enum values."""
        mapping = {
            "Application (SEAL)": "Application (SEAL)",
            "Intelligent Solutions": "Intelligent Solutions",
            "User Tool": "User Tool",
        }
        return mapping.get(app_type)

    @classmethod
    def map_tier_type(cls, tier_type: str) -> str | None:
        """Map tier types to enum values."""
        tier_mapping = {"Tier 1": "Tier 1", "Non Tier 1": "Non Tier 1"}
        return tier_mapping.get(tier_type)

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame, client: Fusion | None = None) -> list[Report]:
        """
        Create a list of Report objects from a DataFrame, applying permanent column mapping.

        Args:
            data (pd.DataFrame): DataFrame containing report data.

        Returns:
            list[Report]: List of Report objects.
        """
        # Apply permanent column mapping
        data = data.rename(columns=Report.COLUMN_MAPPING)  # type: ignore[attr-defined]
        data = data.replace([np.nan, np.inf, -np.inf], None)  # Replace NaN, inf, -inf with None

        # Replace NaN with None
        data = data.where(data.notna(), None)

        # Process each row and create Report objects
        reports = []
        for _, row in data.iterrows():
            report_data = row.to_dict()

            # Handle nested fields like domain and data_node_id
            report_data["domain"] = {"name": report_data.pop("domain_name", None)}  # Populate "name" inside "domain"
            raw_value = report_data.pop("data_node_name", None)

            if raw_value is None:
                name_str = None
            elif isinstance(raw_value, float) and raw_value.is_integer():
                name_str = str(int(raw_value))  # convert 2679.0 → "2679"
            else:
                name_str = str(raw_value)

            report_data["data_node_id"] = {
                "name": name_str,
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
            # Filter out any fields not defined in the class
            # This ensures that only valid fields are passed to the Report constructor
            valid_fields = {f.name for f in fields(cls)}
            report_data = {k: v for k, v in report_data.items() if k in valid_fields}

            report_obj = cls(**report_data)
            report_obj.client = client  # Attach the client if provided

            try:
                report_obj.validate()
                reports.append(report_obj)
            except ValueError as e:
                logger.warning(f"Skipping invalid row: {e}")

        return reports

    @classmethod
    def from_csv(cls, file_path: str, client: Fusion | None = None) -> list[Report]:
        """
        Create a list of Report objects from a CSV file, applying permanent column mapping.

        Args:
            file_path (str): Path to the CSV file.
            client (Any, optional): Client instance to attach to each Report.

        Returns:
            list[Report]: List of Report objects.
        """
        data = pd.read_csv(file_path)
        return cls.from_dataframe(data, client=client)

    @classmethod
    def from_object(cls, source: pd.DataFrame | list[dict[str, Any]] | str, client: Fusion | None = None) -> Reports:
        """Unified loader for Reports from CSV path, DataFrame, list of dicts, or JSON string."""
        if isinstance(source, pd.DataFrame):
            return Reports(cls.from_dataframe(source, client=client))

        elif isinstance(source, list) and all(isinstance(item, dict) for item in source):
            df = pd.DataFrame(source)  # noqa
            return Reports(cls.from_dataframe(df, client=client))

        elif isinstance(source, str):
            if source.strip().endswith(".csv"):
                return Reports(cls.from_csv(source, client=client))
            elif source.strip().startswith("[{"):
                import json

                data = json.loads(source)
                df = pd.DataFrame(data)  # noqa
                return Reports(cls.from_dataframe(df, client=client))
            else:
                raise ValueError("Unsupported string input — must be .csv path or JSON array string")

        raise TypeError("source must be a DataFrame, list of dicts, or string (.csv path or JSON)")

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

    @classmethod
    def link_attributes_to_terms(
        cls,
        report_id: str,
        mappings: list[Report.AttributeTermMapping],
        client: Fusion,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """
        Class method to link attributes to business terms for a report.

        Can be called without an actual Report object.

        Args:
            report_id (str): ID of the report to link terms to.
            mappings (list): List of attribute-to-term mappings.
            client (Fusion): Fusion client instance.
            return_resp_obj (bool): Whether to return the raw response object.

        Returns:
            requests.Response | None: API response
        """
        url = (
            f"{client._get_new_root_url()}/api/corelineage-service/v1/reports/{report_id}/reportElements/businessTerms"  # noqa: E501
        )
        resp = client.session.post(url, json=mappings)
        requests_raise_for_status(resp)

        return resp if return_resp_obj else None


Report.COLUMN_MAPPING = {  # type: ignore[attr-defined]
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
    "CDO Office": "domain_name",
    "Application ID": "data_node_name",
    "Application Type": "data_node_type",
}


class Reports:
    def __init__(self, reports: list[Report] | None = None) -> None:
        self.reports = reports or []
        self._client: Fusion | None = None

    def __getitem__(self, index: int) -> Report:
        return self.reports[index]

    def __iter__(self) -> Iterator[Report]:
        return iter(self.reports)

    def __len__(self) -> int:
        return len(self.reports)

    @property
    def client(self) -> Fusion | None:
        return self._client

    @client.setter
    def client(self, client: Fusion | None) -> None:
        self._client = client
        for report in self.reports:
            report.client = client

    @classmethod
    def from_csv(cls, file_path: str, client: Fusion | None = None) -> Reports:
        data = pd.read_csv(file_path)
        return cls.from_dataframe(data, client=client)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, client: Fusion | None = None) -> Reports:
        report_objs = Report.from_dataframe(df, client=client)
        obj = cls(report_objs)
        obj.client = client
        return obj

    def create_all(self) -> None:
        for report in self.reports:
            report.create()

    @classmethod
    def from_object(cls, source: pd.DataFrame | list[dict[str, Any]] | str, client: Fusion | None = None) -> Reports:
        """Load Reports from DataFrame, list of dicts, .csv path, or JSON string."""
        import json

        if isinstance(source, pd.DataFrame):
            return cls.from_dataframe(source, client=client)

        elif isinstance(source, list) and all(isinstance(item, dict) for item in source):
            return cls.from_dataframe(pd.DataFrame(source), client=client)

        elif isinstance(source, str):
            if source.lower().endswith(".csv") and Path(source).exists():
                return cls.from_csv(source, client=client)

            elif source.strip().startswith("[{"):
                dict_list = json.loads(source)
                return cls.from_dataframe(pd.DataFrame(dict_list), client=client)
            else:
                raise ValueError("Unsupported string input — must be .csv path or JSON array string")

        raise TypeError("source must be a DataFrame, list of dicts, or string (.csv path or JSON)")


class ReportsWrapper(Reports):
    def __init__(self, client: Fusion) -> None:
        super().__init__([])
        self.client = client

    def from_csv(self, file_path: str) -> Reports:  # type: ignore[override]
        return Reports.from_csv(file_path, client=self.client)

    def from_dataframe(self, df: pd.DataFrame) -> Reports:  # type: ignore[override]
        return Reports.from_dataframe(df, client=self.client)

    def from_object(self, source: pd.DataFrame | list[dict[str, Any]] | str) -> Reports:  # type: ignore[override]
        return Reports.from_object(source, client=self.client)
