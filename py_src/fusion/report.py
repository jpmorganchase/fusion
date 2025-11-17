"""Fusion Report class and functions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import requests

from .data_dependency import DataMapping
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

    from fusion import Fusion

    from .data_dependency import AttributeTermMapping

logger = logging.getLogger(__name__)


@dataclass
class Report(metaclass=CamelCaseMeta):
    """
    Fusion Report class for managing report metadata.

    Attributes:
        title (str | None): Title (Display Name) of the report. Mandatory for report creation.
        description (str | None): Detailed description of the report. Mandatory for report creation.
        frequency (str | None): Frequency of the report. Mandatory for report creation.
        category (str | None): Category of the report. Mandatory for report creation.
        sub_category (str | None): Sub-category of the report providing more specific categorization.
        Mandatory for report creation.
        business_domain (str | None): Business domain for the report.
        This field cannot be left blank if provided.
        regulatory_related (bool | None): Whether the report is related to regulatory requirements.
        Mandatory for creation.
        owner_node (dict[str, str] | None): {"name","type"} of the owner node. Mandatory for creation.
        publisher_node (dict[str, Any] | None): {"name","type"} of the publisher node. Mandatory for creation.
        source_system (dict[str, Any] | None): Source system details for the report.
        lob (str | None): Line of business.
        sub_lob (str | None): Subdivision of the line of business.
        is_bcbs239_program (bool | None): Indicates whether the report is associated with the BCBS 239 program.
        risk_area (str | None): Risk area.
        risk_stripe (str | None): Risk stripe.
        sap_code (str | None): SAP code associated with the report.
        tier_designation (str | None): Tier designation of report.
        region (str | None): Region.
        mnpi_indicator (bool | None): Indicateds whether the report is classified as Material Non-Public Information.
        country_of_reporting_obligation (str | None): Country where the reporting obligation applies.
        primary_regulator (str | None): Primary regulator associated with the report.
        id (str | None): Server-assigned report identifier (needed for update/patch/delete if already known).

        _client (Fusion | None): Fusion client for API calls.
    """

    id: str | None = None
    title: str | None = None
    description: str | None = None
    frequency: str | None = None
    category: str | None = None
    sub_category: str | None = None
    business_domain: str | None = None
    regulatory_related: bool | None = None
    owner_node: dict[str, str] | None = None
    publisher_node: dict[str, Any] | None = None
    source_system: dict[str, Any] | None = None
    lob: str | None = None
    sub_lob: str | None = None
    is_bcbs239_program: bool | None = None
    risk_area: str | None = None
    risk_stripe: str | None = None
    sap_code: str | None = None
    tier_designation: str | None = None
    region: str | None = None
    mnpi_indicator: bool | None = None
    country_of_reporting_obligation: str | None = None
    primary_regulator: str | None = None

    _client: Fusion | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        """Normalize certain text fields after initialization."""
        self.title = tidy_string(self.title or "") if self.title is not None else None
        self.description = tidy_string(self.description or "") if self.description is not None else None
        for n in (
            "business_domain",
            "lob",
            "sub_lob",
            "risk_area",
            "risk_stripe",
            "sap_code",
            "tier_designation",
            "region",
            "country_of_reporting_obligation",
            "primary_regulator",
        ):
            v = getattr(self, n, None)
            if isinstance(v, str):
                setattr(self, n, tidy_string(v))

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
        """Returns the bound Fusion client"""
        return self._client

    @client.setter
    def client(self, client: Fusion | None) -> None:
        """Bind a Fusion client to the Report instance."""
        self._client = client

    def _use_client(self, client: Fusion | None) -> Fusion:
        """Resolve the client to use."""
        res = self._client if client is None else client
        if res is None:
            raise ValueError("A Fusion client object is required.")
        return res

    @classmethod
    def from_dict(cls: type[Report], data: dict[str, Any]) -> Report:
        """Instantiate a Report from a dict."""
        mapped: dict[str, Any] = {camel_to_snake(k): v for k, v in data.items()}

        for k, v in list(mapped.items()):
            if isinstance(v, str) and v.strip() == "":
                mapped[k] = None

        pub = mapped.get("publisher_node")
        if isinstance(pub, dict) and "publisherNodeIdentifier" in pub:
            pub_copy = dict(pub)
            pub_copy["publisher_node_identifier"] = pub_copy.pop("publisherNodeIdentifier")
            mapped["publisher_node"] = pub_copy

        if "is_bcbs239_program" in mapped:
            mapped["is_bcbs239_program"] = make_bool(mapped["is_bcbs239_program"])
        if "regulatory_related" in mapped:
            mapped["regulatory_related"] = make_bool(mapped["regulatory_related"])

        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in mapped.items() if k in allowed}

        report = cls.__new__(cls)
        for fdef in fields(cls):
            setattr(report, fdef.name, filtered.get(fdef.name, None))
        report.__post_init__()
        return report

    def to_dict(self, *, drop_none: bool = False) -> dict[str, Any]:
        """Convert the Report to a dict with camelCase top-level keys.

        Args:
            drop_none(bool): If True, omit top-level fields whose values are None.
        """
        payload: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue

            out_key = "isBCBS239Program" if k == "is_bcbs239_program" else snake_to_camel(k)

            if k == "publisher_node" and isinstance(v, dict):
                node = dict(v)
                if "publisher_node_identifier" in node:
                    node["publisherNodeIdentifier"] = node.pop("publisher_node_identifier")
                payload[out_key] = node
            else:
                payload[out_key] = v

        if drop_none:
            payload = {fn: fv for fn, fv in payload.items() if fv is not None}
        return payload

    @staticmethod
    def _str_or_none(raw: Any) -> str | None:
        """Return string form or None, collapsing floats like 5.0 to '5'."""
        if raw is None:
            return None
        if isinstance(raw, float) and raw.is_integer():
            return str(int(raw))
        return str(raw)

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame, client: Fusion | None = None) -> list[Report]:
        """Create a list of Report objects from a DataFrame, applying permanent column mapping."""
        df_df = data.rename(columns=Report.COLUMN_MAPPING)  # type: ignore[attr-defined]
        df_df = df_df.replace([np.nan, np.inf, -np.inf], None)
        df_df = df_df.where(df_df.notna(), None)

        reports: list[Report] = []
        for _, row in df_df.iterrows():
            report_data: dict[str, Any] = row.to_dict()

            def build_node(d: dict[str, Any], name_key: str, type_key: str) -> dict[str, Any] | None:
                name_val = Report._str_or_none(d.pop(name_key, None))
                type_val = d.pop(type_key, None)
                if name_val or type_val:
                    return {"name": name_val or "", "type": type_val or ""}
                return None

            publisher_node = build_node(report_data, "publisher_node_name", "publisher_node_type")
            owner_node = build_node(report_data, "owner_node_name", "owner_node_type")

            pub_ident = Report._str_or_none(report_data.pop("publisher_node_identifier", None))
            if pub_ident:
                if publisher_node is None:
                    publisher_node = {"name": "", "type": "", "publisher_node_identifier": pub_ident}
                else:
                    publisher_node["publisher_node_identifier"] = pub_ident

            report_data["owner_node"] = owner_node
            report_data["publisher_node"] = publisher_node

            for key in ("is_bcbs239_program", "mnpi_indicator", "regulatory_related"):
                val = report_data.get(key)
                if isinstance(val, str):
                    low = val.strip().lower()
                    if low == "yes":
                        report_data[key] = True
                    elif low == "no":
                        report_data[key] = False

            valid_fields = {f.name for f in fields(cls)}
            report_data = {k: v for k, v in report_data.items() if k in valid_fields}

            report_obj = cls(**report_data)
            report_obj.client = client

            try:
                report_obj.validate()
                reports.append(report_obj)
            except ValueError as e:
                logger.warning("Skipping invalid row: %s", e)

        return reports

    @classmethod
    def from_csv(cls, file_path: str, client: Fusion | None = None) -> list[Report]:
        """Create a list of Report objects from a CSV file, applying permanent column mapping."""
        data = pd.read_csv(file_path)
        return cls.from_dataframe(data, client=client)

    def validate(self) -> None:
        """Validate presence of required fields and node sub-keys."""
        required_fields = [
            "title",
            "description",
            "frequency",
            "category",
            "sub_category",
            "business_domain",
            "regulatory_related",
        ]
        missing = [f for f in required_fields if getattr(self, f, None) in (None, "")]
        if not (self.owner_node and self.owner_node.get("name") and self.owner_node.get("type")):
            missing.append("owner_node.name/type")
        if not (self.publisher_node and self.publisher_node.get("name") and self.publisher_node.get("type")):
            missing.append("publisher_node.name/type")

        if missing:
            raise ValueError(f"Missing required fields in Report: {', '.join(missing)}")

    def create(
        self,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Upload a new report to a Fusion catalog.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> rpt = fusion.report(
            ...     title="Quarterly Report",
            ...     owner_node={"name": "Node1", "type": "User Tool"},
            ...     publisher_node={"name": "Dash1", "type": "Intelligent Solutions"},
            ...     description="Quarterly risk analysis",
            ...     frequency="Quarterly",
            ...     category="Risk",
            ...     sub_category="Ops",
            ...     business_domain="CDO",
            ...     regulatory_related=True,
            ... )
            >>> rpt.create()
        """
        client = self._use_client(client)

        payload = self.to_dict()
        payload.pop("id", None)
        for node_key in ("ownerNode", "publisherNode", "sourceSystem"):
            node = payload.get(node_key)
            if isinstance(node, dict):
                node.pop("id", None)

        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/reports"
        resp: requests.Response = client.session.post(url, json=payload)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def update(
        self,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Update this report with the current object state.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> rpt = fusion.report(id="abc-999")
            >>> rpt.description = "Updated description"
            >>> rpt.frequency = "Monthly"
            >>> rpt.update()
        """
        client = self._use_client(client)
        if not self.id:
            raise ValueError("Report ID is required on the object (set id before update()).")

        payload = self.to_dict()
        payload.pop("id", None)
        payload.pop("title", None)

        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/reports/{self.id}"
        resp: requests.Response = client.session.put(url, json=payload)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def update_fields(
        self,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Partially update this report using the current object state.
        Set attributes you want to change on the object, then call `update_fields()`.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> rpt = fusion.report(id="abc-999", description="deci1")
            >>> rpt.update_fields()
        """
        client = self._use_client(client)
        if not self.id:
            raise ValueError("Report ID is required on the object (set self.id before patch()).")

        payload = self.to_dict(drop_none=True)
        payload.pop("id", None)
        payload.pop("title", None)

        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/reports/{self.id}"
        resp: requests.Response = client.session.patch(url, json=payload)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def delete(
        self,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Delete this report using self.id

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> rpt = fusion.report(id="abc-999")
            >>> rpt.delete()
        """
        client = self._use_client(client)
        if not self.id:
            raise ValueError("Report ID is required on the object (set self.id before delete()).")
        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/reports/{self.id}"
        resp: requests.Response = client.session.delete(url)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    @classmethod
    def link_attributes_to_terms(
        cls,
        mappings: list[AttributeTermMapping],
        client: Fusion,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Link attributes to business terms using data mapping API.
        Args:
            mappings (list[AttributeTermMapping]): List of attribute-to-term mappings.
                Each mapping contains:
                - attribute: DependencyAttribute object with entity details
                - term: dict with term information
                - is_kde: bool indicating if it's a KDE term
            client (Fusion): Fusion client for API requests.
            return_resp_obj (bool): Whether to return the raw response object.
        Returns:
            requests.Response | None: API response if return_resp_obj is True, otherwise None."""

        data_mapping = DataMapping()
        data_mapping.client = client
        return data_mapping.link_attribute_to_term(mappings=mappings, client=client, return_resp_obj=return_resp_obj)


Report.COLUMN_MAPPING = {  # type: ignore[attr-defined]
    "Report/Process Name": "title",
    "Report/Process Description": "description",
    "Frequency": "frequency",
    "Category": "category",
    "Sub Category": "sub_category",
    "Regulatory Designated": "regulatory_related",
    "businessDomain": "business_domain",
    "CDO Office": "business_domain",
    "ownerNode_name": "owner_node_name",
    "ownerNode_type": "owner_node_type",
    "publisherNode_name": "publisher_node_name",
    "publisherNode_type": "publisher_node_type",
    "publisherNode_publisherNodeIdentifier": "publisher_node_identifier",
    "sourceSystem": "source_system",
    "LOB": "lob",
    "Sub-LOB": "sub_lob",
    "JPMSE BCBS Related": "is_bcbs239_program",
    "Report Type": "risk_stripe",
    "Risk Area": "risk_area",
    "Tier Type": "tier_designation",
    "Region": "region",
    "MNPI Indicator": "mnpi_indicator",
    "Country of Reporting Obligation": "country_of_reporting_obligation",
    "Primary Regulator": "primary_regulator",
}


class Reports:
    """Container for a list of Report objects with convenience loaders."""

    def __init__(self, reports: list[Report] | None = None, client: Fusion | None = None) -> None:
        self.reports = reports or []
        self._client: Fusion | None = client
        if client:
            for report in self.reports:
                report.client = client

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

    def _use_client(self, client: Fusion | None) -> Fusion:
        res = self._client if client is None else client
        if res is None:
            raise ValueError("A Fusion client object is required.")
        return res

    def from_csv(self, file_path: str) -> Reports:
        """Load Reports from a CSV file path.


        Examples:
            >>> reports = fusion.reports()
            >>> reports = reports.from_csv("reports.csv")
        """
        csv_df = pd.read_csv(file_path)
        return self.from_dataframe(csv_df)

    def from_dataframe(self, df: pd.DataFrame) -> Reports:
        """Load Reports from a pandas DataFrame.

        Examples:
            >>> reports = fusion.reports()
            >>> df = pd.DataFrame([...])
            >>> reports = reports.from_dataframe(df)
        """
        client = self._use_client(None)
        report_objs = Report.from_dataframe(df, client=client)
        self.reports = report_objs
        self.client = client
        return self

    def from_object(self, source: pd.DataFrame | list[dict[str, Any]] | str) -> Reports:
        """Load Reports from DataFrame, list of dicts, .csv path, or JSON string

        Examples:
            >>> reports = fusion.reports()
            >>> reports = reports.from_object("reports.csv")
            >>> reports = reports.from_object(pd.DataFrame([...]))
            >>> reports = reports.from_object('[{"title": "R1", ...}]')
        """
        import json

        if isinstance(source, pd.DataFrame):
            return self.from_dataframe(source)

        if isinstance(source, list) and all(isinstance(item, dict) for item in source):
            return self.from_dataframe(pd.DataFrame(source))

        if isinstance(source, str):
            s = source.strip()
            if s.lower().endswith(".csv") and Path(s).exists():
                return self.from_csv(s)
            if s.startswith("[{"):
                return self.from_dataframe(pd.DataFrame(json.loads(s)))

        raise TypeError("source must be a DataFrame, list of dicts, or string (.csv path or JSON)")

    def create_all(
        self,
        client: Fusion | None = None,
        *,
        return_resp_obj: bool = False,
    ) -> list[requests.Response] | None:
        client = self._use_client(client)
        if not return_resp_obj:
            for r in self.reports:
                r.create(client=client)
            return None

        responses: list[requests.Response] = [
            cast(requests.Response, r.create(client=client, return_resp_obj=True)) for r in self.reports
        ]
        return responses
