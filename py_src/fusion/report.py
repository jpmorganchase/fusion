"""Fusion Report class and functions."""

from __future__ import annotations

import logging
from contextlib import suppress
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
        id (str | None): Server-assigned report identifier. Required for update/patch/delete.
        title (str | None): Title/ Display name of the report.
        description (str | None): Description of the report.
        frequency (str | None): Frequency of the report.
        category (str | None): Primary category classification.
        sub_category (str | None): Sub-category under the main category.
        business_domain (str | None): Business domain string (e.g., "CDAO Office").
        regulatory_related (bool | None): Whether the report is regulatory-related.

        owner_node (dict[str, str] | None): Owner node with keys {"name", "type"}.
        publisher_node (dict[str, Any] | None): Publisher node with keys {"name", "type"} and optional
            {"publisher_node_identifier"}.

        source_system (dict[str, Any] | None): Source system object if provided.

        lob (str | None): Line of Business associated with the Report.
        sub_lob (str | None): Subdivision of the Line of Business.
        is_bcbs239_program (bool | None): Flag indicating BCBS 239 program inclusion.
        risk_stripe (str | None): Stripe under risk category.
        risk_area (str | None): The area of risk addressed.
        sap_code (str | None): Associated SAP cost code.
        tier_designation (str | None): Tier designation (e.g., Tier 1, Non Tier 1).
        region (str | None): Associated region.
        mnpi_indicator (bool | None): Whether report contains MNPI.
        country_of_reporting_obligation (str | None): Country of regulatory obligation.
        primary_regulator (str | None): Main regulatory authority.

        _client (Fusion | None): Fusion client for making API calls (injected automatically).
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
        """Instantiate Report from a dict with light-touch key handling.

        - Convert only top-level keys from camelCase → snake_case
        - Collapse top-level empty strings "" → None
        - Minimal nested fix: publisherNode.publisherNodeIdentifier → publisher_node["publisher_node_identifier"]
        - Minimal bool normalization for isBCBS239Program / regulatoryRelated
        """
        # Top-level camelCase → snake_case
        mapped: dict[str, Any] = {camel_to_snake(k): v for k, v in data.items()}

        # Collapse top-level "" → None
        for k, v in list(mapped.items()):
            if isinstance(v, str) and v.strip() == "":
                mapped[k] = None

        # Targeted nested handling for publisherNodeIdentifier
        pub = mapped.get("publisher_node")
        if isinstance(pub, dict) and "publisherNodeIdentifier" in pub:
            pub_copy = dict(pub)
            pub_copy["publisher_node_identifier"] = pub_copy.pop("publisherNodeIdentifier")
            mapped["publisher_node"] = pub_copy

        # Minimal bool normalization
        if "is_bcbs239_program" in mapped:
            mapped["is_bcbs239_program"] = make_bool(mapped["is_bcbs239_program"])
        if "regulatory_related" in mapped:
            mapped["regulatory_related"] = make_bool(mapped["regulatory_related"])

        # Keep only valid dataclass fields
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in mapped.items() if k in allowed}

        # Construct without calling __init__ so we can set fields directly, then run post-init
        report = cls.__new__(cls)
        for fdef in fields(cls):
            setattr(report, fdef.name, filtered.get(fdef.name, None))
        report.__post_init__()
        return report

    def to_dict(self) -> dict[str, Any]:
        """Convert the Report instance to a dictionary (camelCase top-level keys, minimal nesting changes)."""
        payload: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue

            # Special-case BCBS field, otherwise camelCase the top-level key
            out_key = "isBCBS239Program" if k == "is_bcbs239_program" else snake_to_camel(k)

            # Minimal nested handling: only fix publisher_node_identifier inside publisher_node
            if k == "publisher_node" and isinstance(v, dict):
                node = dict(v)  # shallow copy
                if "publisher_node_identifier" in node:
                    node["publisherNodeIdentifier"] = node.pop("publisher_node_identifier")
                payload[out_key] = node
            else:
                payload[out_key] = v

        return payload

    @staticmethod
    def _str_or_none(raw: Any) -> str | None:
        """Return string form or None, collapsing floats like 5.0 -> '5'."""
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

    @classmethod
    def from_object(cls, source: pd.DataFrame | list[dict[str, Any]] | str, client: Fusion | None = None) -> Reports:
        """Unified loader for Reports from CSV path, DataFrame, list of dicts, or JSON string."""
        if isinstance(source, pd.DataFrame):
            return Reports(cls.from_dataframe(source, client=client))

        elif isinstance(source, list) and all(isinstance(item, dict) for item in source):
            df = pd.DataFrame(source)  # noqa
            return Reports(cls.from_dataframe(df, client=client))

        elif isinstance(source, str):
            s = source.strip()
            if s.endswith(".csv"):
                return Reports(cls.from_csv(source, client=client))
            elif s.startswith("[{"):
                import json
                data = json.loads(s)
                df = pd.DataFrame(data)  # noqa
                return Reports(cls.from_dataframe(df, client=client))
            else:
                raise ValueError("Unsupported string input — must be .csv path or JSON array string")

        raise TypeError("source must be a DataFrame, list of dicts, or string (.csv path or JSON)")

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
        """Upload a new report to a Fusion catalog."""
        client = self._use_client(client)
        payload = self.to_dict()

        def _strip_ids(x: Any) -> Any:
            if isinstance(x, dict):
                return {k: _strip_ids(v) for k, v in x.items() if k != "id"}
            if isinstance(x, list):
                return [_strip_ids(i) for i in x]
            return x

        payload = _strip_ids(payload)
        payload.pop("id", None)

        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/reports"
        resp: requests.Response = client.session.post(url, json=payload)
        requests_raise_for_status(resp)
        with suppress(Exception):
            self.id = resp.json().get("id", self.id)
        return resp if return_resp_obj else None

    def update(
        self,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Update this report with the current object state."""
        client = self._use_client(client)
        if not self.id:
            raise ValueError("Report ID is required on the object (set self.id before update()).")

        payload = self.to_dict()
        payload.pop("id", None)
        payload.pop("title", None)  # title immutable for PUT

        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/reports/{self.id}"
        resp: requests.Response = client.session.put(url, json=payload)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def update_fields(
        self,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Partially update this report (PATCH) using the current object state.

        This sends the current field values of the instance (except `id`) as a PATCH.
        Set attributes you want to change on the object, then call `update_fields()`.
        """
        client = self._use_client(client)
        if not self.id:
            raise ValueError("Report ID is required on the object (set self.id before patch()).")

        payload = self.to_dict()
        payload.pop("id", None)

        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/reports/{self.id}"
        resp: requests.Response = client.session.patch(url, json=payload)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def delete(
        self,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Delete this report using self.id."""
        client = self._use_client(client)
        if not self.id:
            raise ValueError("Report ID is required on the object (set self.id before delete()).")
        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/reports/{self.id}"
        resp: requests.Response = client.session.delete(url)
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
        """Link attributes to business terms for a report."""
        url = (
            f"{client._get_new_root_url()}/api/corelineage-service/v1/reports/{report_id}/reportElements/businessTerms"
        )
        resp = client.session.post(url, json=mappings)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None


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
        """Return the bound client for all contained reports (if any)."""
        return self._client

    @client.setter
    def client(self, client: Fusion | None) -> None:
        """Bind a client to this collection and all contained reports."""
        self._client = client
        for report in self.reports:
            report.client = client

    @classmethod
    def from_csv(cls, file_path: str, client: Fusion | None = None) -> Reports:
        """Load Reports from a CSV file path."""
        data = pd.read_csv(file_path)
        return cls.from_dataframe(data, client=client)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, client: Fusion | None = None) -> Reports:
        """Load Reports from a pandas DataFrame."""
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