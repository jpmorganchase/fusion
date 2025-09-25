"""Fusion Report class and functions."""

from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

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
        id (str, optional): Server-assigned report identifier. Required for update/patch/delete.
        title (str): Title of the report or process.
        description (str): Description of the report.
        frequency (str): Reporting frequency (e.g., Monthly, Quarterly).
        category (str): Primary category classification.
        sub_category (str): Sub-category under the main category.
        business_domain (str): Business domain string (e.g., "CDAO Office").
        regulatory_related (bool): Whether the report is regulatory-related.

        owner_node (dict[str, str], optional): Owner node with keys {"name", "type"}.
        publisher_node (dict[str, Any], optional): Publisher node with keys {"name", "type"} and optional
            {"publisher_node_identifier"} which serializes to "publisherNodeIdentifier" in API.

        source_system (dict[str, Any], optional): Source system object if provided.

        lob (str, optional): Line of Business.
        sub_lob (str, optional): Subdivision of the Line of Business.
        is_bcbs239_program (bool, optional): Flag indicating BCBS 239 program inclusion.
        risk_stripe (str, optional): Stripe under risk category.
        risk_area (str, optional): The area of risk addressed.
        sap_code (str, optional): Associated SAP cost code.
        tier_designation (str, optional): Tier designation (e.g., Tier 1, Non Tier 1).
        region (str, optional): Associated region.
        mnpi_indicator (bool, optional): Whether report contains MNPI.
        country_of_reporting_obligation (str, optional): Country of regulatory obligation.
        primary_regulator (str, optional): Main regulatory authority.

        _client (Fusion, optional): Fusion client for making API calls (injected automatically).
    """

    # Server id for existing reports
    id: str | None = None

    # Required (per new schema)
    title: str | None = None
    description: str | None = None
    frequency: str | None = None
    category: str | None = None
    sub_category: str | None = None
    business_domain: str | None = None
    regulatory_related: bool | None = None

    # Node fields (new schema)
    owner_node: dict[str, str] | None = None
    # may include {"publisher_node_identifier"} (becomes publisherNodeIdentifier in payload)
    publisher_node: dict[str, Any] | None = None

    # Optional complex
    source_system: dict[str, Any] | None = None

    # Optionals retained
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

    # -------------------- lifecycle --------------------

    def __post_init__(self) -> None:
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
        return self._client

    @client.setter
    def client(self, client: Fusion | None) -> None:
        self._client = client

    def _use_client(self, client: Fusion | None) -> Fusion:
        res = self._client if client is None else client
        if res is None:
            raise ValueError("A Fusion client object is required.")
        return res

    # -------------------- conversions --------------------

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
            converted: dict[str, Any] = {}
            for k, v in d.items():
                key = k if k == "isBCBS239Program" else camel_to_snake(k)
                if isinstance(v, dict):
                    converted[key] = convert_keys(v)
                else:
                    converted[key] = normalize_value(v)
            return converted

        converted_data = convert_keys(data)

        # explicit boolean normalization
        if "isBCBS239Program" in converted_data:
            converted_data["isBCBS239Program"] = make_bool(converted_data["isBCBS239Program"])

        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in converted_data.items() if k in valid}

        report = cls.__new__(cls)
        for fdef in fields(cls):
            setattr(report, fdef.name, filtered.get(fdef.name, None))
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

        def _camelize(obj: Any) -> Any:
            if isinstance(obj, dict):
                out: dict[str, Any] = {}
                for k, v in obj.items():
                    if k.startswith("_"):
                        continue
                    if k == "is_bcbs239_program":
                        ck = "isBCBS239Program"
                    elif k == "regulatory_related":
                        ck = "regulatoryRelated"
                    else:
                        ck = snake_to_camel(k)
                    out[ck] = _camelize(v)
                return out
            if isinstance(obj, list):
                return [_camelize(x) for x in obj]
            return obj

        payload = cast(dict[str, Any], _camelize({k: v for k, v in self.__dict__.items() if not k.startswith("_")}))

        # make publisherNode order pretty: name, type, publisherNodeIdentifier
        pn = payload.get("publisherNode")
        if isinstance(pn, dict):
            ordered: dict[str, Any] = {}
            for key in ("name", "type", "publisherNodeIdentifier"):
                if key in pn:
                    ordered[key] = pn[key]
            for key, val in pn.items():
                if key not in ordered:
                    ordered[key] = val
            payload["publisherNode"] = ordered

        return payload

    # -------------------- helpers --------------------

    @staticmethod
    def _str_or_none(raw: Any) -> str | None:
        if raw is None:
            return None
        if isinstance(raw, float) and raw.is_integer():
            return str(int(raw))
        return str(raw)

    @classmethod
    def map_node_type(cls, t: str | None) -> str | None:
        """Map human labels to enum strings expected by API."""
        if t is None:
            return None
        mapping = {
            "Application (SEAL)": "Application (SEAL)",
            "Intelligent Solutions": "Intelligent Solutions",
            "User Tool": "User Tool",
            # common synonyms
            "Application": "Application (SEAL)",
            "UserTool": "User Tool",
        }
        return mapping.get(t, t)

    @classmethod
    def map_tier_type(cls, tier_type: str | None) -> str | None:
        """Map tier types to enum values."""
        tier_mapping = {"Tier 1": "Tier 1", "Non Tier 1": "Non Tier 1"}
        return tier_mapping.get(tier_type) if tier_type else None

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame, client: Fusion | None = None) -> list[Report]:
        """
        Create a list of Report objects from a DataFrame, applying permanent column mapping.

        Args:
            data (pandas.DataFrame): DataFrame containing report data.

        Returns:
            list[Report]: List of Report objects.
        """
        df_df = data.rename(columns=Report.COLUMN_MAPPING)  # type: ignore[attr-defined]
        df_df = df_df.replace([np.nan, np.inf, -np.inf], None)
        df_df = df_df.where(df_df.notna(), None)

        reports: list[Report] = []
        for _, row in df_df.iterrows():
            report_data: dict[str, Any] = row.to_dict()

            # business_domain now plain string; if only domain_name available, use it
            if report_data.get("business_domain") is None:
                report_data["business_domain"] = Report._str_or_none(report_data.pop("domain_name", None))

            # Build owner/publisher nodes (using agreed CSV column names)
            def build_node(report_data: dict[str, Any], name_key: str, type_key: str) -> dict[str, Any] | None:
                name_val = Report._str_or_none(report_data.pop(name_key, None))
                type_val = cls.map_node_type(report_data.pop(type_key, None))
                if name_val or type_val:
                    return {"name": name_val or "", "type": type_val or ""}
                return None

            # UPDATED to use your new headers
            publisher_node = build_node(report_data, "publisherNode_name", "publisherNode_type")
            owner_node = build_node(report_data, "ownerNode_name", "ownerNode_type")

            # Backward-compat: old single-node → owner_node
            if owner_node is None and ("data_node_name" in report_data or "data_node_type" in report_data):
                name_str = Report._str_or_none(report_data.pop("data_node_name", None))
                type_str = cls.map_node_type(report_data.pop("data_node_type", None))
                owner_node = {"name": name_str or "", "type": type_str or ""}

            # Funnel flat "publisher_node_identifier" into nested publisher_node
            pub_ident = Report._str_or_none(report_data.pop("publisher_node_identifier", None))
            if pub_ident:
                if publisher_node is None:
                    publisher_node = {"name": "", "type": "", "publisher_node_identifier": pub_ident}
                else:
                    publisher_node["publisher_node_identifier"] = pub_ident

            report_data["owner_node"] = owner_node
            report_data["publisher_node"] = publisher_node

            # Convert boolean fields from Yes/No if needed
            for key in ("is_bcbs239_program", "mnpi_indicator", "regulatory_related"):
                val = report_data.get(key)
                if isinstance(val, str):
                    low = val.strip().lower()
                    if low == "yes":
                        report_data[key] = True
                    elif low == "no":
                        report_data[key] = False

            # Map tier designation
            report_data["tier_designation"] = cls.map_tier_type(report_data.get("tier_designation"))

            # Filter to valid fields
            valid_fields = {f.name for f in fields(cls)}
            report_data = {k: v for k, v in report_data.items() if k in valid_fields}

            report_obj = cls(**report_data)
            report_obj.client = client

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


    # -------------------- validation --------------------

    def validate(self) -> None:
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
        # Enforce nodes as per schema
        if not (self.owner_node and self.owner_node.get("name") and self.owner_node.get("type")):
            missing.append("owner_node.name/type")
        if not (self.publisher_node and self.publisher_node.get("name") and self.publisher_node.get("type")):
            missing.append("publisher_node.name/type")

        if missing:
            raise ValueError(f"Missing required fields in Report: {', '.join(missing)}")

    # -------------------- API calls --------------------

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
        with suppress(Exception):
            self.id = resp.json().get("id", self.id)
        return resp if return_resp_obj else None


    def update(
        self,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Update this report using HTTP PUT with the current object state.

        Args:
            client (Fusion, optional): A Fusion client object. Defaults to the instance's _client.
            return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

        Returns:
            requests.Response | None: The response object from the API call if return_resp_obj is True, otherwise None.
        """
        client = self._use_client(client)
        if not self.id:
            raise ValueError("Report ID is required on the object (set self.id before update()).")
        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/reports/{self.id}"
        resp: requests.Response = client.session.put(url, json=self.to_dict())
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def patch(
        self,
        fields_to_update: dict[str, Any],
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Partially update this report using HTTP PATCH.

        Args:
            fields_to_update (dict[str, Any]): Dictionary of fields to update. Keys may be snake_case or camelCase.
            client (Fusion, optional): A Fusion client object. Defaults to the instance's _client.
            return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

        Returns:
            requests.Response | None: The response object from the API call if return_resp_obj is True, otherwise None.
        """
        client = self._use_client(client)
        if not self.id:
            raise ValueError("Report ID is required on the object (set self.id before patch()).")

        payload: dict[str, Any] = {}
        for k, v in fields_to_update.items():
            # boolean specials
            if k in ("is_bcbs239_program", "isBCBS239Program"):
                payload["isBCBS239Program"] = v
                continue
            if k in ("regulatory_related", "regulatoryRelated"):
                payload["regulatoryRelated"] = v
                continue

            # flat "publisher_node_identifier" -> nested publisherNode.publisherNodeIdentifier
            if k in ("publisher_node_identifier", "publisherNodeIdentifier"):
                node = payload.setdefault("publisherNode", {})
                node["publisherNodeIdentifier"] = v
                continue

            # nested publisher_node dict (snake or camel inside)
            if k in ("publisher_node", "publisherNode"):
                if not isinstance(v, dict):
                    raise TypeError("publisher_node must be a dict with keys like name/type/publisher_node_identifier")
                node = payload.setdefault("publisherNode", {})
                for nk, nv in v.items():
                    node[snake_to_camel(camel_to_snake(nk))] = nv
                continue

            # default top-level
            payload[snake_to_camel(camel_to_snake(k))] = v

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

    # -------------------- terms linking --------------------

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
            f"{client._get_new_root_url()}/api/corelineage-service/v1/reports/{report_id}/reportElements/businessTerms"
        )
        resp = client.session.post(url, json=mappings)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None


# ----------- permanent column mapping for DataFrame/CSV ingestion -----------

Report.COLUMN_MAPPING = {  # type: ignore[attr-defined]
    # Core
    "Report/Process Name": "title",
    "Report/Process Description": "description",
    "Frequency": "frequency",
    "Category": "category",
    "Sub Category": "sub_category",
    "Regulatory Designated": "regulatory_related",

    # Business domain (new agreed header)
    "businessDomain": "business_domain",
    # Optional backward-compat (old sheet header)
    "CDO Office": "business_domain",

    # Old single-node columns (backward-compat → owner_node)
    "Application ID": "data_node_name",
    "Application Type": "data_node_type",

    # NEW: publisherNode identifier CSV header → nested field
    "publisherNode_publisherNodeIdentifier": "publisher_node_identifier",

    # NEW: source system CSV header → field
    "sourceSystem": "source_system",

    # Optionals / flags
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
        """Create all Report objects in this collection."""
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
        """Load Reports from CSV using the bound client."""
        return Reports.from_csv(file_path, client=self.client)

    def from_dataframe(self, df: pd.DataFrame) -> Reports:  # type: ignore[override]
        """Load Reports from DataFrame using the bound client."""
        return Reports.from_dataframe(df, client=self.client)

    def from_object(self, source: pd.DataFrame | list[dict[str, Any]] | str) -> Reports:  # type: ignore[override]
        """Load Reports from object using the bound client."""
        return Reports.from_object(source, client=self.client)
