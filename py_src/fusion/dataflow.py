from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .utils import (
    CamelCaseMeta,
    camel_to_snake,
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
class Dataflow(metaclass=CamelCaseMeta):
    """Model for a lineage dataflow edge (provider → consumer).

    Notes:
        - Uses a metaclass to support camelCase attribute access while storing
          values internally as snake_case.
        - `providerNode` and `consumerNode` identify the topology and are
          immutable for updates (cannot be changed via PUT/PATCH).
        - When calling `create()`, if the API returns an `id` in the JSON
          response, it will be stored on the instance as `self.id`.

    Attributes:
        providerNode: Source node details (API-facing camelCase). Expected to
            include at least "name" and "dataNodeType".
        consumerNode: Destination node details (API-facing camelCase).
        description: Optional description of the dataflow.
        id: Server-assigned identifier (set after successful `create()` if
            present in response).
        alternativeId: Optional alternate identifiers.
        transportType: Optional transport mechanism (validated against an
            allowed set; normalized to uppercase). May be None/empty.
        frequency: Optional frequency string (validated against an allowed set;
            normalized to uppercase). May be None/empty.
        startTime: Optional start time string (as provided).
        endTime: Optional end time string (as provided).
        boundarySets: Optional list of boundary set objects.
        dataAssets: Optional list of data asset objects.
        _client: Attached Fusion client used for API calls.
    """

    # Required (API identity/wiring)
    providerNode: dict[str, str]
    consumerNode: dict[str, str]

    # Optional fields
    description: str | None = None
    id: str | None = None
    alternativeId: dict[str, Any] | None = None
    transportType: str | None = None
    frequency: str | None = None
    startTime: str | None = None
    endTime: str | None = None
    boundarySets: list[dict[str, Any]] = field(default_factory=list)
    dataAssets: list[dict[str, Any]] = field(default_factory=list)

    _client: Fusion | None = field(init=False, repr=False, compare=False, default=None)

    # ---- Allowed values (exact spellings as provided) ----
    ALLOWED_TRANSPORT_TYPES = {
        "SYNCHRONOUS MESSAGING",
        "FILE TRANSFER",
        "API",
        "ASYCHRONOUS MESSAGING",  # spelling per your request
    }
    ALLOWED_FREQUENCIES = {
        "BI-WEEKLY",
        "WEEKLY",
        "SEMI-ANNUALY",  # spelling per your request
        "QUARTERLY",
        "ANNUALLY",
        "DAILY",
        "ADHOC",          # <-- comma was missing previously
        "INTRA-DAY",
        "MONTHLY",
        "TWICE-WEEKLY",
        "BI-MONTHLY",
    }
    # ------------------------------------------------------

    def __post_init__(self) -> None:
        """Normalize and validate fields immediately after dataclass init.

        - Tidies description (trims/normalizes).
        - Uppercases and validates `transport_type` and `frequency` if present
          and non-empty. Empty strings are treated as None.

        Raises:
            ValueError: If `transport_type` or `frequency` is not in their
                respective allowed sets when provided.
        """
        # tidy description
        self.description = tidy_string(self.description or "")

        # normalize + validate transportType (allow None/empty)
        if self.transport_type is not None:
            tt_raw = tidy_string(self.transport_type)
            if tt_raw == "":
                self.transport_type = None
            else:
                tt = tt_raw.upper()
                if tt not in self.ALLOWED_TRANSPORT_TYPES:
                    raise ValueError(
                        f"Invalid transportType '{self.transport_type}'. "
                        f"Allowed: {sorted(self.ALLOWED_TRANSPORT_TYPES)}"
                    )
                self.transport_type = tt

        # normalize + validate frequency (allow None/empty)
        if self.frequency is not None:
            fq_raw = tidy_string(self.frequency)
            if fq_raw == "":
                self.frequency = None
            else:
                fq = fq_raw.upper()
                if fq not in self.ALLOWED_FREQUENCIES:
                    raise ValueError(
                        f"Invalid frequency '{self.frequency}'. "
                        f"Allowed: {sorted(self.ALLOWED_FREQUENCIES)}"
                    )
                self.frequency = fq

    def __getattr__(self, name: str) -> Any:
        """Provide camelCase attribute access backed by snake_case storage.

        Args:
            name: Attribute name as accessed by the caller.

        Returns:
            The attribute value if present; otherwise None.
        """
        snake_name = camel_to_snake(name)
        return self.__dict__.get(snake_name, None)

    def __setattr__(self, name: str, value: Any) -> None:
        """Store attributes as snake_case, except for the `client` property.

        Args:
            name: Attribute name.
            value: Value to assign.
        """
        if name == "client":
            object.__setattr__(self, name, value)
        else:
            snake_name = camel_to_snake(name)
            self.__dict__[snake_name] = value

    @property
    def client(self) -> Fusion | None:
        """Return the attached Fusion client (if any)."""
        return self._client

    @client.setter
    def client(self, client: Fusion | None) -> None:
        """Attach a Fusion client to this instance.

        Args:
            client: The Fusion client to attach, or None to clear.
        """
        self._client = client

    def _use_client(self, client: Fusion | None) -> Fusion:
        """Resolve the client to use (argument overrides attached client).

        Args:
            client: Optional client passed by the caller.

        Returns:
            The resolved Fusion client.

        Raises:
            ValueError: If no client is available.
        """
        res = self._client if client is None else client
        if res is None:
            raise ValueError("A Fusion client object is required.")
        return res

    @classmethod
    def from_dict(cls: type[Dataflow], data: dict[str, Any]) -> Dataflow:
        """Construct a Dataflow from a dict (camel/snake keys both accepted).

        This:
            - Normalizes empty strings to None.
            - Recursively converts keys to snake_case.
            - Filters unknown fields.
            - Applies `__post_init__` to normalize/validate enums.

        Args:
            data: Source mapping (e.g., parsed JSON).

        Returns:
            A Dataflow instance populated from the mapping.
        """
        def normalize_value(val: Any) -> Any:
            if isinstance(val, str) and val.strip() == "":
                return None
            return val

        def convert_keys(d: dict[str, Any]) -> dict[str, Any]:
            converted = {}
            for k, v in d.items():
                key = camel_to_snake(k)
                if isinstance(v, dict):
                    converted[key] = convert_keys(v)
                elif isinstance(v, list):
                    converted[key] = [convert_keys(i) if isinstance(i, dict) else i for i in v]
                else:
                    converted[key] = normalize_value(v)
            return converted

        converted_data = convert_keys(data)
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in converted_data.items() if k in valid_fields}

        dataflow = cls.__new__(cls)
        for field_obj in fields(cls):
            setattr(dataflow, field_obj.name, filtered_data.get(field_obj.name, None))

        dataflow.__post_init__()
        return dataflow

    def validate(self) -> None:
        """Validate required fields and enum constraints.

        Raises:
            ValueError: If required fields are missing, or enum fields are
                outside their allowed sets.
        """
        required_fields = ["provider_node", "consumer_node"]
        missing = [f for f in required_fields if getattr(self, f, None) in [None, ""]]
        if missing:
            raise ValueError(f"Missing required fields in Dataflow: {', '.join(missing)}")

        # re-validate enums in case fields were changed post-init
        if self.transport_type is not None:
            tt_raw = tidy_string(str(self.transport_type))
            if tt_raw != "":
                if tt_raw.upper() not in self.ALLOWED_TRANSPORT_TYPES:
                    raise ValueError(
                        f"Invalid transportType '{self.transport_type}'. "
                        f"Allowed: {sorted(self.ALLOWED_TRANSPORT_TYPES)}"
                    )
            else:
                self.transport_type = None

        if self.frequency is not None:
            fq_raw = tidy_string(str(self.frequency))
            if fq_raw != "":
                if fq_raw.upper() not in self.ALLOWED_FREQUENCIES:
                    raise ValueError(
                        f"Invalid frequency '{self.frequency}'. "
                        f"Allowed: {sorted(self.ALLOWED_FREQUENCIES)}"
                    )
            else:
                self.frequency = None

    def to_dict(
        self,
        *,
        drop_none: bool = True,
        exclude: set[str] | None = None,
    ) -> dict[str, Any]:
        """Serialize the instance to an API-ready camelCase dict.

        Args:
            drop_none: If True, omit keys whose values are None.
            exclude: Optional set of snake_case field names to exclude
                (e.g., {"id"} or {"id", "provider_node"}).

        Returns:
            A dict with camelCase keys suitable for API calls.
        """
        out: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if exclude and k in exclude:
                continue
            if drop_none and v is None:
                continue
            out[snake_to_camel(k)] = v
        return out

    def create(
        self,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Create this dataflow via the API.

        Excludes `id` from the payload. If the response JSON contains an `id`,
        it will be stored on the instance.

        Args:
            client: Optional Fusion client; if omitted, uses the attached client.
            return_resp_obj: If True, return the `requests.Response`.

        Returns:
            The response object if `return_resp_obj` is True; otherwise None.

        Raises:
            ValueError: If no client is available.
            requests.HTTPError: If the API returns an error status code.
        """
        client = self._use_client(client)

        payload = self.to_dict(drop_none=True, exclude={"id"})
        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/lineage/dataflows"
        resp: requests.Response = client.session.post(url, json=payload)
        requests_raise_for_status(resp)

        # Capture server-assigned id if present
        try:
            data = resp.json()
            if isinstance(data, dict) and "id" in data:
                self.id = data["id"]
        except Exception:
            pass  # some endpoints return no body

        return resp if return_resp_obj else None

    def delete(
        self,
        id: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Delete a dataflow by ID.

        If `id` is not provided, `self.id` will be used if available.

        Args:
            id: The identifier of the dataflow to delete.
            client: Optional Fusion client; if omitted, uses the attached client.
            return_resp_obj: If True, return the `requests.Response`.

        Returns:
            The response object if `return_resp_obj` is True; otherwise None.

        Raises:
            ValueError: If neither `id` nor `self.id` is available.
            requests.HTTPError: If the API returns an error status code.
        """
        client = self._use_client(client)

        target_id = id or getattr(self, "id", None)
        if not target_id:
            raise ValueError("Dataflow ID is required for delete (pass id= or set self.id).")

        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/lineage/dataflows/{target_id}"
        resp: requests.Response = client.session.delete(url)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def update(
        self,
        id: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Full replace (PUT) of all mutable fields.

        Excludes `id`, `provider_node`, and `consumer_node` from the payload
        (those are immutable/identity fields).

        Args:
            id: The identifier to update; if omitted, uses `self.id`.
            client: Optional Fusion client; if omitted, uses the attached client.
            return_resp_obj: If True, return the `requests.Response`.

        Returns:
            The response object if `return_resp_obj` is True; otherwise None.

        Raises:
            ValueError: If no id is available (neither argument nor `self.id`).
            requests.HTTPError: If the API returns an error status code.
        """
        client = self._use_client(client)
        target_id = id or self.id
        if not target_id:
            raise ValueError("Dataflow ID is required for update (pass id= or call create() first).")

        payload = self.to_dict(
            drop_none=True,
            exclude={"id", "provider_node", "consumer_node"},
        )

        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/lineage/dataflows/{target_id}"
        resp: requests.Response = client.session.put(url, json=payload)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def update_fields(
        self,
        changes: dict[str, Any],
        id: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Partial update (PATCH) of selected fields.

        Provider/consumer nodes are immutable and cannot be changed via PATCH.

        Args:
            changes: Mapping of fields to update. Keys may be snake_case or
                camelCase; they will be converted to camelCase for the API.
                Values of "" are treated as None for enum fields.
            id: The identifier to patch; if omitted, uses `self.id`.
            client: Optional Fusion client; if omitted, uses the attached client.
            return_resp_obj: If True, return the `requests.Response`.

        Returns:
            The response object if `return_resp_obj` is True; otherwise None.

        Raises:
            ValueError: If attempting to change provider/consumer, or if no id
                is available (neither argument nor `self.id`).
            requests.HTTPError: If the API returns an error status code.
        """
        client = self._use_client(client)
        target_id = id or self.id
        if not target_id:
            raise ValueError("Dataflow ID is required for update_fields (pass id= or call create() first).")

        # Normalize keys to snake to check constraints, then back to camel for API.
        forbidden = {"provider_node", "consumer_node"}
        normalized = {camel_to_snake(k): v for k, v in changes.items()}

        used = forbidden.intersection(normalized.keys())
        if used:
            raise ValueError(
                f"Cannot update {sorted(used)} via PATCH; provider/consumer nodes are immutable for updates."
            )

        # Optional normalization of enums on patch (allow None/empty)
        if "transport_type" in normalized:
            val = normalized["transport_type"]
            if val is None:
                pass
            else:
                tt_raw = tidy_string(str(val))
                if tt_raw == "":
                    normalized["transport_type"] = None
                else:
                    tt = tt_raw.upper()
                    if tt not in self.ALLOWED_TRANSPORT_TYPES:
                        raise ValueError(
                            f"Invalid transportType '{val}'. "
                            f"Allowed: {sorted(self.ALLOWED_TRANSPORT_TYPES)}"
                        )
                    normalized["transport_type"] = tt

        if "frequency" in normalized:
            val = normalized["frequency"]
            if val is None:
                pass
            else:
                fq_raw = tidy_string(str(val))
                if fq_raw == "":
                    normalized["frequency"] = None
                else:
                    fq = fq_raw.upper()
                    if fq not in self.ALLOWED_FREQUENCIES:
                        raise ValueError(
                            f"Invalid frequency '{val}'. "
                            f"Allowed: {sorted(self.ALLOWED_FREQUENCIES)}"
                        )
                    normalized["frequency"] = fq

        patch_body = {snake_to_camel(k): v for k, v in normalized.items()}

        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/lineage/dataflows/{target_id}"
        resp: requests.Response = client.session.patch(url, json=patch_body)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None


class Dataflows:
    """Collection wrapper for multiple `Dataflow` instances.

    Provides loading helpers from CSV/DataFrame/JSON-like sources, applies
    per-row validation, and exposes simple bulk operations.
    """

    def __init__(self, dataflows: list[Dataflow] | None = None) -> None:
        """Initialize a collection.

        Args:
            dataflows: Optional list of pre-built `Dataflow` instances.
        """
        self.dataflows = dataflows or []
        self._client: Fusion | None = None

    def __getitem__(self, index: int) -> Dataflow:
        """Return the item at `index`."""
        return self.dataflows[index]

    def __iter__(self) -> Iterator[Dataflow]:
        """Iterate over contained `Dataflow` instances."""
        return iter(self.dataflows)

    def __len__(self) -> int:
        """Return the number of items in the collection."""
        return len(self.dataflows)

    @property
    def client(self) -> Fusion | None:
        """Return the attached Fusion client (if any)."""
        return self._client

    @client.setter
    def client(self, client: Fusion | None) -> None:
        """Attach a Fusion client and propagate it to all children.

        Args:
            client: The Fusion client to attach, or None to clear.
        """
        self._client = client
        for df in self.dataflows:
            df.client = client

    @classmethod
    def from_csv(cls, file_path: str, client: Fusion | None = None) -> Dataflows:
        """Load dataflows from a CSV file.

        Args:
            file_path: Path to a CSV file.
            client: Optional Fusion client to attach to each instance.

        Returns:
            A `Dataflows` collection populated from the CSV.
        """
        df = pd.read_csv(file_path)
        return cls.from_dataframe(df, client=client)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, client: Fusion | None = None) -> Dataflows:
        """Load dataflows from a pandas DataFrame.

        Replaces NaN/±inf with None, row-wise builds `Dataflow` via `from_dict`,
        validates each, and skips invalid rows with a warning.

        Args:
            df: Source DataFrame.
            client: Optional Fusion client to attach to each instance.

        Returns:
            A `Dataflows` collection containing valid rows.
        """
        df = df.replace([np.nan, np.inf, -np.inf], None).where(df.notna(), None)
        dataflow_objs = []

        for _, row in df.iterrows():
            try:
                obj = Dataflow.from_dict(row.to_dict())
                obj.client = client
                obj.validate()
                dataflow_objs.append(obj)
            except ValueError as e:
                logger.warning(f"Skipping invalid row: {e}")

        result = cls(dataflow_objs)
        result.client = client
        return result

    def create_all(self) -> None:
        """POST all contained dataflows to the API (sequentially)."""
        for df in self.dataflows:
            df.create()

    @classmethod
    def from_object(
        cls,
        source: pd.DataFrame | list[dict[str, Any]] | str,
        client: Fusion | None = None,
    ) -> Dataflows:
        """Load dataflows from a DataFrame, list of dicts, or JSON/CSV string.

        Args:
            source: Either a DataFrame, a list of dictionaries, a JSON-array
                string, or a CSV file path (ending with `.csv`).
            client: Optional Fusion client to attach to each instance.

        Returns:
            A `Dataflows` collection.

        Raises:
            ValueError: If the string input is neither a JSON array nor a
                valid `.csv` path.
            TypeError: If `source` is not one of the supported types.
        """
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


class DataflowsWrapper(Dataflows):
    """Client-bound `Dataflows` facade with convenience constructors."""

    def __init__(self, client: Fusion) -> None:
        """Initialize an empty collection bound to `client`.

        Args:
            client: Fusion client to attach.
        """
        super().__init__([])
        self.client = client

    def from_csv(self, file_path: str) -> Dataflows:  # type: ignore[override]
        """Proxy `Dataflows.from_csv`, auto-attaching this wrapper's client."""
        return Dataflows.from_csv(file_path, client=self.client)

    def from_dataframe(self, df: pd.DataFrame) -> Dataflows:  # type: ignore[override]
        """Proxy `Dataflows.from_dataframe`, auto-attaching this wrapper's client."""
        return Dataflows.from_dataframe(df, client=self.client)

    def from_object(self, source: pd.DataFrame | list[dict[str, Any]] | str) -> Dataflows:  # type: ignore[override]
        """Proxy `Dataflows.from_object`, auto-attaching this wrapper's client."""
        return Dataflows.from_object(source, client=self.client)
