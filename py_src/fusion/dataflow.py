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
    import requests
    from fusion import Fusion

logger = logging.getLogger(__name__)


@dataclass
class Dataflow(metaclass=CamelCaseMeta):
    """Represents a single Dataflow object with CRUD operations and Dataset-style loaders."""

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

    # -----------------------
    # Lifecycle & attributes
    # -----------------------

    def __post_init__(self) -> None:
        """Normalize description immediately after initialization."""
        self.description = tidy_string(self.description or "")

    def __getattr__(self, name: str) -> Any:
        """Allow camelCase access for snake_case attributes."""
        snake_name = camel_to_snake(name)
        return self.__dict__.get(snake_name, None)

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow camelCase assignment to snake_case attributes."""
        if name == "client":
            object.__setattr__(self, name, value)
        else:
            snake_name = camel_to_snake(name)
            self.__dict__[snake_name] = value

    @property
    def client(self) -> Fusion | None:
        """Fusion client associated with this Dataflow."""
        return self._client

    @client.setter
    def client(self, client: Fusion | None) -> None:
        self._client = client

    def _use_client(self, client: Fusion | None) -> Fusion:
        """Resolve the Fusion client (either provided or bound to the object)."""
        res = self._client if client is None else client
        if res is None:
            raise ValueError("A Fusion client object is required.")
        return res

    # -----------------------
    # Converters / loaders (Dataset-style)
    # -----------------------

    @classmethod
    def from_dict(cls: type["Dataflow"], data: dict[str, Any]) -> "Dataflow":
        """Create a Dataflow object from a dictionary (public constructor).

        Accepts camelCase or snake_case keys.
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

        obj = cls.__new__(cls)
        for field_obj in fields(cls):
            setattr(obj, field_obj.name, filtered_data.get(field_obj.name, None))

        obj.__post_init__()
        return obj

    @classmethod
    def _from_series(cls: type["Dataflow"], series: pd.Series) -> "Dataflow":
        """Instantiate a single Dataflow from a pandas Series (one row)."""
        # Keep the same normalization path via from_dict
        return cls.from_dict(series.to_dict())

    @classmethod
    def _from_dict(cls: type["Dataflow"], data: dict[str, Any]) -> "Dataflow":
        """Instantiate a single Dataflow from a dict (alias to from_dict)."""
        return cls.from_dict(data)

    @classmethod
    def _from_csv(cls: type["Dataflow"], file_path: str, row: int | None = None) -> "Dataflow":
        """Instantiate a single Dataflow from a CSV file (select one row).

        Args:
            file_path: Path to CSV.
            row: Optional 0-based row index to select (defaults to 0).

        Returns:
            Dataflow
        """
        df = pd.read_csv(file_path)
        idx = 0 if row is None else row
        if not (0 <= idx < len(df)):
            raise IndexError(f"Row index {idx} is out of range for CSV with {len(df)} rows.")
        # Replace NaN/±inf like your bulk path did
        s = df.replace([np.nan, np.inf, -np.inf], None).iloc[idx]
        return cls._from_series(s)

    def from_object(
        self,
        dataflow_source: "Dataflow" | dict[str, Any] | str | pd.Series,
        *,
        row: int | None = None,
    ) -> "Dataflow":
        """Instantiate a single Dataflow from a Dataflow/dict/JSON-object/CSV row/Series and bind this client's session.

        Mirrors Dataset.from_object:
          - If `dataflow_source` is a dict → build one Dataflow.
          - If JSON string and looks like a single object ('{...}') → parse that object.
          - If CSV path → pick one row (default first, or row=n).
          - If pandas Series → convert that row.
          - If Dataflow instance → rebind the client and return it.

        Returns:
            Dataflow: Single Dataflow with `client` attached to `self._client`.
        """
        import json

        if isinstance(dataflow_source, Dataflow):
            obj = dataflow_source
        elif isinstance(dataflow_source, dict):
            obj = self._from_dict(dataflow_source)
        elif isinstance(dataflow_source, pd.Series):
            obj = self._from_series(dataflow_source)
        elif isinstance(dataflow_source, str):
            s = dataflow_source.strip()
            if s.startswith("{"):  # JSON object string
                obj = self._from_dict(json.loads(s))
            else:
                # Treat as CSV path and select a single row
                obj = self._from_csv(dataflow_source, row=row)
        else:
            raise TypeError(f"Could not resolve the object provided: {type(dataflow_source).__name__}")

        obj.client = self._client
        return obj

    # -----------------------
    # Validation
    # -----------------------

    def validate(self) -> None:
        """Validate that required fields exist.

        Raises:
            ValueError: If required fields are missing.
        """
        required_fields = ["provider_node", "consumer_node"]
        missing = [f for f in required_fields if getattr(self, f, None) in [None, ""]]
        if missing:
            raise ValueError(f"Missing required fields in Dataflow: {', '.join(missing)}")

    # -----------------------
    # Serialization
    # -----------------------

    def to_dict(
        self,
        *,
        drop_none: bool = True,
        exclude: set[str] | None = None,
    ) -> dict[str, Any]:
        """Convert Dataflow object into a JSON-serializable dictionary.

        Args:
            drop_none: Exclude None values if True.
            exclude: Fields to exclude from output (snake_case names).

        Returns:
            dict: Serialized Dataflow representation with camelCase keys.
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

    # -----------------------
    # CRUD
    # -----------------------

    def create(
        self,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Create the dataflow via API and set its server-assigned ID."""
        client = self._use_client(client)

        payload = self.to_dict(drop_none=True, exclude={"id"})
        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/lineage/dataflows"
        resp: requests.Response = client.session.post(url, json=payload)
        requests_raise_for_status(resp)

        try:
            data = resp.json()
            if isinstance(data, dict) and "id" in data:
                self.id = data["id"]
        except Exception:
            # Some endpoints return no body.
            pass

        return resp if return_resp_obj else None

    def delete(
        self,
        id: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Delete a dataflow by ID.

        If `id` is not provided, uses `self.id`.
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
        """Full replace (PUT) excluding provider/consumer nodes from the payload.

        If `id` is not provided, uses `self.id`.
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
        """Partial update (PATCH).

        Provider/consumer nodes cannot be changed here.
        If `id` is not provided, uses `self.id`.
        """
        client = self._use_client(client)
        target_id = id or self.id
        if not target_id:
            raise ValueError("Dataflow ID is required for update_fields (pass id= or call create() first).")

        forbidden = {"provider_node", "consumer_node"}
        normalized = {camel_to_snake(k): v for k, v in changes.items()}

        used = forbidden.intersection(normalized.keys())
        if used:
            raise ValueError(
                f"Cannot update {sorted(used)} via PATCH; provider/consumer nodes are immutable for updates."
            )

        patch_body = {snake_to_camel(k): v for k, v in normalized.items()}

        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/lineage/dataflows/{target_id}"
        resp: requests.Response = client.session.patch(url, json=patch_body)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None
