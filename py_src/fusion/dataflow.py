from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
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

    # Required
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

    @classmethod
    def from_dict(cls: type[Dataflow], data: dict[str, Any]) -> Dataflow:
        """Create a Dataflow object from a dictionary."""

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
                    converted[key] = [convert_keys(i) if isinstance(i, dict) else i for i in v]  # type: ignore[assignment]
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
    def _from_series(cls: type[Dataflow], series: pd.Series) -> Dataflow:
        """Instantiate a single Dataflow from a pandas Series."""
        return cls.from_dict(series.to_dict())

    @classmethod
    def _from_dict(cls: type[Dataflow], data: dict[str, Any]) -> Dataflow:
        """Instantiate a single Dataflow from a dict ."""
        return cls.from_dict(data)

    @classmethod
    def from_dataframe(cls, frame: pd.DataFrame, client: Fusion | None = None) -> list[Dataflow]:
        """Instantiate multiple Dataflow objects from a DataFrame (row-wise)."""
        frame = frame.replace([np.nan, np.inf, -np.inf], None).where(frame.notna(), None)
        results: list[Dataflow] = []
        for _, row in frame.iterrows():
            try:
                obj = cls._from_series(row)
                obj.client = client
                obj.validate()
                results.append(obj)
            except ValueError as e:  # noqa: PERF203
                logger.warning("Skipping invalid row: %s", e)
        return results

    def from_object(self, dataflow_source: Dataflow | dict[str, Any] | str | pd.Series) -> Dataflow:
        """Instantiate a single Dataflow from a Dataflow/dict/JSON-object/Series."""
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
                raise ValueError("Unsupported string input — must be JSON object string")
        else:
            raise TypeError(f"Could not resolve the object provided: {type(dataflow_source).__name__}")

        obj.client = self._client
        return obj

    def validate(self) -> None:
        """Validate that required fields exist."""
        required_fields = ["provider_node", "consumer_node"]
        missing = [f for f in required_fields if getattr(self, f, None) in [None, ""]]
        if missing:
            raise ValueError(f"Missing required fields in Dataflow: {', '.join(missing)}")

    def to_dict(
        self,
        *,
        drop_none: bool = True,
        exclude: set[str] | None = None,
    ) -> dict[str, Any]:
        """Convert Dataflow object into a JSON-serializable dictionary."""
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
        except (ValueError, TypeError):
            pass

        return resp if return_resp_obj else None

    def delete(
        self,
        dataflow_id: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Delete a dataflow by ID."""
        client = self._use_client(client)
        target_id = dataflow_id or getattr(self, "id", None)
        if not target_id:
            raise ValueError("Dataflow ID is required for delete (pass dataflow_id= or set self.id).")

        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/lineage/dataflows/{target_id}"
        resp: requests.Response = client.session.delete(url)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def update(
        self,
        dataflow_id: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Full replace (PUT) excluding provider/consumer nodes from the payload."""
        client = self._use_client(client)
        target_id = dataflow_id or self.id
        if not target_id:
            raise ValueError("Dataflow ID is required for update (pass dataflow_id= or call create() first).")

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
        dataflow_id: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Partial update (PATCH)."""
        client = self._use_client(client)
        target_id = dataflow_id or self.id
        if not target_id:
            raise ValueError("Dataflow ID is required for update_fields (pass dataflow_id= or call create() first).")

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
