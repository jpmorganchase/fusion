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
    """Fusion Dataflow class for managing dataflow metadata in the Fusion system.

    Attributes (snake_case on the model; camelCase emitted to the API):
        provider_node (dict[str, str] | None):
            Provider/source node. Required for create().
            Keys: ``name``, ``type``.

        consumer_node (dict[str, str] | None):
            Consumer/target node. Required for create().
            Keys: ``name``, ``type``.

        description (str | None, optional):
            Purpose/summary of the flow. If present, must not be blank.

        id (str | None, optional):
            Server-assigned identifier. Must be set for ``update()``, ``update_fields()``, and ``delete()``.

        transport_type (str | None, optional):
            Transport type (e.g., API, Batch).

        frequency (str | None, optional):
            Flow cadence.

        start_time (str | None, optional):
            Scheduled start time (e.g., ``HH:mm:ss`` or ISO-8601).

        end_time (str | None, optional):
            Scheduled end time.

        source_system (dict[str, Any] | None, optional):
            Optional source system metadata.

        datasets (list[dict[str, Any]], optional):
            Datasets involved in the data flow. Defaults to empty list.

        connection_type (str | None, required for ``create()``):
            Connection type for the dataflow.

        _client (Fusion | None):
            Fusion client (injected by Fusion factory).
    """

    provider_node: dict[str, str] | None = None
    consumer_node: dict[str, str] | None = None
    description: str | None = None
    id: str | None = None
    transport_type: str | None = None
    frequency: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    source_system: dict[str, Any] | None = None
    datasets: list[dict[str, Any]] = field(default_factory=list)
    connection_type: str | None = None

    _client: Fusion | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        """Normalize strings/empties across the instance after initialization."""
        def norm_val(v: Any) -> Any:
            if isinstance(v, str):
                s = tidy_string(v)
                return None if s == "" else s
            return v

        def norm_tree(o: Any) -> Any:
            if isinstance(o, dict):
                return {k: norm_tree(v) for k, v in o.items()}
            if isinstance(o, list):
                return [norm_tree(i) for i in o]
            return norm_val(o)

        for f in fields(self):
            setattr(self, f.name, norm_tree(getattr(self, f.name)))


    def __getattr__(self, name: str) -> Any:
        """Allow camelCase attribute access (e.g., providerNode) for snake_case fields."""
        snake = camel_to_snake(name)
        if snake in self.__dict__:
            return self.__dict__[snake]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "client":
            object.__setattr__(self, name, value)
        else:
            snake = camel_to_snake(name)
            self.__dict__[snake] = value


    @property
    def client(self) -> Fusion | None:
        """Return the client."""
        return self._client

    @client.setter
    def client(self, client: Fusion | None) -> None:
        """Set the client for the Dataflow. Set automatically if instantiated from a Fusion object.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> flow = fusion.dataflow(
            ...     provider_node={"name": "CRM_DB", "type": "Database"},
            ...     consumer_node={"name": "DWH", "type": "Database"},
            ... )
            >>> flow.client = fusion
        """
        self._client = client

    def _use_client(self, client: Fusion | None) -> Fusion:
        """Resolve the client or raise if missing."""
        res = self._client if client is None else client
        if res is None:
            raise ValueError("A Fusion client object is required.")
        return res

    # --- constructors ---

    @classmethod
    def from_dict(cls: type[Dataflow], data: dict[str, Any]) -> Dataflow:
        """Instantiate a Dataflow object from a dictionary (accepts snake or camel keys).

        Returns:
            Dataflow: The constructed object.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> flow = fusion.dataflow().from_object({
            ...     "providerNode": {"name": "CRM_DB", "type": "Database"},
            ...     "consumerNode": {"name": "DWH", "type": "Database"},
            ...     "connectionType": "Consumes From"
            ... })
        """
        # Map incoming keys to snake_case, keep only known dataclass fields
        keys = {f.name for f in fields(cls)}
        mapped = {camel_to_snake(k): v for k, v in data.items()}
        filtered = {k: v for k, v in mapped.items() if k in keys}
        obj = cls(**filtered)
        return obj

    @classmethod
    def _from_series(cls: type[Dataflow], series: pd.Series) -> Dataflow:  # type: ignore[type-arg]
        """Instantiate a single Dataflow from a pandas Series."""
        return cls.from_dict(series.to_dict())

    @classmethod
    def _from_dict(cls: type[Dataflow], data: dict[str, Any]) -> Dataflow:
        """Instantiate a single Dataflow from a dict."""
        return cls.from_dict(data)

    @classmethod
    def from_dataframe(cls, frame: pd.DataFrame, client: Fusion | None = None) -> list[Dataflow]:
        """Instantiate multiple Dataflow objects from a DataFrame (row-wise).

        NaN/inf values are converted to None before object creation. Rows that fail validation are skipped with a log.
        """
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

    def from_object(self, dataflow_source: Dataflow | dict[str, Any] | str | pd.Series) -> Dataflow:  # type: ignore[type-arg]
        """Instantiate a Dataflow from a Dataflow, dict, JSON-object string, or pandas Series.

        Note: CSV input is not supported here.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> flow = fusion.dataflow().from_object('{"providerNode":{"name":"A","type":"DB"},
            "consumerNode":{"name":"B","type":"DB"},"connectionType":"Consumes From"}')
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
            if s.startswith("{"):
                obj = self._from_dict(json.loads(s))
            else:
                raise ValueError("Unsupported string input — must be a JSON object string")
        else:
            raise TypeError(f"Could not resolve the object provided: {type(dataflow_source).__name__}")

        obj.client = self._client
        return obj

    # --- validation ---

    def validate(self) -> None:
        """Validate that required fields exist."""
        required_fields = ["provider_node", "consumer_node"]
        missing = [f for f in required_fields if getattr(self, f, None) in [None, ""]]
        if missing:
            raise ValueError(f"Missing required fields in Dataflow: {', '.join(missing)}")

    def _validate_nodes_for_create(self) -> None:
        """Ensure provider/consumer nodes are present with non-blank name and type for create()."""
        for attr in ("provider_node", "consumer_node"):
            node = getattr(self, attr, None)
            if not isinstance(node, dict):
                raise ValueError(f"{attr} must be a dict with 'name' and 'type' for create().")
            if not node.get("name") or not node.get("type"):
                raise ValueError(f"{attr} requires non-empty 'name' and 'type' for create().")
        if not self.connection_type:
            raise ValueError("connection_type is required for create().")

    # --- serialization ---

    def to_dict(
        self,
        *,
        drop_none: bool = True,
        exclude: set[str] | None = None,
    ) -> dict[str, Any]:
        """Convert the Dataflow instance to a dictionary (camelCase keys for API).

        Returns:
            dict[str, Any]: Dataflow metadata as a dictionary ready for API calls.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> flow = fusion.dataflow(
            ...     provider_node={"name": "CRM_DB", "type": "Database"},
            ...     consumer_node={"name": "DWH", "type": "Database"},
            ...     connection_type="Consumes From",
            ... )
            >>> flow_dict = flow.to_dict()
        """
        out: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if exclude and k in exclude:
                continue
            if drop_none and (v is None or (isinstance(v, str) and v.strip() == "")):
                continue
            out[snake_to_camel(k)] = v
        return out

    # --- API calls ---

    def create(
        self,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Create the dataflow via API.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> flow = fusion.dataflow(
            ...     provider_node={"name": "CRM_DB", "type": "Database"},
            ...     consumer_node={"name": "DWH", "type": "Database"},
            ...     connection_type="Consumes From",
            ... )
            >>> flow.create()
        """
        client = self._use_client(client)
        self._validate_nodes_for_create()

        payload = self.to_dict(drop_none=True, exclude={"id"})
        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/lineage/dataflows"
        resp: requests.Response = client.session.post(url, json=payload)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def update(
        self,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Full update using id.

        Examples:
            >>> flow = fusion.dataflow(id="abc-123")
            >>> flow.description = "Updated"
            >>> flow.update()
        """
        client = self._use_client(client)
        if not self.id:
            raise ValueError("Dataflow ID is required on the object (set self.id before update()).")

        payload = self.to_dict(
            drop_none=True,
            exclude={"id", "provider_node", "consumer_node"},
        )
        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/lineage/dataflows/{self.id}"
        resp: requests.Response = client.session.put(url, json=payload)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def update_fields(
        self,
        changes: dict[str, Any],
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Partial update using id.

        Examples:
            >>> flow = fusion.dataflow(id="abc-123")
            >>> flow.update_fields({"frequency": "WEEKLY"})
        """
        client = self._use_client(client)
        if not self.id:
            raise ValueError("Dataflow ID is required on the object (set self.id before update_fields()).")

        forbidden = {"provider_node", "consumer_node"}
        used = forbidden.intersection(changes.keys())
        if used:
            raise ValueError(
                f"Cannot update {sorted(used)} via PATCH; provider/consumer nodes are immutable for updates."
            )

        # Normalize strings/empties like __post_init__
        def norm_val(v: Any) -> Any:
            if isinstance(v, str):
                s = tidy_string(v)
                return None if s == "" else s
            return v

        def norm_tree(o: Any) -> Any:
            if isinstance(o, dict):
                return {k: norm_tree(v) for k, v in o.items()}
            if isinstance(o, list):
                return [norm_tree(i) for i in o]
            return norm_val(o)

        # Accept snake or camel in 'changes'; send camel to API
        snake_changes = {camel_to_snake(k): v for k, v in changes.items()}
        patch_body = {snake_to_camel(k): norm_tree(v) for k, v in snake_changes.items()}

        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/lineage/dataflows/{self.id}"
        resp: requests.Response = client.session.patch(url, json=patch_body)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def delete(
        self,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Delete this dataflow using self.id.

        Examples:
            >>> flow = fusion.dataflow(id="abc-123")
            >>> flow.delete()
        """
        client = self._use_client(client)
        if not self.id:
            raise ValueError("Dataflow ID is required on the object (set self.id before delete()).")

        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/lineage/dataflows/{self.id}"
        resp: requests.Response = client.session.delete(url)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None
