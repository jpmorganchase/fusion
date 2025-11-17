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

    Attributes:
        provider_node (dict[str, str] | None):
            Provider node of the dataflow. It must be distinct from the consumer node. Required for create().
            Keys: ``name``, ``type``.

        consumer_node (dict[str, str] | None):
            Consumer node of the dataflow. It must be distinct from the provider node. Required for create().
            Keys: ``name``, ``type``.

        description (str | None, optional):
            Specifies the purpose of the data movement.

        id (str | None, optional):
            Server-assigned identifier. Must be set for ``update()``, ``update_fields()``, and ``delete()``.

        transport_type (str | None, optional):
            Transport type

        frequency (str | None, optional):
            Frequency of the data flow

        start_time (str | None, optional):
            Scheduled start time of the Dataflow.

        end_time (str | None, optional):
            Scheduled end time of the Dataflow.

        source_system (dict[str, Any] | None, optional):
         Source System of the data flow.

        datasets (list[dict[str, Any]], optional):
            Specifies a list of datasets involved in the data flow, requiring a visibility license for each.
            Maximum limit is of 100 datasets per dataflow.
            An error will be thrown if the list contains duplicate entries. Defaults to empty list.

        connection_type (str | None, required for ``create()``):
            Connection type for the dataflow.

        _client (Fusion | None):
            Fusion client .
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
        """Normalize key fields after initialization."""
        self.description = tidy_string(self.description) if self.description is not None else None
        self.id = tidy_string(self.id) if self.id is not None else None
        self.transport_type = tidy_string(self.transport_type) if self.transport_type is not None else None
        self.frequency = tidy_string(self.frequency) if self.frequency is not None else None
        self.start_time = tidy_string(self.start_time) if self.start_time is not None else None
        self.end_time = tidy_string(self.end_time) if self.end_time is not None else None
        self.connection_type = tidy_string(self.connection_type) if self.connection_type is not None else None

        if isinstance(self.provider_node, dict):
            if isinstance(self.provider_node.get("name"), str):
                self.provider_node["name"] = tidy_string(self.provider_node["name"])
            if isinstance(self.provider_node.get("type"), str):
                self.provider_node["type"] = tidy_string(self.provider_node["type"])

        if isinstance(self.consumer_node, dict):
            if isinstance(self.consumer_node.get("name"), str):
                self.consumer_node["name"] = tidy_string(self.consumer_node["name"])
            if isinstance(self.consumer_node.get("type"), str):
                self.consumer_node["type"] = tidy_string(self.consumer_node["type"])

        if self.datasets is None:
            self.datasets = []
        elif not isinstance(self.datasets, list):
            self.datasets = [self.datasets]

    def __getattr__(self, name: str) -> Any:
        """Allow camelCase attribute access"""
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

    @classmethod
    def _from_dict(cls: type[Dataflow], data: dict[str, Any]) -> Dataflow:
        """Instantiate a Dataflow object from a dictionary (private)."""
        keys = {f.name for f in fields(cls)}
        mapped = {camel_to_snake(k): v for k, v in data.items()}
        filtered = {k: v for k, v in mapped.items() if k in keys}
        obj = cls(**filtered)
        return obj

    @classmethod
    def _from_series(cls: type[Dataflow], series: pd.Series) -> Dataflow:  # type: ignore[type-arg]
        """Instantiate a single Dataflow from a pandas Series."""
        return cls._from_dict(series.to_dict())

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

    def to_dict(
        self,
        *,
        drop_none: bool = True,
        exclude: set[str] | None = None,
    ) -> dict[str, Any]:
        """Convert the Dataflow instance to a dictionary.

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
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Partial update using the object's current state.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> flow = fusion.dataflow(id="abc-123", description="hello")
            >>> flow.update_fields()
        """
        client = self._use_client(client)
        if not self.id:
            raise ValueError("Dataflow ID is required on the object (set self.id before update_fields()).")

        payload = self.to_dict(
            drop_none=True,
            exclude={"id", "provider_node", "consumer_node"},
        )

        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/lineage/dataflows/{self.id}"
        resp: requests.Response = client.session.patch(url, json=payload)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None

    def delete(
        self,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Delete this dataflow using Id.

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
