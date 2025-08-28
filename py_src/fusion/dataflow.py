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
        providerNode (dict[str, str]):
            Defines the provider/source node of the data flow.
            Required when creating a data flow.
            Expected keys:
              - ``name`` (str): Provider node name.
              - ``dataNodeType`` (str): Provider node type.

        consumerNode (dict[str, str]):
            Defines the consumer/target node of the data flow.
            Required when creating a data flow and must be distinct from the provider node.
            Expected keys:
              - ``name`` (str): Consumer node name.
              - ``dataNodeType`` (str): Consumer node type.

        description (str | None, optional):
            Purpose/summary of the data flow. If this field is present, it must not be blank.
            (API range reference: length 1..65535). Defaults to ``None``.

        id (str | None, optional):
            Server-assigned unique identifier of the data flow. Must be set on the object for
            ``update()``, ``update_fields()``, and ``delete()``. Defaults to ``None``.

        alternativeId (dict[str, Any] | None, optional):
            Alternative/secondary identifier object for the data flow. Based on the API schema,
            this may include:
              - ``value`` (str): Up to 255 characters.
              - ``domain`` (str): A domain string.
              - ``isSor`` (bool): Whether this is a System-of-Record id.
            Defaults to ``None``.

        transportType (str | None, optional):
            Transport type of the data flow. API allows values like
            ``"SYNCHRONOUS MESSAGING"``, ``"FILE TRANSFER"``, ``"API"``, ``"ASYNCHRONOUS MESSAGING"``.
            Defaults to ``None``.

        frequency (str | None, optional):
            Frequency of the data flow. API allows values such as
            ``"BI-WEEKLY"``, ``"WEEKLY"``, ``"SEMI-ANNUALLY"``, ``"QUARTERLY"``, ``"ANNUALLY"``,
            ``"DAILY"``, ``"ADHOC"``, ``"INTRA-DAY"``, ``"MONTHLY"``, ``"TWICE-WEEKLY"``, ``"BI-MONTHLY"``.
            Defaults to ``None``.

        startTime (str | None, optional):
            Scheduled start time (ISO 8601 / time-of-day formats like ``HH:mm:ss`` or ``HH:mm:ssZ``).
             Defaults to ``None``.

        endTime (str | None, optional):
            Scheduled end time (ISO 8601 / time-of-day formats like ``HH:mm:ss`` or ``HH:mm:ssZ``).
             Defaults to ``None``.

        dataAssets (list[dict[str, Any]], optional):
            List of data asset objects involved in the data flow (up to API-defined limits). Defaults to empty list.

        boundarySets (list[dict[str, Any]], optional):
            Boundary set objects for the data flow; items are stored as provided. Defaults to empty list.
    """

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
        """Return the client."""
        return self._client

    @client.setter
    def client(self, client: Fusion | None) -> None:
        """Set the client for the Dataflow. Set automatically if instantiated from a Fusion object.

        Args:
            client (Any): Fusion client object.

        Examples:
            >>> from fusion import Fusion
            >>> fusion = Fusion()
            >>> flow = fusion.dataflow(
            ...     provider_node={"name": "CRM_DB", "dataNodeType": "Database"},
            ...     consumer_node={"name": "DWH", "dataNodeType": "Database"},
            ... )
            >>> flow.client = fusion
        """
        self._client = client

    def _use_client(self, client: Fusion | None) -> Fusion:
        """Determine client.

        Returns:
            Fusion: The resolved Fusion client to use.

        Raises:
            ValueError: If neither a provided client nor a bound client is available.
        """
        res = self._client if client is None else client
        if res is None:
            raise ValueError("A Fusion client object is required.")
        return res

    @classmethod
    def from_dict(cls: type[Dataflow], data: dict[str, Any]) -> Dataflow:
        """Create a Dataflow object from a dictionary.

        Accepts camelCase or snake_case keys and normalizes empty strings to None.

        Args:
            data (dict[str, Any]): Dataflow fields, potentially nested, in camelCase or snake_case.

        Returns:
            Dataflow: A populated Dataflow instance.

        Examples:
            >>> df = Dataflow.from_dict({
            ...     "providerNode": {"name": "CRM_DB", "dataNodeType": "Database"},
            ...     "consumerNode": {"name": "DWH", "dataNodeType": "Database"},
            ...     "description": "CRM to DWH load",
            ... })
        """

        def normalize_value(val: Any) -> Any:
            if isinstance(val, str) and val.strip() == "":
                return None
            return val

        def convert_keys(d: dict[str, Any]) -> dict[str, Any]:
            converted: dict[str, Any] = {}
            for k, v in d.items():
                key = camel_to_snake(k)
                if isinstance(v, dict):
                    converted[key] = convert_keys(v)
                elif isinstance(v, list):
                    converted[key] = [
                        convert_keys(i) if isinstance(i, dict) else i for i in v
                    ]
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
    def _from_series(cls: type[Dataflow], series: pd.Series) -> Dataflow:  # type: ignore[type-arg]
        """Instantiate a single Dataflow from a pandas Series.

        Args:
            series (pd.Series): One row with dataflow fields.

        Returns:
            Dataflow: A populated Dataflow instance.

        Examples:
            >>> row = pd.Series({
            ...     "providerNode": {"name": "S3", "dataNodeType": "Storage"},
            ...     "consumerNode": {"name": "Analytics", "dataNodeType": "Dashboard"},
            ... })
            >>> Dataflow._from_series(row)
        """
        return cls.from_dict(series.to_dict())

    @classmethod
    def _from_dict(cls: type[Dataflow], data: dict[str, Any]) -> Dataflow:
        """Instantiate a single Dataflow from a dict.

        Args:
            data (dict[str, Any]): Dataflow fields.

        Returns:
            Dataflow: A populated Dataflow instance.
        """
        return cls.from_dict(data)

    @classmethod
    def from_dataframe(cls, frame: pd.DataFrame, client: Fusion | None = None) -> list[Dataflow]:
        """Instantiate multiple Dataflow objects from a DataFrame (row-wise).

        NaN/inf values are converted to None before object creation. Rows that fail validation are skipped with a log.

        Args:
            frame (pd.DataFrame): Tabular data with one dataflow per row.
            client (Fusion | None): Optional Fusion client to bind to each created Dataflow.

        Returns:
            list[Dataflow]: Successfully validated Dataflow objects.

        Examples:
            >>> flows = Dataflow.from_dataframe(df, client=fusion)
            >>> len(flows)
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
        """Instantiate a single Dataflow from a Dataflow, dict, JSON-object string, or pandas Series.

        Note: CSV input is not supported here.

        Args:
            dataflow_source (Dataflow | dict[str, Any] | str | pd.Series): Source to construct a Dataflow from.

        Returns:
            Dataflow: The constructed Dataflow with this instance's client bound.

        Raises:
            ValueError: If a string source is not a JSON object string.
            TypeError: If the source type is unsupported.

        Examples:
            From a dict:
            >>> fusion = Fusion()
            >>> handle = fusion.dataflow(
            ...     provider_node={"name": "TMP", "dataNodeType": "TMP"},
            ...     consumer_node={"name": "TMP", "dataNodeType": "TMP"},
            ... )
            >>> flow = handle.from_object({
            ...     "providerNode": {"name": "CRM_DB", "dataNodeType": "Database"},
            ...     "consumerNode": {"name": "DWH", "dataNodeType": "Database"},
            ...     "description": "CRM to DWH load"
            ... })

            From a Series:
            >>> s = pd.Series({
            ...     "providerNode": {"name": "S3", "dataNodeType": "Storage"},
            ...     "consumerNode": {"name": "Analytics", "dataNodeType": "Dashboard"},
            ... })
            >>> flow = handle.from_object(s)

            From JSON:
            >>> flow = handle.from_object('{"providerNode":{"name":"A","dataNodeType":"DB"},'
            ...                           '"consumerNode":{"name":"B","dataNodeType":"DB"}}')
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
        """Validate that required fields exist.

        Raises:
            ValueError: If required fields are missing.
        """
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
        """Convert Dataflow object into a JSON-serializable dictionary.

        Args:
            drop_none (bool, optional): Exclude None/blank-string values if True. Defaults to True.
            exclude (set[str] | None, optional): Snake_case field names to omit from the output.

        Returns:
            dict[str, Any]: Serialized Dataflow with camelCase keys.

        Examples:
            >>> flow.to_dict()
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

        Args:
            client (Fusion | None, optional): Fusion session. Defaults to the bound client.
            return_resp_obj (bool, optional): If True, return the response object. Defaults to False.

        Returns:
            requests.Response | None: The response object if requested, otherwise None.

        Examples:
            >>> flow = fusion.dataflow(
            ...     provider_node={"name": "CRM_DB", "dataNodeType": "Database"},
            ...     consumer_node={"name": "DWH", "dataNodeType": "Database"},
            ...     description="CRM to DWH",
            ... )
            >>> flow.create()
        """
        client = self._use_client(client)

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
        """Full replace (PUT) using self.id, excluding provider/consumer nodes from the payload.

        Args:
            client (Fusion | None, optional): Fusion session. Defaults to the bound client.
            return_resp_obj (bool, optional): If True, return the response object. Defaults to False.

        Returns:
            requests.Response | None: The response object if requested, otherwise None.

        Raises:
            ValueError: If the object has no `id` set.

        Examples:
            >>> flow = fusion.dataflow(
            ...     provider_node={"name": "A", "dataNodeType": "DB"},
            ...     consumer_node={"name": "B", "dataNodeType": "DB"},
            ... )
            >>> flow.id = "abc-123"
            >>> flow.description = "Updated description"
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
        """Partial update (PATCH) using self.id. Provider/consumer nodes are not allowed.

        Args:
            changes (dict[str, Any]): Fields to update; accepts snake_case or camelCase keys.
            client (Fusion | None, optional): Fusion session. Defaults to the bound client.
            return_resp_obj (bool, optional): If True, return the response object. Defaults to False.

        Returns:
            requests.Response | None: The response object if requested, otherwise None.

        Raises:
            ValueError: If no id is set, or if forbidden fields are provided.

        Examples:
            >>> flow = fusion.dataflow(
            ...     provider_node={"name": "A", "dataNodeType": "DB"},
            ...     consumer_node={"name": "B", "dataNodeType": "DB"},
            ... )
            >>> flow.id = "abc-123"
            >>> flow.update_fields({"description": "Patched"})
        """
        client = self._use_client(client)
        if not self.id:
            raise ValueError("Dataflow ID is required on the object (set self.id before update_fields()).")

        forbidden = {"provider_node", "consumer_node"}
        normalized = {camel_to_snake(k): v for k, v in changes.items()}
        used = forbidden.intersection(normalized.keys())
        if used:
            raise ValueError(
                f"Cannot update {sorted(used)} via PATCH; provider/consumer nodes are immutable for updates."
            )

        patch_body = {snake_to_camel(k): v for k, v in normalized.items()}

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

        Args:
            client (Fusion | None, optional): Fusion session. Defaults to the bound client.
            return_resp_obj (bool, optional): If True, return the response object. Defaults to False.

        Returns:
            requests.Response | None: The response object if requested, otherwise None.

        Raises:
            ValueError: If the object has no `id` set.

        Examples:
            >>> flow = fusion.dataflow(
            ...     provider_node={"name": "A", "dataNodeType": "DB"},
            ...     consumer_node={"name": "B", "dataNodeType": "DB"},
            ... )
            >>> flow.id = "abc-123"
            >>> flow.delete()
        """
        client = self._use_client(client)
        if not self.id:
            raise ValueError("Dataflow ID is required on the object (set self.id before delete()).")

        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/lineage/dataflows/{self.id}"
        resp: requests.Response = client.session.delete(url)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None
