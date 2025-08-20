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
class Dataflow(metaclass=CamelCaseMeta):
    id: str | None = None  
    providerNode: dict[str, str]
    consumerNode: dict[str, str]

    description: str | None = None
    alternativeId: dict[str, Any] | None = None
    transportType: str | None = None
    frequency: str | None = None
    startTime: str | None = None
    endTime: str | None = None
    boundarySets: list[dict[str, Any]] = field(default_factory=list)
    dataAssets: list[dict[str, Any]] = field(default_factory=list)

    _client: Fusion | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
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
    def from_dict(cls: type[Dataflow], data: dict[str, Any]) -> Dataflow:
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
        required_fields = ["provider_node", "consumer_node"]
        missing = [f for f in required_fields if getattr(self, f, None) in [None, ""]]
        if missing:
            raise ValueError(f"Missing required fields in Dataflow: {', '.join(missing)}")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            out[snake_to_camel(k)] = v
        return out

    def create(
        self,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        client = self._use_client(client)
        data = self.to_dict()

        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/dataflows"
        resp: requests.Response = client.session.post(url, json=data)
        requests_raise_for_status(resp)

        # optional: capture id from response if present
        try:
            new_id = resp.json().get("id")
            if new_id:
                self.id = new_id
        except Exception:
            pass

        return resp if return_resp_obj else None


    def delete(
        self,
        id: str | None = None,                     # ← allow passing an id directly
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Delete a dataflow by ID. If id is not provided, uses self.id."""
        client = self._use_client(client)

        target_id = id or getattr(self, "id", None)
        if not target_id:
            raise ValueError("Dataflow ID is required for delete (pass id= or set self.id).")

        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/dataflows/{target_id}"
        resp: requests.Response = client.session.delete(url)
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None



    def put(self, client: Fusion | None = None, return_resp_obj: bool = False) -> requests.Response | None:
        """Replace a dataflow entirely using PUT."""
        client = self._use_client(client)
        if not hasattr(self, "id") or not self.id:
            raise ValueError("Dataflow ID is required for update.")


        url = f"{client._get_new_root_url()}/api/corelineage-service/v1/dataflows/{self.id}"
        
        resp = client.session.put(url, json=self.to_dict())
        requests_raise_for_status(resp)
        return resp if return_resp_obj else None



class Dataflows:
    def __init__(self, dataflows: list[Dataflow] | None = None) -> None:
        self.dataflows = dataflows or []
        self._client: Fusion | None = None

    def __getitem__(self, index: int) -> Dataflow:
        return self.dataflows[index]

    def __iter__(self) -> Iterator[Dataflow]:
        return iter(self.dataflows)

    def __len__(self) -> int:
        return len(self.dataflows)

    @property
    def client(self) -> Fusion | None:
        return self._client

    @client.setter
    def client(self, client: Fusion | None) -> None:
        self._client = client
        for df in self.dataflows:
            df.client = client

    @classmethod
    def from_csv(cls, file_path: str, client: Fusion | None = None) -> Dataflows:
        df = pd.read_csv(file_path)
        return cls.from_dataframe(df, client=client)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, client: Fusion | None = None) -> Dataflows:
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
        for df in self.dataflows:
            df.create()

    @classmethod
    def from_object(
        cls,
        source: pd.DataFrame | list[dict[str, Any]] | str,
        client: Fusion | None = None,
    ) -> Dataflows:
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
    def __init__(self, client: Fusion) -> None:
        super().__init__([])
        self.client = client

    def from_csv(self, file_path: str) -> Dataflows:  # type: ignore[override]
        return Dataflows.from_csv(file_path, client=self.client)

    def from_dataframe(self, df: pd.DataFrame) -> Dataflows:  # type: ignore[override]
        return Dataflows.from_dataframe(df, client=self.client)

    def from_object(self, source: pd.DataFrame | list[dict[str, Any]] | str) -> Dataflows:  # type: ignore[override]
        return Dataflows.from_object(source, client=self.client)
