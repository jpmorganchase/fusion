from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fusion.dataset import Dataset
from fusion.utils import requests_raise_for_status

if TYPE_CHECKING:
    import requests

    from fusion.fusion import Fusion


@dataclass
class DataFlow(Dataset):
    producer_application_id: dict[str, str] | None = None
    consumer_application_id: list[dict[str, str]] | dict[str, str] | None = None
    flow_details: dict[str, str] | None = None
    type_: str | None = "Flow"

    def __post_init__(self: DataFlow) -> None:
        self.consumer_application_id = (
            [self.consumer_application_id]
            if isinstance(self.consumer_application_id, dict)
            else self.consumer_application_id
        )
        super().__post_init__()

    def add_registered_attribute(
        self: DataFlow,
        attribute_identifier: str,
        is_kde: bool,
        catalog: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)
        dataset = self.identifier

        url = f"{client.root_url}catalogs/{catalog}/datasets/{dataset}/attributes/{attribute_identifier}/registration"

        data = {
            "isCriticalDataElement": is_kde,
        }

        resp: requests.Response = client.session.post(url, json=data)
        requests_raise_for_status(resp)

        return resp if return_resp_obj else None


@dataclass
class InputDataFlow(DataFlow):
    flow_details: dict[str, str] | None = field(default_factory=lambda: {"flowDirection": "Input"})

    def __repr__(self: InputDataFlow) -> str:
        """Return an object representation of the InputDataFlow object.

        Returns:
            str: Object representation of the dataset.

        """
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"InputDataFlow(\n" + ",\n ".join(f"{k}={v!r}" for k, v in attrs.items()) + "\n)"


@dataclass
class OutputDataFlow(DataFlow):
    flow_details: dict[str, str] | None = field(default_factory=lambda: {"flowDirection": "Output"})

    def __repr__(self: OutputDataFlow) -> str:
        """Return an object representation of the OutputDataFlow object.

        Returns:
            str: Object representation of the dataset.

        """
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"OutputDataFlow(\n" + ",\n ".join(f"{k}={v!r}" for k, v in attrs.items()) + "\n)"
