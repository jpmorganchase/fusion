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
    """Dataflow class for maintaining data flow metadata.

    Attributes:
        producer_application_id (dict[str, str] | None): The producer application ID (upstream application
            producing the flow).
        consumer_application_id (list[dict[str, str]] | dict[str, str] | None): The consumer application ID (downstream
            application, consuming the flow).
        flow_details (dict[str, str] | None): The flow details. Specifies input versus output flow.
        type_ (str | None): The type of dataset. Defaults to "Flow".

    """

    producer_application_id: dict[str, str] | None = None
    consumer_application_id: list[dict[str, str]] | dict[str, str] | None = None
    flow_details: dict[str, str] | None = None
    type_: str | None = "Flow"

    def __post_init__(self: DataFlow) -> None:
        """Format the Data Flow object."""
        self.consumer_application_id = (
            [self.consumer_application_id]
            if isinstance(self.consumer_application_id, dict)
            else self.consumer_application_id
        )
        super().__post_init__()

    def add_registered_attribute(
        self: DataFlow,
        attribute_identifier: str,
        catalog: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Add a registered attribute to the Data Flow.

        Args:
            attribute_identifier (str): Attribute identifier.
            catalog (str | None, optional): Catalog identifier. Defaults to 'common'.
            client (Fusion, optional): A Fusion client object. Defaults to the instance's _client.
                If instantiated from a Fusion object, then the client is set automatically.
            return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

        Returns:
            requests.Response | None: The response object from the API call if return_resp_obj is True, otherwise None.
        """
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)
        dataset = self.identifier

        url = f"{client.root_url}catalogs/{catalog}/datasets/{dataset}/attributes/{attribute_identifier}/registration"

        data = {
            "isCriticalDataElement": False,
        }

        resp: requests.Response = client.session.post(url, json=data)
        requests_raise_for_status(resp)

        return resp if return_resp_obj else None


@dataclass
class InputDataFlow(DataFlow):
    """InputDataFlow class for maintaining input data flow metadata."""

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
    """OutputDataFlow class for maintaining output data flow metadata."""

    flow_details: dict[str, str] | None = field(default_factory=lambda: {"flowDirection": "Output"})

    def __repr__(self: OutputDataFlow) -> str:
        """Return an object representation of the OutputDataFlow object.

        Returns:
            str: Object representation of the dataset.

        """
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"OutputDataFlow(\n" + ",\n ".join(f"{k}={v!r}" for k, v in attrs.items()) + "\n)"
