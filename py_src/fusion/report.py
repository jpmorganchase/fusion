from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fusion.dataset import Dataset
from fusion.utils import requests_raise_for_status

if TYPE_CHECKING:
    import requests

    from fusion.fusion import Fusion


@dataclass
class Report(Dataset):
    """Fusion Report class for managing regulatory reporting metadata.

    Attributes:
        report (dict[str, str]): The report metadata. Specifies the tier of the report.
        type_ (str): The dataset type. Defaults to "Report", which is the required value for creating a Report object.

    """

    report: dict[str, str] | None = field(default_factory=lambda: {"tier": ""})
    type_: str | None = "Report"

    def __repr__(self: Report) -> str:
        """Return an object representation of the Report object.

        Returns:
            str: Object representation of the dataset.

        """
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"Report(\n" + ",\n ".join(f"{k}={v!r}" for k, v in attrs.items()) + "\n)"

    def add_registered_attribute(
        self: Report,
        attribute_identifier: str,
        is_key_data_element: bool,
        catalog: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        """Add a registered attribute to the Report.

        Args:
            attribute_identifier (str): Attribute identifier.
            is_key_data_element (bool): Key Data Element flag. An attribute can be proposed as a key data element when
                it is linked to a report. This property is specific to the relationship between the attribute and the
                report.
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
            "isCriticalDataElement": is_key_data_element,
        }

        resp: requests.Response = client.session.post(url, json=data)
        requests_raise_for_status(resp)

        return resp if return_resp_obj else None
