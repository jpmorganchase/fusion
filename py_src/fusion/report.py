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
        type_ (str): The type of dataset. Defaults to "Report", which is required for creating a Report object.

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

