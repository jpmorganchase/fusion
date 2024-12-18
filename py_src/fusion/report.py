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
    report: dict[str, str] | None = None
    type_: str = field(default="Report", init=False)

    def __post_init__(self: Report) -> None:
        super().__post_init__()
        if not self.report:
            raise ValueError("The 'report' attribute is required and cannot be empty.")
        
    def add_registered_attribute(
        self: Report,
        attribute_identifier: str,
        is_kde: bool,
        application_id: str | dict[str, str],  # noqa: ARG002
        catalog: str | None = None,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> requests.Response | None:
        client = self._use_client(client)
        catalog = client._use_catalog(catalog)
        dataset = self.identifier

        url = f"{client.root_url}catalogs/{catalog}/datasets/{dataset}?attributes/{attribute_identifier}/registration"

        data = {
            "isCriticalDataElement": is_kde,
        }

        resp: requests.Response = client.session.post(url, json=data)
        requests_raise_for_status(resp)

        return resp if return_resp_obj else None
