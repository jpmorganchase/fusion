from __future__ import annotations

from dataclasses import dataclass, field

from fusion.dataset import Dataset
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
        application_id: str | dict[str, str],
        catalog: str | None = None,
        is_critical_data_element: bool = True,
        client: Fusion | None = None,
        return_resp_obj: bool = False,
    ) -> None:
        
