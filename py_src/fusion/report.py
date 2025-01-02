from __future__ import annotations

from dataclasses import dataclass

from fusion.dataset import Dataset


@dataclass
class Report(Dataset):
    report: dict[str, str] | None = None
    type_: str | None = "Report"

    def __post_init__(self: Report) -> None:
        super().__post_init__()
        if not self.report:
            raise ValueError("The 'report' attribute is required and cannot be empty.")

