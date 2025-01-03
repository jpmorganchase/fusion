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
        
    def __repr__(self: Report) -> str:
        """Return an object representation of the Report object.

        Returns:
            str: Object representation of the dataset.

        """
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"Report(\n" + ",\n ".join(f"{k}={v!r}" for k, v in attrs.items()) + "\n)"

