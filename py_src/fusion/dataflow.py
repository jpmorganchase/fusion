from __future__ import annotations

from dataclasses import dataclass, field

from fusion.dataset import Dataset


@dataclass
class InputDataFlow(Dataset):
    producer_application_id: dict[str, str] | None = None
    consumer_application_id: list[dict[str, str]] | None = None
    flow_details: dict[str, str] | None = field(default_factory=lambda: {"flowDirection": "Input"})
    type_: str | None = "Flow"

    def __repr__(self: InputDataFlow) -> str:
        """Return an object representation of the InputDataFlow object.

        Returns:
            str: Object representation of the dataset.

        """
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"InputDataFlow(\n" + ",\n ".join(f"{k}={v!r}" for k, v in attrs.items()) + "\n)"


@dataclass
class OutputDataFlow(Dataset):
    producer_application_id: dict[str, str] | None = None
    consumer_application_id: list[dict[str, str]] | None = None
    flow_details: dict[str, str] | None = field(default_factory=lambda: {"flowDirection": "Output"})
    type_: str | None = "Flow"

    def __repr__(self: OutputDataFlow) -> str:
        """Return an object representation of the OutputDataFlow object.

        Returns:
            str: Object representation of the dataset.

        """
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"OutputDataFlow(\n" + ",\n ".join(f"{k}={v!r}" for k, v in attrs.items()) + "\n)"
