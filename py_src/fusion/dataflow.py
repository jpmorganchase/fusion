from __future__ import annotations

from dataclasses import dataclass, field

from fusion.dataset import Dataset


@dataclass
class InputDataFlow(Dataset):
    producerApplicationId: dict[str, str] | None = None
    consumerApplicationId: list[dict[str, str]] | None = None
    flowDetails: dict[str, str] | None = None
    type_: str = field(default="Flow", init=False)

    def __repr__(self: InputDataFlow) -> str:
        """Return an object representation of the InputDataFlow object.

        Returns:
            str: Object representation of the dataset.

        """
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"InputDataFlow(\n" + ",\n ".join(f"{k}={v!r}" for k, v in attrs.items()) + "\n)"


@dataclass
class OutputDataFlow(Dataset):
    producerApplicationId: list[dict[str, str]] | None = None
    consumerApplicationId: dict[str, str] | None = None
    flowDetails: dict[str, str] | None = None
    type_: str = field(default="Flow", init=False)

    def __repr__(self: OutputDataFlow) -> str:
        """Return an object representation of the OutputDataFlow object.

        Returns:
            str: Object representation of the dataset.

        """
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"OutputDataFlow(\n" + ",\n ".join(f"{k}={v!r}" for k, v in attrs.items()) + "\n)"
