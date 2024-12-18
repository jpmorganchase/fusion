from __future__ import annotations

from dataclasses import dataclass, field

from fusion.dataset import Dataset


@dataclass
class InputDataFlow(Dataset):
    producerApplicationId: dict[str, str] | None = None
    consumerApplicationId: list[dict[str, str]] | None = None
    flowDetails: dict[str, str] | None = None
    type_: str = field(default="Flow", init=False)


@dataclass
class OutputDataFlow(Dataset):
    producerApplicationId: list[dict[str, str]] | None = None
    consumerApplicationId: dict[str, str] | None = None
    flowDetails: dict[str, str] | None = None
    type_: str = field(default="Flow", init=False)
