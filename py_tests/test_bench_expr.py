from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest


class TestPy:
    def __init__(
        self,
        i64_1: int | None,
        i64_2: int | None,
        str_1: str | None,
        str_2: str | None,
        str_3: str | None,
        str_4: str | None,
    ) -> None:
        self.i64_1 = i64_1
        self.i64_2 = i64_2
        self.str_1 = str_1
        self.str_2 = str_2
        self.str_3 = str_3
        self.str_4 = str_4
        dummy_val = 1234
        if i64_1 == dummy_val:
            self.i64_1 = dummy_val**dummy_val

    @classmethod
    def factory(
        cls: type[TestPy],
        i64_1: int | None,
        i64_2: int | None,
        str_1: str | None,
        str_2: str | None,
        str_3: str | None,
        str_4: str | None,
    ) -> TestPy:
        return cls(i64_1, i64_2, str_1, str_2, str_3, str_4)

    @classmethod
    def factory_with_file(  # noqa: PLR0913
        cls: type[TestPy],
        file_path: Path,
        _i64_1: int | None,
        i64_2: int | None,
        str_1: str | None,
        str_2: str | None,
        str_3: str | None,
        str_4: str | None,
    ) -> TestPy:
        with Path(file_path).open() as f:
            data = f.read()
            x = len(data)
            return cls(x, i64_2, str_1, str_2, str_3, str_4)

    @classmethod
    def factory_with_deser(cls: type[TestPy], file_path: Path) -> TestPy:
        with Path(file_path).open() as f:
            data = json.load(f)

            i64_1 = int(os.environ["TEST_I64_1_VAR"]) if "i64_1" not in data else data["i64_1"]
            i64_2 = int(os.environ["TEST_I64_2_VAR"]) if "i64_2" not in data else data["i64_2"]
            str_1 = os.environ["TEST_STR_1_VAR"] if "str_1" not in data else data["str_1"]
            str_2 = os.environ["TEST_STR_2_VAR"] if "str_2" not in data else data["str_2"]
            str_3 = os.environ["TEST_STR_3_VAR"] if "str_3" not in data else data["str_3"]
            str_4 = os.environ["TEST_STR_4_VAR"] if "str_4" not in data else data["str_4"]

            return cls(i64_1, i64_2, str_1, str_2, str_3, str_4)


ITERS = 10
ROUNDS = 5_000


@pytest.fixture
def example_dict() -> dict[str, Any]:
    return {"i64_1": 42, "i64_2": 69, "str_1": "ME", "str_2": "YOU", "str_3": "HIM", "str_4": "HER"}


@pytest.fixture
def example_file(tmp_path: Path, example_dict: dict[str, Any]) -> Path:
    in_file = tmp_path / "example.json"
    with Path(in_file).open("w") as f:
        json.dump(example_dict, f)
    return in_file


@pytest.fixture
def empty_file(tmp_path: Path) -> Path:
    in_file = tmp_path / "example.json"
    with Path(in_file).open("w") as f:
        json.dump({}, f)
    return in_file
