from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

try:
    from fusion._fusion import RustTestClass  # type: ignore
except ImportError:
    print("Ensure to build rust lib with --features experiments flag")  # noqa: T201


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


@pytest.mark.experiments
@pytest.mark.benchmark(group="class__init__")
def test_rs_py_cl(benchmark: Any) -> None:
    benchmark.pedantic(TestPy, args=(1, 2, "s1", "s2", "s3", "s4"), iterations=ITERS, rounds=ROUNDS)


@pytest.mark.experiments
@pytest.mark.benchmark(group="class__init__")
def test_rs_py_cl_ts(benchmark: Any) -> None:
    benchmark.pedantic(TestPy, args=(1234, 2, "s1", "s2", "s3", "s4"), iterations=ITERS, rounds=ROUNDS)


@pytest.mark.experiments
@pytest.mark.benchmark(group="class__init__")
def test_rs_rs_cls(benchmark: Any) -> None:
    benchmark.pedantic(RustTestClass, args=(1, 2, "s1", "s2", "s3", "s4"), iterations=ITERS, rounds=ROUNDS)


@pytest.mark.experiments
@pytest.mark.benchmark(group="factory")
def test_rs_py_fac(benchmark: Any) -> None:
    benchmark.pedantic(TestPy.factory, args=(1, 2, "s1", "s2", "s3", "s4"), iterations=ITERS, rounds=ROUNDS)


@pytest.mark.experiments
@pytest.mark.benchmark(group="factory")
def test_rs_rs_fac(benchmark: Any) -> None:
    benchmark.pedantic(RustTestClass.factory, args=(1, 2, "s1", "s2", "s3", "s4"), iterations=ITERS, rounds=ROUNDS)


@pytest.mark.experiments
@pytest.mark.benchmark(group="class_w_file")
def test_rs_py_fac_wf(benchmark: Any, example_file: Path) -> None:
    benchmark.pedantic(
        TestPy.factory_with_file, args=(example_file, 1, 2, "s1", "s2", "s3", "s4"), iterations=ITERS, rounds=ROUNDS
    )


@pytest.mark.experiments
@pytest.mark.benchmark(group="class_w_file")
def test_rs_rs_fac_wf(benchmark: Any, example_file: Path) -> None:
    benchmark.pedantic(
        RustTestClass.factory_with_file,
        args=(example_file, 1, 2, "s1", "s2", "s3", "s4"),
        iterations=ITERS,
        rounds=ROUNDS,
    )


@pytest.mark.experiments
@pytest.mark.benchmark(group="deser")
def test_rs_py_fac_dsr(benchmark: Any, example_file: Path) -> None:
    benchmark.pedantic(TestPy.factory_with_deser, args=(example_file,), iterations=ITERS, rounds=ROUNDS)


@pytest.mark.experiments
@pytest.mark.benchmark(group="deser")
def test_rs_rs_fac_dsr(benchmark: Any, example_file: Path) -> None:
    benchmark.pedantic(RustTestClass.factory_with_file_serde, args=(example_file,), iterations=ITERS, rounds=ROUNDS)


@pytest.mark.experiments
@pytest.mark.benchmark(group="factory_env")
def test_rs_py_fac_env(benchmark: Any, empty_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_I64_1_VAR", "101")
    monkeypatch.setenv("TEST_I64_2_VAR", "102")
    monkeypatch.setenv("TEST_STR_1_VAR", "env_name_1")
    monkeypatch.setenv("TEST_STR_2_VAR", "env_name_2")
    monkeypatch.setenv("TEST_STR_3_VAR", "env_name_3")
    monkeypatch.setenv("TEST_STR_4_VAR", "env_name_4")

    benchmark.pedantic(TestPy.factory_with_deser, args=(empty_file,), iterations=ITERS, rounds=ROUNDS)


@pytest.mark.experiments
@pytest.mark.benchmark(group="factory_env")
def test_rs_rs_fac_env(benchmark: Any, empty_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_I64_1_VAR", "101")
    monkeypatch.setenv("TEST_I64_2_VAR", "102")
    monkeypatch.setenv("TEST_STR_1_VAR", "env_name_1")
    monkeypatch.setenv("TEST_STR_2_VAR", "env_name_2")
    monkeypatch.setenv("TEST_STR_3_VAR", "env_name_3")
    monkeypatch.setenv("TEST_STR_4_VAR", "env_name_4")

    benchmark.pedantic(
        RustTestClass.factory_with_file_serde,
        args=(empty_file,),
        iterations=ITERS,
        rounds=ROUNDS,
    )
