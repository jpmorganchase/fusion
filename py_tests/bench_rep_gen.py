#  noqa
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from jinja2 import Environment, FileSystemLoader
from rich import print as rprint

app = typer.Typer()

PyVerOpt = typer.Option("-p", "--py_vers", help="Comma sep Python versions to run the benchmarks with. Ex 3.9,3.10")
JsonDict = dict[str, Any]


def format_datetime(dt_str: str) -> str:
    dt = datetime.fromisoformat(dt_str)
    return dt.isoformat()


def _load_json(path: Path) -> JsonDict:
    with path.open() as f:
        return cast(JsonDict, json.load(f))


def _get_benchmark_files(py_ver: str) -> tuple[Path, Path | None] | None:
    bench_list = sorted(Path(".benchmarks").glob(f"*{py_ver}*/*.json"))
    if not bench_list:
        rprint(f"No benchmarks found for Python version {py_ver}")
        return None

    current_file = bench_list[-1]
    previous_file = bench_list[-2] if len(bench_list) > 1 else None
    return current_file, previous_file


def _merge_previous_metadata(curr_data: JsonDict, prev_data: JsonDict | None) -> None:
    if not prev_data:
        return

    for key, value in curr_data.items():
        if key not in prev_data[key]:
            prev_data[key] = value

    curr_data.update(
        {
            "prev": {
                "machine_info": prev_data["machine_info"],
                "commit_info": prev_data["commit_info"],
                "datetime_info": prev_data["datetime"],
                "version": prev_data["version"],
            }
        }
    )


def _attach_previous_stats(curr_data: JsonDict, prev_data: JsonDict | None) -> None:
    curr_map = {benchmark["fullname"]: benchmark for benchmark in curr_data["benchmarks"]}
    prev_map = {benchmark["fullname"]: benchmark for benchmark in prev_data["benchmarks"]} if prev_data else {}

    for fullname, benchmark in prev_map.items():
        if fullname in curr_map and benchmark["options"] == curr_map[fullname]["options"]:
            curr_map[fullname]["prev_stats"] = benchmark["stats"]


def _collect_benchmark_data(py_ver: str) -> JsonDict | None:
    benchmark_files = _get_benchmark_files(py_ver)
    if benchmark_files is None:
        return None

    curr_json_file, prev_json_file = benchmark_files
    rprint(f"Current JSON file: {curr_json_file}")
    curr_data = _load_json(curr_json_file)

    prev_data = _load_json(prev_json_file) if prev_json_file else None
    if prev_json_file:
        rprint(f"Previous JSON file: {prev_json_file}")

    _merge_previous_metadata(curr_data, prev_data)
    _attach_previous_stats(curr_data, prev_data)
    return curr_data


@app.command()
def py_bench(py_vers: Annotated[str, PyVerOpt]) -> None:
    py_vers_list = map(str.strip, py_vers.split(","))

    all_data = {}
    for py_ver in py_vers_list:
        rprint(f"Python version: {py_ver}")
        curr_data = _collect_benchmark_data(py_ver)
        if curr_data is None:
            continue

        env = Environment(loader=FileSystemLoader("."))
        template = env.get_template("py_tests/bench_template.html")

        rprint(curr_data)
        all_data[py_ver] = curr_data

    with Path("test_rep.json").open("w") as f:
        json.dump(all_data, f, indent=4)

    html_content = template.render(
        all_data=all_data,
    )

    html_file = Path(f".reports/py/py_bench.html")
    with html_file.open("w") as f:
        f.write(html_content)

    rprint(f"Generated HTML report at {html_file}")  # noqa: T201


if __name__ == "__main__":
    app()
