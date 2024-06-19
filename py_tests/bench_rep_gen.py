import json
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader


def calculate_difference(current: dict, previous: dict) -> dict:  # type: ignore
    """
    Calculate the difference between current and previous benchmark results.
    """
    difference = {}
    for key in ["min", "max", "mean", "stddev", "median", "iqr", "ops"]:
        current_value = current["stats"].get(key, 0) * 1e6 if key != "ops" else current["stats"].get(key, 0) / 1e3
        previous_value = previous["stats"].get(key, 0) * 1e6 if key != "ops" else previous["stats"].get(key, 0) / 1e3
        difference[key] = current_value - previous_value
    return difference


def generate_benchmark_html(current_json_file: Path, previous_json_file: Optional[Path], html_file: Path) -> None:
    with current_json_file.open() as f:
        current_data = json.load(f)
    if previous_json_file is None:
        previous_data: Any = {"benchmarks": []}
    else:
        with previous_json_file.open() as f:
            previous_data = json.load(f)

    current_benchmarks = current_data["benchmarks"]
    previous_benchmarks = {b["name"]: b for b in previous_data["benchmarks"]}

    for benchmark in current_benchmarks:
        benchmark["difference"] = calculate_difference(
            benchmark, previous_benchmarks.get(benchmark["name"], {"stats": {}})
        )

    compare_groups = {}  # type: ignore
    for benchmark in current_benchmarks:
        compare_group = benchmark["extra_info"].get("compare_group")
        if compare_group:
            if compare_group not in compare_groups:
                compare_groups[compare_group] = []
            compare_groups[compare_group].append(benchmark)

    for benchmarks in compare_groups.values():
        base_benchmark = benchmarks[0]
        for benchmark in benchmarks[1:]:
            benchmark["comparison"] = {
                "mean_diff": (benchmark["stats"]["mean"] - base_benchmark["stats"]["mean"]) * 1e6,
                "median_diff": (benchmark["stats"]["median"] - base_benchmark["stats"]["median"]) * 1e6,
                "ops_diff": (benchmark["stats"]["ops"] - base_benchmark["stats"]["ops"]) / 1e3,
            }

    machine_info = current_data["machine_info"]
    commit_info = current_data["commit_info"]
    datetime_info = current_data["datetime"]
    version = current_data["version"]

    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template("py_tests/bench_template.html")

    html_content = template.render(
        benchmarks=current_benchmarks,
        compare_groups=compare_groups,
        machine_info=machine_info,
        commit_info=commit_info,
        datetime_info=datetime_info,
        version=version,
    )

    with html_file.open("w") as f:
        f.write(html_content)

    print(f"Generated HTML report at {html_file}")  # noqa: T201
