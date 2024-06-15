import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def generate_benchmark_html(json_file: Path, html_file: Path) -> None:
    with json_file.open() as f:
        data = json.load(f)

    benchmarks = data["benchmarks"]
    machine_info = data["machine_info"]
    commit_info = data["commit_info"]
    datetime_info = data["datetime"]
    version = data["version"]

    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template("py_tests/bench_template.html")

    html_content = template.render(
        benchmarks=benchmarks,
        machine_info=machine_info,
        commit_info=commit_info,
        datetime_info=datetime_info,
        version=version,
    )

    with html_file.open("w") as f:
        f.write(html_content)

    print(f"Generated HTML report at {html_file}")  # noqa: T201
