#  noqa
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from jinja2 import Environment, FileSystemLoader
from rich import print as rprint

app = typer.Typer()

PyVerOpt = typer.Option("-p", "--py_vers", help="Comma sep Python versions to run the benchmarks with. Ex 3.9,3.10")


def format_datetime(dt_str: str) -> str:
    dt = datetime.fromisoformat(dt_str)
    return dt.isoformat()


@app.command()
def py_bench(py_vers: Annotated[str, PyVerOpt]) -> None:
    py_vers_list = map(str.strip, py_vers.split(","))

    all_data = {}
    for py_ver in py_vers_list:
        rprint(f"Python version: {py_ver}")
        bench_list = sorted(Path(".benchmarks").glob(f"*{py_ver}*/*.json"))
        prev_json_file, prev_data = None, None
        if len(bench_list) == 0:
            rprint(f"No benchmarks found for Python version {py_ver}")
            continue
        if len(bench_list) > 1:
            prev_json_file = bench_list[-2]
        curr_json_file = bench_list[-1]
        rprint(f"Current JSON file: {curr_json_file}")
        with curr_json_file.open() as f:
            curr_data = json.load(f)

        if prev_json_file:
            with prev_json_file.open() as f:
                prev_data = json.load(f)
            rprint(f"Previous JSON file: {prev_json_file}")

        if prev_data:
            for k, v in curr_data.items():
                if k not in prev_data[k]:
                    prev_data[k] = v
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

        # curr_data from list to a dict keyed on fullname
        curr_map = {b["fullname"]: b for b in curr_data["benchmarks"]}
        prev_map = {b["fullname"]: b for b in prev_data["benchmarks"]} if prev_data else {}

        for fullname, benchmark in prev_map.items():
            if fullname in curr_map and benchmark["options"] == curr_map[fullname]["options"]:
                curr_map[fullname]["prev_stats"] = benchmark["stats"]

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
