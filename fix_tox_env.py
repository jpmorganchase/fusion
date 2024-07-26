import subprocess
from pathlib import Path

py_versions = ["3.9", "3.10", "3.11", "3.12"]


def fetch_toolchains() -> None:
    for version in py_versions:
        subprocess.run(["rye", "toolchain", "fetch", version], check=True)


def get_cpython_paths() -> list[str]:
    # Inspect the environment to find cpython installations managed by rye
    home = Path.home()
    rye_path = Path.home() / ".rye/py"
    return [str(p).replace(str(home), "{env:HOME}") for p in rye_path.glob("cpython@*/lib")]


def generate_tox_env(cpython_paths: list[str]) -> tuple[str, str]:
    ld_library_path = ":".join(cpython_paths) + ":$LD_LIBRARY_PATH"
    rustflags = " ".join([f"-L {path}" for path in cpython_paths])
    return ld_library_path, rustflags


def update_tox_file(ld_library_path: str, rustflags: str) -> None:
    fetch_toolchains()
    tox_file = "tox.ini"
    with Path(tox_file).open() as file:
        lines = file.readlines()

    with Path(tox_file).open("w") as file:
        for line in lines:
            if line.strip().startswith("LD_LIBRARY_PATH ="):
                file.write(f"    LD_LIBRARY_PATH = {ld_library_path}\n")
            elif line.strip().startswith("RUSTFLAGS ="):
                file.write(f"    RUSTFLAGS = {rustflags}\n")
            else:
                file.write(line)


def main() -> None:
    cpython_paths = get_cpython_paths()
    ld_library_path, rustflags = generate_tox_env(cpython_paths)
    update_tox_file(ld_library_path, rustflags)
    print(  # noqa: T201
        f"""tox.ini has been updated with the new LD_LIBRARY_PATH and RUSTFLAGS
LD_LIBRARY_PATH: {ld_library_path}
RUSTFLAGS: {rustflags}"""
    )


if __name__ == "__main__":
    main()
