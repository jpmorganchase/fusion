from __future__ import annotations

from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Any, get_args

if TYPE_CHECKING:
    import pytest


class TestFusionMainModule:
    @staticmethod
    def _run_main_module() -> None:
        main_path = Path(__file__).resolve().parents[1] / "py_src" / "fusion" / "__main__.py"
        namespace = {"__file__": str(main_path), "__name__": "__main__"}
        exec(compile(main_path.read_text(), str(main_path), "exec"), namespace)  # noqa: S102

    def test_invokes_requested_method(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import fusion

        captured: dict[str, Any] = {}

        class FakeFusion:
            def __init__(
                self,
                root_url: str | None = None,
                credentials: str | None = None,
                download_folder: str | None = None,
                log_level: str | None = None,
                log_path: str | None = None,
            ) -> None:
                captured["init"] = {
                    "root_url": root_url,
                    "credentials": credentials,
                    "download_folder": download_folder,
                    "log_level": log_level,
                    "log_path": log_path,
                }

            def publish(self, dataset: str | None = None, overwrite: bool = False, dry_run: bool = True) -> None:
                captured["method"] = {
                    "dataset": dataset,
                    "overwrite": overwrite,
                    "dry_run": dry_run,
                }

        monkeypatch.setattr(fusion, "Fusion", FakeFusion)
        monkeypatch.setattr(
            "sys.argv",
            [
                "fusion",
                "--root_url",
                "https://example.test/api/v1/",
                "--credentials",
                "client_credentials.json",
                "--download_folder",
                "downloads-dir",
                "--log_level",
                "20",
                "--log_path",
                "logs",
                "--method",
                "publish",
                "--dataset",
                "prices",
                "--overwrite",
                "True",
                "--dry_run",
                "False",
            ],
        )

        self._run_main_module()

        assert captured["init"] == {
            "root_url": "https://example.test/api/v1/",
            "credentials": "client_credentials.json",
            "download_folder": "downloads-dir",
            "log_level": "20",
            "log_path": "logs",
        }
        assert captured["method"] == {
            "dataset": "prices",
            "overwrite": True,
            "dry_run": False,
        }

    def test_without_method_only_initializes_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import fusion

        captured: dict[str, Any] = {}

        class FakeFusion:
            def __init__(
                self,
                root_url: str | None = None,
                credentials: str | None = None,
                download_folder: str | None = None,
                log_level: str | None = None,
                log_path: str | None = None,
            ) -> None:
                captured["init"] = {
                    "root_url": root_url,
                    "credentials": credentials,
                    "download_folder": download_folder,
                    "log_level": log_level,
                    "log_path": log_path,
                }

            def publish(self, dataset: str | None = None) -> None:
                captured["method"] = dataset

        monkeypatch.setattr(fusion, "Fusion", FakeFusion)
        monkeypatch.setattr("sys.argv", ["fusion", "--credentials", "client_credentials.json"])

        self._run_main_module()

        assert captured["init"] == {
            "root_url": None,
            "credentials": "client_credentials.json",
            "download_folder": None,
            "log_level": None,
            "log_path": None,
        }
        assert "method" not in captured


class TestCompatibilityShims:
    def test_legacy_fusion_module_exports_fusion_credentials(self) -> None:
        shim_path = Path(__file__).resolve().parents[1] / "py_src" / "fusion" / "_fusion.py"
        namespace: dict[str, Any] = {"__file__": str(shim_path), "__name__": "fusion._fusion_py"}

        exec(compile(shim_path.read_text(), str(shim_path), "exec"), namespace)  # noqa: S102

        assert namespace["FusionCredentials"].__name__ == "FusionCredentials"
        assert namespace["__all__"] == ["FusionCredentials"]

    def test_types_aliases_are_available(self) -> None:
        from fusion.types import PyArrowFilterT, WorkerQueueT

        assert WorkerQueueT == Queue[tuple[int, int, int]]
        assert get_args(PyArrowFilterT) == (list[tuple[Any]], list[list[tuple[Any]]])
