import types
from typing import Any
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from fusion.exceptions import APIResponseError, FileFormatError
from fusion.fusion import Fusion
from fusion.utils import distribution_to_url, path_to_url


def test_fusion_helper_functions_cover_small_branches() -> None:
    date_range_pattern = Fusion._valid_download_date_range()
    assert date_range_pattern.match("2024-01-01:2024-01-31")
    assert not date_range_pattern.match("20240101")

    assert Fusion._normalize_upload_dt_str("20240102") == "20240102"
    assert Fusion._normalize_upload_dt_str("custom") == "custom"

    upload_df = Fusion._build_upload_dataframe(["a.csv"], ["catalogs/c1"], ["a.csv"], include_file_name=True)
    assert list(upload_df.columns) == ["path", "url", "file_name"]

    event = {"id": "evt-1", "metaData": {"source": "api"}}
    assert Fusion._flatten_event_metadata(event)["source"] == "api"

    title_lookup = Fusion._build_dataset_title_lookup(
        [
            {"identifier": "ds1", "title": "Dataset 1", "status": "Available"},
            {"identifier": "ds2", "title": "Dataset 2", "status": "Restricted"},
        ]
    )
    assert title_lookup == {"ds1": "Dataset 1", "ds2": "Access Restricted"}
    assert Fusion._build_lineage_entry("upstream", "common", "ds1", title_lookup) == (
        "upstream",
        "common",
        "Dataset 1",
    )


def test_get_exact_dataset_match_returns_none_without_string_or_on_non_200(fusion_obj: Fusion) -> None:
    assert fusion_obj._get_exact_dataset_match("common", ["prices"], display_all_columns=False) is None

    response = MagicMock(status_code=404)
    with patch.object(fusion_obj.session, "get", return_value=response):
        assert fusion_obj._get_exact_dataset_match("common", "prices", display_all_columns=False) is None


def test_get_exact_dataset_match_and_report_helpers_cover_success_paths(fusion_obj: Fusion) -> None:
    dataset_response = MagicMock(status_code=200)
    dataset_response.json.return_value = {
        "identifier": "prices",
        "title": "Prices",
        "containerType": "Snapshot",
        "region": ["US"],
        "category": ["Market"],
        "description": "Dataset",
        "status": "Available",
        "type": "Source",
        "extra": "kept-when-expanded",
    }
    with patch.object(fusion_obj.session, "get", return_value=dataset_response):
        match_df = fusion_obj._get_exact_dataset_match("common", "prices", display_all_columns=True)

    assert match_df is not None
    assert match_df.to_dict(orient="records") == [dataset_response.json.return_value]

    report_response = MagicMock(status_code=200)
    report_response.json.return_value = {"id": "r1", "name": "Report", "description": "Desc", "ignored": "x"}
    with patch.object(fusion_obj.session, "get", return_value=report_response):
        report_df = fusion_obj._get_report_by_id("r1", False, Fusion._report_key_columns())

    assert report_df.to_dict(orient="records") == [{"id": "r1", "name": "Report", "description": "Desc"}]

    list_response = MagicMock(status_code=200)
    list_response.json.return_value = {"content": [{"id": "r2", "name": "Listed", "description": "Desc"}]}
    with patch.object(fusion_obj.session, "post", return_value=list_response):
        list_df = fusion_obj._list_reports_df(False, Fusion._report_key_columns())

    assert list_df.to_dict(orient="records") == [{"id": "r2", "name": "Listed", "description": "Desc"}]


def test_report_helpers_raise_and_select_columns(fusion_obj: Fusion) -> None:
    rep_df = pd.DataFrame([{"id": "r1", "name": "Report", "other": "ignored"}])

    assert Fusion._select_report_columns(rep_df, True, Fusion._report_key_columns()).equals(rep_df)
    assert Fusion._report_key_columns()[0] == "id"

    response = MagicMock(status_code=500)
    response.raise_for_status.side_effect = RuntimeError("boom")
    with (
        patch.object(fusion_obj.session, "get", return_value=response),
        pytest.raises(RuntimeError, match="boom"),
    ):
        fusion_obj._get_report_by_id("broken", False, Fusion._report_key_columns())


def test_fusion_dataset_filter_helpers() -> None:
    ds_df = pd.DataFrame(
        {
            "identifier": ["prices_us", "rates_eu"],
            "description": ["US prices", "European rates"],
            "category": [["a"], ["b"]],
            "region": [["us"], ["eu"]],
        }
    )

    filtered_by_id = Fusion._filter_datasets_by_contains(ds_df, "prices", id_contains=True)
    filtered_by_text = Fusion._filter_datasets_by_contains(ds_df, ["rates", "prices"], id_contains=False)
    selected = Fusion._select_dataset_columns(ds_df.copy(), display_all_columns=False)

    assert filtered_by_id["identifier"].tolist() == ["prices_us"]
    assert filtered_by_text["identifier"].tolist() == ["prices_us", "rates_eu"]
    assert selected["category"].tolist() == ["a", "b"]
    assert selected["region"].tolist() == ["us", "eu"]


def test_filter_datasets_by_product_supports_string_and_list(fusion_obj: Fusion) -> None:
    ds_df = pd.DataFrame({"identifier": ["prices_us", "rates_eu"]})
    product_df = pd.DataFrame(
        {
            "product": ["prod-a", "prod-b"],
            "dataset": ["PRICES_US", "rates_eu"],
        }
    )

    with patch.object(Fusion, "_call_for_dataframe", return_value=product_df):
        string_result = fusion_obj._filter_datasets_by_product(ds_df, "prod-a", "common")
        list_result = fusion_obj._filter_datasets_by_product(ds_df, ["prod-b"], "common")

    assert string_result["identifier"].tolist() == ["prices_us"]
    assert list_result["identifier"].tolist() == ["rates_eu"]


def test_resolve_download_folders_creates_hive_partition_paths(fusion_obj: Fusion) -> None:
    fusion_obj.fs = MagicMock()
    fusion_obj.fs.exists.return_value = False

    required_series = [
        ("common", "prices", "20240101/", "csv"),
        ("other", "rates", "20240102/", "parquet"),
    ]

    download_folders = fusion_obj._resolve_download_folders(required_series, "downloads", partitioning="hive")

    assert download_folders == [
        "downloads/common/prices/20240101",
        "downloads/other/rates/20240102",
    ]
    fusion_obj.fs.mkdir.assert_has_calls(
        [
            call("downloads/common/prices/20240101", create_parents=True),
            call("downloads/other/rates/20240102", create_parents=True),
        ]
    )


def test_resolve_distribution_file_names_handles_default_and_explicit_inputs(fusion_obj: Fusion) -> None:
    series = ("common", "prices", "20240101", "csv")
    with patch.object(
        fusion_obj,
        "list_distribution_files",
        return_value=pd.DataFrame({"@id": ["file_a/", "file_b"]}),
    ) as mock_list_distribution_files:
        resolved = fusion_obj._resolve_distribution_file_names(series, None)

    assert resolved == ["file_a", "file_b"]
    mock_list_distribution_files.assert_called_once_with(
        dataset="prices",
        series="20240101",
        file_format="csv",
        catalog="common",
    )
    assert fusion_obj._resolve_distribution_file_names(series, "single_file/") == ["single_file"]
    assert fusion_obj._resolve_distribution_file_names(series, ["file_1/", "file_2"]) == ["file_1", "file_2"]


def test_download_resolution_and_validation_helpers(fusion_obj: Fusion) -> None:
    with patch.object(fusion_obj, "_resolve_distro_tuples", return_value=[("common", "prices", "latest", "csv")]):
        assert fusion_obj._resolve_download_series("prices", "latest", "csv", "common") == [
            ("common", "prices", "latest", "csv")
        ]

    with (
        patch.object(
            fusion_obj,
            "list_distributions",
            return_value=pd.DataFrame({"identifier": ["csv"]}),
        ),
        patch.object(
            fusion_obj,
            "list_datasetmembers",
            return_value=pd.DataFrame({"identifier": ["sample"]}),
        ),
    ):
        assert fusion_obj._resolve_download_series("prices", "sample", "raw", "common") == [
            ("common", "prices", "sample", "csv")
        ]

    with (
        patch.object(fusion_obj, "list_datasetmembers", return_value=pd.DataFrame({"identifier": ["20240101"]})),
        pytest.raises(APIResponseError, match="datasetseries 'missing' not found"),
    ):
        fusion_obj._resolve_download_series("prices", "missing", "csv", "common")

    Fusion._validate_required_series([("common", "prices", "20240101", "csv")], "prices", "common", "20240101")
    Fusion._validate_download_dataset_format("csv")

    with pytest.raises(APIResponseError, match="No data available for dataset prices"):
        Fusion._validate_required_series([], "prices", "common", "20240101")

    with pytest.raises(FileFormatError, match="Dataset format exe is not supported"):
        Fusion._validate_download_dataset_format("exe")


def test_execute_downloads_without_progress_uses_fusion_filesystem_downloads(fusion_obj: Fusion) -> None:
    mock_fs = MagicMock()
    mock_fs.download.side_effect = [(True, "path-1", None), (False, "path-2", "boom")]
    download_spec = [{"rpath": "remote-1"}, {"rpath": "remote-2"}]

    with (
        patch("fusion.fusion.cpu_count", return_value=4) as mock_cpu_count,
        patch.object(fusion_obj, "get_fusion_filesystem", return_value=mock_fs),
    ):
        result = fusion_obj._execute_downloads(download_spec, n_par=2, show_progress=False)

    assert result == [(True, "path-1", None), (False, "path-2", "boom")]
    mock_cpu_count.assert_called_once_with(2)
    mock_fs.download.assert_has_calls([call(rpath="remote-1"), call(rpath="remote-2")])


def test_execute_downloads_with_progress_only_updates_for_successful_results(fusion_obj: Fusion) -> None:
    mock_fs = MagicMock()
    mock_fs.download.side_effect = [(True, "path-1", None), (False, "path-2", "boom")]
    progress = MagicMock()
    tqdm_cm = MagicMock()
    tqdm_cm.__enter__.return_value = progress
    tqdm_cm.__exit__.return_value = None
    download_spec = [{"rpath": "remote-1"}, {"rpath": "remote-2"}]

    with (
        patch("fusion.fusion.cpu_count", return_value=2),
        patch.object(fusion_obj, "get_fusion_filesystem", return_value=mock_fs),
        patch("fusion.fusion.tqdm", return_value=tqdm_cm),
    ):
        result = fusion_obj._execute_downloads(download_spec, n_par=None, show_progress=True)

    assert result == [(True, "path-1", None), (False, "path-2", "boom")]
    progress.update.assert_called_once_with(1)


def test_warn_failed_downloads_warns_only_for_failed_results() -> None:
    with pytest.warns(UserWarning, match="path-2"):
        Fusion._warn_failed_downloads([(True, "path-1", None), (False, "path-2", "boom")])

    Fusion._warn_failed_downloads([])
    Fusion._warn_failed_downloads([(True, "path-1", None)])


def test_resolve_upload_mappings_dispatches_by_path_type(fusion_obj: Fusion) -> None:
    expected_call_count = 2
    fusion_obj.fs = MagicMock()
    fusion_obj.fs.info.side_effect = [{"type": "directory"}, {"type": "file"}]
    mock_fusion_fs = MagicMock()

    with (
        patch.object(fusion_obj, "get_fusion_filesystem", return_value=mock_fusion_fs) as mock_get_fusion_filesystem,
        patch.object(
            fusion_obj,
            "_resolve_directory_upload_mappings",
            return_value=(["dir_file"], ["dir_url"], ["dir_name"]),
        ) as mock_dir,
        patch.object(
            fusion_obj,
            "_resolve_file_upload_mappings",
            return_value=(["file"], ["file_url"], ["file_name"]),
        ) as mock_file,
    ):
        directory_result = fusion_obj._resolve_upload_mappings("dir_path", "prices", "latest", "common", False)
        file_result = fusion_obj._resolve_upload_mappings("file_path", "prices", "latest", "common", False)

    assert directory_result == (["dir_file"], ["dir_url"], ["dir_name"])
    assert file_result == (["file"], ["file_url"], ["file_name"])
    assert mock_get_fusion_filesystem.call_count == expected_call_count
    mock_dir.assert_called_once_with("dir_path", "prices", "latest", "common", mock_fusion_fs)
    mock_file.assert_called_once_with("file_path", "prices", "latest", "common", False, mock_fusion_fs)


def test_resolve_file_upload_mappings_uses_local_name_validation_without_dataset(
    fusion_obj: Fusion,
) -> None:
    local_path = "/tmp/prices__common__20240102.csv"
    with (
        patch("fusion.fusion.validate_file_names", return_value=[True]),
        patch("fusion.fusion.is_dataset_raw", return_value=[False]),
    ):
        file_paths, local_urls, file_names = fusion_obj._resolve_file_upload_mappings(
            local_path,
            dataset=None,
            dt_str="latest",
            catalog="common",
            preserve_original_name=False,
            fs_fusion=MagicMock(),
        )

    assert file_paths == [local_path]
    assert local_urls == [path_to_url(local_path, False)]
    assert file_names == ["prices__common__20240102.csv"]


def test_resolve_file_upload_mappings_rejects_preserve_original_name_without_dataset(fusion_obj: Fusion) -> None:
    local_path = "/tmp/prices__common__20240102.csv"
    with (
        patch("fusion.fusion.validate_file_names", return_value=[True]),
        patch("fusion.fusion.is_dataset_raw", return_value=[False]),
        pytest.raises(ValueError, match="preserve_original_name can only be used"),
    ):
        fusion_obj._resolve_file_upload_mappings(
            local_path,
            dataset=None,
            dt_str="latest",
            catalog="common",
            preserve_original_name=True,
            fs_fusion=MagicMock(),
        )


def test_resolve_file_upload_mappings_builds_distribution_url_with_dataset_and_raw_fallback(
    fusion_obj: Fusion,
) -> None:
    file_paths, local_urls, file_names = fusion_obj._resolve_file_upload_mappings(
        "/tmp/prices.unknown",
        dataset="prices",
        dt_str="20240102",
        catalog="common",
        preserve_original_name=False,
        fs_fusion=MagicMock(),
    )

    expected_local_url = "/".join(distribution_to_url("", "prices", "20240102", "raw", "common", False).split("/")[1:])

    assert file_paths == ["/tmp/prices.unknown"]
    assert local_urls == [expected_local_url]
    assert file_names == ["prices.unknown"]


def test_normalize_upload_dt_str_and_append_background_event(fusion_obj: Fusion) -> None:
    with patch("fusion.fusion.pd.Timestamp") as mock_timestamp:
        mock_timestamp.return_value.date.return_value.strftime.return_value = "20240103"
        assert Fusion._normalize_upload_dt_str("latest") == "20240103"

    collected_events: list[dict[str, Any]] = []
    first_event = {"id": "e1", "type": "Update", "timestamp": "t1"}
    fusion_obj.events = None
    fusion_obj._append_background_event(first_event, collected_events)
    assert collected_events == [first_event]
    assert isinstance(fusion_obj.events, pd.DataFrame)
    assert fusion_obj.events.empty

    fusion_obj.events = pd.DataFrame([first_event])
    fusion_obj._append_background_event({"id": "e1", "type": "Update", "timestamp": "t1"}, collected_events)
    fusion_obj._append_background_event({"id": "e2", "type": "Create", "timestamp": "t2"}, collected_events)

    assert fusion_obj.events.to_dict(orient="records") == [
        {"id": "e1", "type": "Update", "timestamp": "t1"},
        {"id": "e2", "type": "Create", "timestamp": "t2"},
    ]


def test_prepare_event_listener_kwargs_collect_events_and_build_lineage_map(fusion_obj: Fusion) -> None:
    assert fusion_obj._prepare_event_listener_kwargs(None) == {}
    assert fusion_obj._prepare_event_listener_kwargs("evt-1") == {"headers": {"Last-Event-ID": "evt-1"}}

    messages = iter(
        [
            MagicMock(data='{"id": "1", "type": "HeartBeatNotification"}'),
            MagicMock(data='{"id": "2", "type": "DatasetUpdated", "metaData": {"catalog": "common"}}'),
        ]
    )
    fake_sseclient = types.SimpleNamespace(SSEClient=MagicMock(return_value=messages))
    with (
        patch.object(fusion_obj, "catalog_resources"),
        patch.dict("sys.modules", {"sseclient": fake_sseclient}),
    ):
        events_df = fusion_obj._collect_foreground_events("https://fusion.example/subscribe", "evt-1")

    assert events_df is not None
    assert events_df.to_dict(orient="records") == [
        {
            "id": "2",
            "type": "DatasetUpdated",
            "metaData": {"catalog": "common"},
            "catalog": "common",
        }
    ]

    relations = [
        {
            "source": {"dataset": "upstream", "catalog": "common"},
            "destination": {"dataset": "target", "catalog": "common"},
        },
        {
            "source": {"dataset": "target", "catalog": "common"},
            "destination": {"dataset": "downstream", "catalog": "restricted"},
        },
    ]
    title_lookup = {"upstream": "Upstream", "downstream": "Access Restricted"}

    assert fusion_obj._build_lineage_map("target", relations, title_lookup) == {
        "upstream": ("source", "common", "Upstream"),
        "downstream": ("produced", "restricted", "Access Restricted"),
    }


def test_add_event_auth_and_proxy_preserves_headers_and_prefers_http_proxy(fusion_obj: Fusion) -> None:
    kwargs: dict[str, Any] = {"headers": {"Last-Event-ID": "evt-1"}}

    with patch.object(fusion_obj, "catalog_resources") as mock_catalog_resources:
        fusion_obj._add_event_auth_and_proxy(kwargs)

    mock_catalog_resources.assert_called_once_with()
    assert kwargs["headers"]["Last-Event-ID"] == "evt-1"
    assert kwargs["headers"]["authorization"] == f"bearer {fusion_obj.credentials.bearer_token}"
    assert kwargs["proxy"] == fusion_obj.credentials.proxies["http"]


def test_add_event_auth_and_proxy_falls_back_to_https_proxy(credentials: object) -> None:
    https_only_credentials = credentials
    https_only_credentials.proxies = {"https": "https://proxy.example"}  # type: ignore[attr-defined]
    fusion = Fusion(credentials=https_only_credentials)  # type: ignore[arg-type]
    kwargs: dict[str, object] = {}

    with patch.object(fusion, "catalog_resources"):
        fusion._add_event_auth_and_proxy(kwargs)

    assert kwargs["proxy"] == "https://proxy.example"
