"""Test file for Dataflow integration"""

from typing import Any, Optional, cast
from unittest.mock import patch

import pandas as pd
import pytest

from fusion.dataflow import Dataflow
from fusion.fusion import Fusion


def test_dataflow_basic_fields() -> None:
    flow = Dataflow(
        provider_node={"name": "CRM_DB", "type": "Database"},
        consumer_node={"name": "DWH", "type": "Database"},
        description="CRM to DWH load",
        transport_type="API",
        frequency="DAILY",
    )

    assert flow.providerNode is not None
    assert flow.consumerNode is not None
    assert flow.providerNode["name"] == "CRM_DB"
    assert flow.consumerNode["name"] == "DWH"
    assert flow.transportType == "API"
    assert flow.frequency == "DAILY"


def test_dataflow_to_dict() -> None:
    flow = Dataflow(
        provider_node={"name": "S3", "type": "Storage"},
        consumer_node={"name": "Analytics", "type": "Dashboard"},
        description="S3 to Analytics feed",
        transport_type="FILE TRANSFER",
        frequency="WEEKLY",
    )
    result = flow.to_dict()
    assert result["providerNode"]["name"] == "S3"
    assert result["consumerNode"]["type"] == "Dashboard"
    assert result["frequency"] == "WEEKLY"


def test_dataflow_from_dict() -> None:
    data = {
        "providerNode": {"name": "App1", "type": "User Tool"},
        "consumerNode": {"name": "DWH", "type": "Database"},
        "description": "Dict-based dataflow",
        "transportType": "API",
        "frequency": "DAILY",
    }
    flow = Dataflow().from_object(data)
    assert isinstance(flow, Dataflow)
    assert flow.providerNode is not None
    assert flow.providerNode["name"] == "App1"
    assert flow.transportType == "API"


def test_dataflow_from_object_series() -> None:
    series = pd.Series(
        {
            "providerNode": {"name": "CRM_DB", "type": "Database"},
            "consumerNode": {"name": "DWH", "type": "Database"},
            "description": "Series-based dataflow",
            "transportType": "API",
            "frequency": "DAILY",
        }
    )

    flow = Dataflow().from_object(series)
    assert isinstance(flow, Dataflow)
    assert flow.description == "Series-based dataflow"
    assert flow.providerNode is not None
    assert flow.providerNode["name"] == "CRM_DB"


def test_dataflow_from_object_json() -> None:
    json_str = """{
        "providerNode": {"name": "SystemA", "type": "App"},
        "consumerNode": {"name": "SystemB", "type": "Database"},
        "description": "JSON-based dataflow",
        "frequency": "MONTHLY"
    }"""
    flow = Dataflow().from_object(json_str)
    assert isinstance(flow, Dataflow)
    assert flow.consumerNode is not None
    assert flow.consumerNode["name"] == "SystemB"
    assert flow.frequency == "MONTHLY"


def test_dataflow_from_dataframe(fusion_obj: Fusion) -> None:
    frame = pd.DataFrame(
        [
            {
                "providerNode": {"name": "CRM_DB", "type": "Database"},
                "consumerNode": {"name": "DWH", "type": "Database"},
                "description": "Row1",
                "transportType": "API",
                "frequency": "DAILY",
            },
            {
                "providerNode": {"name": "S3", "type": "Storage"},
                "consumerNode": {"name": "Analytics", "type": "Dashboard"},
                "description": "Row2",
                "transportType": "FILE TRANSFER",
                "frequency": "WEEKLY",
            },
        ]
    )
    flows = Dataflow.from_dataframe(frame, client=fusion_obj)
    assert isinstance(flows, list)
    num = 2
    assert len(flows) == num
    assert all(isinstance(f, Dataflow) for f in flows)
    assert flows[0].description == "Row1"
    assert flows[1].frequency == "WEEKLY"


def test_dataflow_validate_nodes_for_create_passes() -> None:
    flow = Dataflow(
        provider_node={"name": "CRM_DB", "type": "Database"},
        consumer_node={"name": "DWH", "type": "Database"},
        connection_type="API",
    )

    flow._validate_nodes_for_create()


def test_dataflow_validate_nodes_for_create_raises() -> None:
    flow = Dataflow(provider_node=None, consumer_node=None)
    with pytest.raises(ValueError, match="must be a dict"):
        flow._validate_nodes_for_create()


def test_dataflow_to_dict_drop_none_false_includes_nulls() -> None:
    """to_dict with drop_none=False should keep None values and defaults."""
    flow = Dataflow(
        provider_node={"name": "SRC", "type": "Database"},
        consumer_node={"name": "DST", "type": "Database"},
        description=None,
        id=None,
        frequency="DAILY",
    )
    out = flow.to_dict(drop_none=False)

    assert "providerNode" in out
    assert "consumerNode" in out

    assert "description" in out
    assert out["description"] is None
    assert "id" in out
    assert out["id"] is None

    assert "datasets" in out
    assert isinstance(out["datasets"], list)

    assert out["frequency"] == "DAILY"


def test_dataflow_from_dataframe_skips_invalid_rows_and_sets_client(fusion_obj: Fusion) -> None:
    """from_dataframe should skip invalid rows and attach the provided client to valid ones."""
    frame = pd.DataFrame(
        [
            {
                "providerNode": {"name": "OnlyProvider", "type": "Database"},
                "description": "Invalid row",
                "frequency": "DAILY",
            },
            {
                "providerNode": {"name": "SRC", "type": "Database"},
                "consumerNode": {"name": "DST", "type": "Database"},
                "description": "Valid row",
                "transportType": "API",
                "frequency": "WEEKLY",
            },
        ]
    )
    flows = Dataflow.from_dataframe(frame, client=fusion_obj)
    assert isinstance(flows, list)
    assert len(flows) == 1
    assert isinstance(flows[0], Dataflow)
    assert flows[0].description == "Valid row"

    assert flows[0].client is fusion_obj


def test_dataflow_normalizes_fields_and_wraps_datasets() -> None:
    flow = Dataflow(
        provider_node={"name": " CRM_DB ", "type": " Database "},
        consumer_node={"name": " DWH ", "type": " Database "},
        description="  desc  ",
        frequency=" DAILY ",
        datasets={"identifier": "ds1"},  # type: ignore[arg-type]
    )

    assert flow.provider_node == {"name": "CRM_DB", "type": "Database"}
    assert flow.consumer_node == {"name": "DWH", "type": "Database"}
    assert flow.description == "desc"
    assert flow.frequency == "DAILY"
    assert flow.datasets == [{"identifier": "ds1"}]


def test_dataflow_from_object_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="Unsupported string input"):
        Dataflow().from_object("not json")

    with pytest.raises(TypeError, match="Could not resolve the object provided"):
        Dataflow().from_object(123)  # type: ignore[arg-type]


def test_dataflow_use_client_raises_without_bound_client() -> None:
    with pytest.raises(ValueError, match="A Fusion client object is required"):
        Dataflow()._use_client(None)


def test_dataflow_create_posts_expected_payload_and_url() -> None:
    flow = Dataflow(
        id="df-1",
        provider_node={"name": "CRM_DB", "type": "Database"},
        consumer_node={"name": "DWH", "type": "Database"},
        description="Sync",
        connection_type="Consumes From",
    )

    class _Resp:
        status_code = 201
        ok = True
        text = ""
        content = b""

    class _Sess:
        def __init__(self) -> None:
            self.last_url: Optional[str] = None
            self.last_json: Optional[dict[str, Any]] = None

        def post(self, url: str, json: dict[str, Any]) -> _Resp:
            self.last_url = url
            self.last_json = json
            return _Resp()

    class _Fusion:
        def __init__(self) -> None:
            self.session = _Sess()

        def _get_new_root_url(self) -> str:
            return "https://unit.test"

    client = _Fusion()
    flow.client = cast(Fusion, client)

    with patch("fusion.dataflow.requests_raise_for_status") as mock_raise:
        resp = flow.create(return_resp_obj=True)

    assert isinstance(resp, _Resp)
    assert client.session.last_url == "https://unit.test/api/corelineage-service/v1/lineage/dataflows"
    assert client.session.last_json is not None
    assert "id" not in client.session.last_json
    assert client.session.last_json["providerNode"]["name"] == "CRM_DB"
    mock_raise.assert_called_once()


def test_dataflow_update_puts_expected_payload_and_url() -> None:
    flow = Dataflow(
        id="df-1",
        provider_node={"name": "SRC", "type": "Database"},
        consumer_node={"name": "DST", "type": "Database"},
        description="Updated",
        connection_type="Consumes From",
    )

    class _Resp:
        status_code = 200
        ok = True
        text = ""
        content = b""

    class _Sess:
        def __init__(self) -> None:
            self.last_url: Optional[str] = None
            self.last_json: Optional[dict[str, Any]] = None

        def put(self, url: str, json: dict[str, Any]) -> _Resp:
            self.last_url = url
            self.last_json = json
            return _Resp()

    class _Fusion:
        def __init__(self) -> None:
            self.session = _Sess()

        def _get_new_root_url(self) -> str:
            return "https://unit.test"

    client = _Fusion()
    flow.client = cast(Fusion, client)

    with patch("fusion.dataflow.requests_raise_for_status") as mock_raise:
        flow.update()

    assert client.session.last_url == "https://unit.test/api/corelineage-service/v1/lineage/dataflows/df-1"
    assert client.session.last_json == {"description": "Updated", "datasets": [], "connectionType": "Consumes From"}
    mock_raise.assert_called_once()


def test_dataflow_update_fields_and_delete_use_expected_request_shapes() -> None:
    flow = Dataflow(
        id="df-2",
        provider_node={"name": "SRC", "type": "Database"},
        consumer_node={"name": "DST", "type": "Database"},
        description="Patch me",
    )

    class _Resp:
        status_code = 200
        ok = True
        text = ""
        content = b""

    class _Sess:
        def __init__(self) -> None:
            self.patch_url: Optional[str] = None
            self.patch_json: Optional[dict[str, Any]] = None
            self.delete_url: Optional[str] = None

        def patch(self, url: str, json: dict[str, Any]) -> _Resp:
            self.patch_url = url
            self.patch_json = json
            return _Resp()

        def delete(self, url: str) -> _Resp:
            self.delete_url = url
            return _Resp()

    class _Fusion:
        def __init__(self) -> None:
            self.session = _Sess()

        def _get_new_root_url(self) -> str:
            return "https://unit.test"

    client = _Fusion()
    flow.client = cast(Fusion, client)

    with patch("fusion.dataflow.requests_raise_for_status") as mock_raise:
        flow.update_fields()
        flow.delete()

    assert client.session.patch_url == "https://unit.test/api/corelineage-service/v1/lineage/dataflows/df-2"
    assert client.session.patch_json == {"description": "Patch me", "datasets": []}
    assert client.session.delete_url == "https://unit.test/api/corelineage-service/v1/lineage/dataflows/df-2"
    expected_request_count = 2
    assert mock_raise.call_count == expected_request_count


@pytest.mark.parametrize(
    ("method_name", "expected"),
    [("update", "update()"), ("update_fields", "update_fields()"), ("delete", "delete()")],
)
def test_dataflow_methods_require_id(method_name: str, expected: str) -> None:
    class _Sess:
        def put(self, url: str, json: dict[str, Any]) -> None:  # noqa: ARG002
            return None

        def patch(self, url: str, json: dict[str, Any]) -> None:  # noqa: ARG002
            return None

        def delete(self, url: str) -> None:  # noqa: ARG002
            return None

    class _Fusion:
        def __init__(self) -> None:
            self.session = _Sess()

        def _get_new_root_url(self) -> str:
            return "https://unit.test"

    flow = Dataflow(
        provider_node={"name": "SRC", "type": "Database"},
        consumer_node={"name": "DST", "type": "Database"},
    )
    flow.client = cast(Fusion, _Fusion())

    with pytest.raises(ValueError, match=expected):
        getattr(flow, method_name)()
