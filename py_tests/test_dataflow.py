"""Test file for Dataflow integration"""

import pandas as pd
import pytest

from fusion.dataflow import Dataflow
from fusion.fusion import Fusion


def test_dataflow_basic_fields() -> None:
    flow = Dataflow(
        providerNode={"name": "CRM_DB", "nodeType": "Database"},
        consumerNode={"name": "DWH", "nodeType": "Database"},
        description="CRM to DWH load",
        transportType="API",
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
        providerNode={"name": "S3", "nodeType": "Storage"},
        consumerNode={"name": "Analytics", "nodeType": "Dashboard"},
        description="S3 to Analytics feed",
        transportType="FILE TRANSFER",
        frequency="WEEKLY",
    )
    result = flow.to_dict()
    assert result["providerNode"]["name"] == "S3"
    assert result["consumerNode"]["nodeType"] == "Dashboard"
    assert result["frequency"] == "WEEKLY"


def test_dataflow_from_dict() -> None:
    data = {
        "providerNode": {"name": "App1", "nodeType": "User Tool"},
        "consumerNode": {"name": "DWH", "nodeType": "Database"},
        "description": "Dict-based dataflow",
        "transportType": "API",
        "frequency": "DAILY",
    }
    flow = Dataflow.from_dict(data)
    assert isinstance(flow, Dataflow)
    assert flow.providerNode is not None
    assert flow.providerNode["name"] == "App1"
    assert flow.transportType == "API"


def test_dataflow_from_object_series() -> None:
    series = pd.Series(
        {
            "providerNode": {"name": "CRM_DB", "nodeType": "Database"},
            "consumerNode": {"name": "DWH", "nodeType": "Database"},
            "description": "Series-based dataflow",
            "transportType": "API",
            "frequency": "DAILY",
        }
    )
    # provider/consumer optional at init so we can start empty and let from_object populate
    flow = Dataflow().from_object(series)
    assert isinstance(flow, Dataflow)
    assert flow.description == "Series-based dataflow"
    assert flow.providerNode is not None
    assert flow.providerNode["name"] == "CRM_DB"


def test_dataflow_from_object_json() -> None:
    json_str = """{
        "providerNode": {"name": "SystemA", "nodeType": "App"},
        "consumerNode": {"name": "SystemB", "nodeType": "Database"},
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
                "providerNode": {"name": "CRM_DB", "nodeType": "Database"},
                "consumerNode": {"name": "DWH", "nodeType": "Database"},
                "description": "Row1",
                "transportType": "API",
                "frequency": "DAILY",
            },
            {
                "providerNode": {"name": "S3", "nodeType": "Storage"},
                "consumerNode": {"name": "Analytics", "nodeType": "Dashboard"},
                "description": "Row2",
                "transportType": "FILE TRANSFER",
                "frequency": "WEEKLY",
            },
        ]
    )
    flows = Dataflow.from_dataframe(frame, client=fusion_obj)
    assert isinstance(flows, list)
    test_value = 2
    assert len(flows) == test_value
    assert all(isinstance(f, Dataflow) for f in flows)
    assert flows[0].description == "Row1"
    assert flows[1].frequency == "WEEKLY"


def test_dataflow_validate_nodes_for_create_passes() -> None:
    flow = Dataflow(
        providerNode={"name": "CRM_DB", "nodeType": "Database"},
        consumerNode={"name": "DWH", "nodeType": "Database"},
    )
    # should not raise
    flow._validate_nodes_for_create()


def test_dataflow_validate_nodes_for_create_raises() -> None:
    flow = Dataflow(providerNode=None, consumerNode=None)
    with pytest.raises(ValueError, match="must be a dict"):
        flow._validate_nodes_for_create()
