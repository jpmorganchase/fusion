"""Test file for Dataflow integration"""

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
    flow = Dataflow.from_dict(data)
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
