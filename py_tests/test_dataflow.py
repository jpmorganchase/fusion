"""Test file for updated dataflow.py and Dataflow integration"""

import pytest
import pandas as pd

from fusion.fusion import Fusion
from fusion.dataflow import Dataflow


def test_dataflow_basic_fields() -> None:
    flow = Dataflow(
        providerNode={"name": "CRM_DB", "dataNodeType": "Database"},
        consumerNode={"name": "DWH", "dataNodeType": "Database"},
        description="CRM to DWH load",
        transportType="API",
        frequency="DAILY",
    )
    assert flow.providerNode["name"] == "CRM_DB"
    assert flow.consumerNode["name"] == "DWH"
    assert flow.transportType == "API"
    assert flow.frequency == "DAILY"


def test_dataflow_to_dict() -> None:
    flow = Dataflow(
        providerNode={"name": "S3", "dataNodeType": "Storage"},
        consumerNode={"name": "Analytics", "dataNodeType": "Dashboard"},
        description="S3 to Analytics feed",
        transportType="FILE TRANSFER",
        frequency="WEEKLY",
    )
    result = flow.to_dict()
    assert result["providerNode"]["name"] == "S3"
    assert result["consumerNode"]["dataNodeType"] == "Dashboard"
    assert result["frequency"] == "WEEKLY"


def test_dataflow_validation_raises() -> None:
    flow = Dataflow(
        providerNode={"name": "", "dataNodeType": "Database"},
        consumerNode={"name": "", "dataNodeType": "Database"},
    )
    with pytest.raises(ValueError, match="Missing required fields"):
        flow.validate()


def test_dataflow_from_dict() -> None:
    data = {
        "providerNode": {"name": "App1", "dataNodeType": "User Tool"},
        "consumerNode": {"name": "DWH", "dataNodeType": "Database"},
        "description": "Dict-based dataflow",
        "transportType": "API",
        "frequency": "DAILY",
    }
    flow = Dataflow.from_dict(data)
    assert isinstance(flow, Dataflow)
    assert flow.providerNode["name"] == "App1"
    assert flow.transportType == "API"


def test_dataflow_from_object_series() -> None:
    series = pd.Series(
        {
            "providerNode": {"name": "CRM_DB", "dataNodeType": "Database"},
            "consumerNode": {"name": "DWH", "dataNodeType": "Database"},
            "description": "Series-based dataflow",
            "transportType": "API",
            "frequency": "DAILY",
        }
    )
    flow = Dataflow(
        providerNode={"name": "TMP", "dataNodeType": "TMP"},
        consumerNode={"name": "TMP", "dataNodeType": "TMP"},
    ).from_object(series)
    assert isinstance(flow, Dataflow)
    assert flow.description == "Series-based dataflow"


def test_dataflow_from_dataframe(fusion_obj: Fusion) -> None:
    frame = pd.DataFrame(
        [
            {
                "providerNode": {"name": "CRM_DB", "dataNodeType": "Database"},
                "consumerNode": {"name": "DWH", "dataNodeType": "Database"},
                "description": "Row1",
                "transportType": "API",
                "frequency": "DAILY",
            },
            {
                "providerNode": {"name": "S3", "dataNodeType": "Storage"},
                "consumerNode": {"name": "Analytics", "dataNodeType": "Dashboard"},
                "description": "Row2",
                "transportType": "FILE TRANSFER",
                "frequency": "WEEKLY",
            },
        ]
    )
    flows = Dataflow.from_dataframe(frame, client=fusion_obj)
    assert isinstance(flows, list)
    assert len(flows) == 2
    assert all(isinstance(f, Dataflow) for f in flows)
    assert flows[0].description == "Row1"
    assert flows[1].frequency == "WEEKLY"
