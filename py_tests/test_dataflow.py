"""Test dataflow.py"""

from fusion.dataflow import InputDataFlow, OutputDataFlow


def test_inputdataflow_class_object_representation() -> None:
    """Test the object representation of the Dataflow class."""
    dataflow = InputDataFlow(identifier="my_dataflow", flow_details={"key": "value"})
    assert repr(dataflow)

def test_outputdataflow_class_object_representation() -> None:
    """Test the object representation of the Dataflow class."""
    dataflow = OutputDataFlow(identifier="my_dataflow", flow_details={"key": "value"})
    assert repr(dataflow)

