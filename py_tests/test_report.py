"""Test file for report.py"""

from fusion.report import Report


def test_report_class_value_error() -> None:
    """Test the Report class when the 'report' attribute is not provided."""
    import pytest

    with pytest.raises(ValueError, match="The 'report' attribute is required and cannot be empty."):
        Report(identifier="my_report")

def test_report_class_object_representation() -> None:
    """Test the object representation of the Report class."""
    report = Report(identifier="my_report", report={"key": "value"})
    assert repr(report)