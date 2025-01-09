"""Test file for report.py"""

from fusion.report import Report


def test_report_class_object_representation() -> None:
    """Test the object representation of the Report class."""
    report = Report(identifier="my_report", report={"key": "value"})
    assert repr(report)