"""Unit test modul for units.py"""

from __future__ import annotations

from logging import getLogger

from pint import UnitRegistry

from fusion import units

log = getLogger()

def test_func_special_cases() -> None:
    """Test func_special_cases."""
    sample1 = "xxx MBbl"
    sample2 = "Mcf yyy"
    sample3 = "zzz Mboe zzz"
    sample4 = "xxxMBbl"
    sample5 = "McfyyyxxxMBbl"

    assert units.func_special_cases(sample1) == "xxx thousandBbl"
    assert units.func_special_cases(sample2) == "thousandcf yyy"
    assert units.func_special_cases(sample3) == "zzz thousandboe zzz"
    assert units.func_special_cases(sample4) == sample4
    assert units.func_special_cases(sample5) == sample5

def test_func_metric() -> None:
    """Test func_metric."""
    sample1 = "metric ton of carbon"
    sample2 = "metrics of"
    sample3 = "metric_ton of"
    sample4 = "the metric of x"
    sample5 = "metric_ton of metric_ton"
    sample6 = "ton of carbon"

    assert units.func_metric(sample1) == "metric_ton of carbon"
    assert units.func_metric(sample2) == sample2
    assert units.func_metric(sample3) == sample3
    assert units.func_metric(sample4) == "the metric_of x"
    assert units.func_metric(sample5) == "metric_ton of metric_ton"
    assert units.func_metric(sample6) == sample6

def test_func_prefix() -> None:
    """test func_prefix."""
    sample1 = "yyy pico quart yyyy"
    sample2 = "the zetta byte"
    sample3 = "zepto gram"
    sample4 = "1111 kilo meter 1111"
    sample5 = "1111 kilometer 1111"
    sample6 = "y byte"
    sample7 = "the semicircle"

    assert units.func_prefix(sample1) == "yyy picoquart yyyy"
    assert units.func_prefix(sample2) == "the zettabyte"
    assert units.func_prefix(sample3) == "zeptogram"
    assert units.func_prefix(sample4) == "1111 kilometer 1111"
    assert units.func_prefix(sample5) == sample5
    assert units.func_prefix(sample6) == "ybyte"
    assert units.func_prefix(sample7) == sample7

def test_func_order() -> None:
    """"Test func_order."""
    sample1 = "10 liters milli"
    sample2 = "10 milli liters"

    assert units.func_order(sample1) == "10 milli liters"
    assert units.func_order(sample2) == sample2


def test_ureg() -> None:
    """Test register_units."""
    ureg = UnitRegistry(
        preprocessors = [lambda s: s.replace("%", " percent "), units.func_special_cases, units.func_metric, units.func_prefix, units.func_order]
    )
    assert ureg is not None
