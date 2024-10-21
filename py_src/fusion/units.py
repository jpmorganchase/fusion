"""Units module"""

import re
from io import StringIO
from pint import UnitRegistry


def func_special_cases(s: str) -> str:
    """Adjust for cases where M refers to roman numeral M meaning 1000 not mega.

    Args:
        s (str): Unit encoded in a string.

    Returns:
        str: Unit encoded in a string.
    """
    for w in s.split():
        if w in ["MBbl", "Mcf", "Mboe"]:
            return s.replace(w, "thousand" + w.split("M")[-1])
    return s

def func_metric(s: str) -> str:
    """Join the word 'metric' and unit following.

    Args:
        s (str): Unit encoded in a string.

    Returns:
        str: Unit encoded in a string.
    """
    if "metric" in s:
        return s.replace("metric ", "metric_")
    return s

def func_prefix(s: str) -> str:
    """Make prefix space insensitive.

    Args:
        s (str): Unit encoded in a string.

    Returns:
        str: Unit encoded in a string.
    """
    for w in s.split(" "):
        if w in ureg._prefixes:
            return s.replace(f"{w} ", w)
    return s

def func_order(s: str) -> str:
    """Make prefix suffixable.

    Args:
        s (str): Unit encoded in a string.

    Returns:
        str: Unit encoded in a string.
    """
    if re.split(r" |/|\n", s)[-1] in ureg._prefixes:
        return " ".join(s.split(" ")[:-2] + [s.split(" ")[-1]] + [s.split(" ")[-2]])
    return s

ureg = UnitRegistry(
    preprocessors = [lambda s: s.replace("%", " percent "), func_special_cases, func_metric, func_prefix, func_order]
)

def register_units(unit_registry: UnitRegistry) -> None:
    """Register units."""
    with StringIO() as fi:
        fi.write(
            """
        kilo- = 1e3 = k- = thousand- = thousands-
        thousand- = 1e3 = thousands- = -thousand
        million- = 1e6 = millions- = mn- = m- = mm-
        billion- = 1e9 = billions- = bn- = -billion
        percent = 1 / 100 = % = pct
        cf = cubic feet
        boe = 6.118e+9 * joule = bbl
        toe = 41.9 Gjoule
        kgoe = 0.001 * toe
        USD = [currency]
        EUR = nan USD
        GBP = nan USD
        JPY = nan USD
        capita = [capita]
        Share = [share] = share
        tC02e = [metric_tonne_of_carbon_dioxide_equivalent]
        kgC02e = 0.001 * tC02e
        """
        )
        fi.seek(0)
        unit_registry.load_definitions(fi)