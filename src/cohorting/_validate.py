"""Validation for cohort split maps."""

from __future__ import annotations

import math

from cohorting._models import SplitInput, _normalize_splits


def validate_splits(splits: SplitInput) -> None:
    """Validate that splits span exactly [0, 1) with no gaps or overlaps.

    Accepts either a SplitMap dict or a list of CohortSplit instances.

    Parameters
    ----------
    splits : SplitInput
        Cohort split map (dict or list of CohortSplit) to validate.

    Raises
    ------
    ValueError
        If splits is empty, doesn't start at 0.0, doesn't end at 1.0,
        has a gap or overlap between adjacent bounds, has inverted or
        zero-width bounds, or has NaN bounds.

    Examples
    --------
    >>> from cohorting import validate_splits, even_split
    >>> validate_splits(splits=even_split(names=["control", "treatment"]))
    >>> try:
    ...     validate_splits(splits={})
    ... except ValueError as exc:
    ...     print(exc)
    splits must not be empty
    """
    split_map = _normalize_splits(splits)

    if not split_map:
        raise ValueError("splits must not be empty")

    sorted_items = sorted(split_map.items(), key=lambda item: item[1]["lower"])

    for name, bounds in sorted_items:
        lower, upper = bounds["lower"], bounds["upper"]
        if math.isnan(lower) or math.isnan(upper):
            raise ValueError(
                f"cohort '{name}' has NaN bounds: lower={lower}, upper={upper}"
            )
        if lower >= upper:
            raise ValueError(
                f"cohort '{name}' has inverted or zero-width bounds: "
                f"lower={lower}, upper={upper}"
            )

    first_name, first_bounds = sorted_items[0]
    if first_bounds["lower"] != 0.0:
        raise ValueError(
            f"splits must start at 0.0; "
            f"'{first_name}' starts at {first_bounds['lower']}"
        )

    last_name, last_bounds = sorted_items[-1]
    if last_bounds["upper"] != 1.0:
        raise ValueError(
            f"splits must end at 1.0; '{last_name}' ends at {last_bounds['upper']}"
        )

    for i in range(len(sorted_items) - 1):
        curr_name, curr_bounds = sorted_items[i]
        next_name, next_bounds = sorted_items[i + 1]
        if curr_bounds["upper"] < next_bounds["lower"]:
            raise ValueError(
                f"gap between '{curr_name}' (upper={curr_bounds['upper']}) "
                f"and '{next_name}' (lower={next_bounds['lower']})"
            )
        if curr_bounds["upper"] > next_bounds["lower"]:
            raise ValueError(
                f"overlap between '{curr_name}' (upper={curr_bounds['upper']}) "
                f"and '{next_name}' (lower={next_bounds['lower']})"
            )
