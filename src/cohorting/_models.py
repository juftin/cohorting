"""Data models and type detection helpers for the cohorting library."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict, TypeGuard

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import polars as pl


class CohortBounds(TypedDict):
    """Defines the [lower, upper) float range for a single cohort bucket.

    Examples
    --------
    >>> bounds: CohortBounds = {"lower": 0.0, "upper": 0.5}
    >>> bounds["lower"], bounds["upper"]
    (0.0, 0.5)
    """

    lower: float
    """Inclusive lower bound."""
    upper: float
    """Exclusive upper bound."""


SplitMap = dict[str, CohortBounds]
"""Maps cohort names to their [lower, upper) float ranges.

Must span exactly [0, 1) with no gaps or overlaps. Pass to validate_splits
before use.
"""


@dataclass(slots=True, frozen=True)
class CohortSplit:
    """OOP alternative to a (name, CohortBounds) dict entry.

    Use when you prefer a class-based API over raw dicts. Pass a list of
    ``CohortSplit`` instances directly to :func:`assign_cohorts` or
    :class:`cohorting.Experiment` — both accept lists and dicts interchangeably.

    Attributes
    ----------
    name : str
        Cohort name (e.g. "control", "treatment").
    lower : float
        Inclusive lower bound of the cohort range.
    upper : float
        Exclusive upper bound of the cohort range.

    Examples
    --------
    >>> from cohorting import assign_cohorts
    >>> splits = [
    ...     CohortSplit(name="control", lower=0.0, upper=0.5),
    ...     CohortSplit(name="treatment", lower=0.5, upper=1.0),
    ... ]
    >>> assign_cohorts(data="user_1", splits=splits, salt="exp")
    'control'
    """

    name: str
    """Cohort name."""
    lower: float
    """Inclusive lower bound."""
    upper: float
    """Exclusive upper bound."""


def even_split(names: list[str]) -> SplitMap:
    """Build a SplitMap that divides [0, 1) equally among the given cohort names.

    Parameters
    ----------
    names : list[str]
        Ordered list of cohort names. Must not be empty.

    Returns
    -------
    SplitMap
        A valid SplitMap with equal-width buckets. The last bucket always ends
        at exactly 1.0 to avoid floating-point rounding gaps.

    Raises
    ------
    ValueError
        If names is empty.

    Examples
    --------
    >>> even_split(names=["control", "treatment"])
    {'control': {'lower': 0.0, 'upper': 0.5}, 'treatment': {'lower': 0.5, 'upper': 1.0}}
    >>> even_split(names=["a", "b", "c"])  # doctest: +NORMALIZE_WHITESPACE
    {'a': {'lower': 0.0, 'upper': 0.3333333333333333},
     'b': {'lower': 0.3333333333333333, 'upper': 0.6666666666666666},
     'c': {'lower': 0.6666666666666666, 'upper': 1.0}}
    """
    if not names:
        raise ValueError("names must not be empty")
    n = len(names)
    step = 1.0 / n
    result: SplitMap = {}
    for i, name in enumerate(names):
        lower = i * step
        upper = 1.0 if i == n - 1 else (i + 1) * step
        result[name] = {"lower": lower, "upper": upper}
    return result


def weighted_split(weights: dict[str, float]) -> SplitMap:
    """Build a SplitMap from a dict of cohort names to fractional weights.

    Parameters
    ----------
    weights : dict[str, float]
        Mapping of cohort name to its fraction of traffic. Values must sum to
        1.0 (within a small tolerance). Order determines bucket placement.

    Returns
    -------
    SplitMap
        A valid SplitMap. The last bucket is pinned to exactly 1.0 to avoid
        floating-point rounding gaps.

    Raises
    ------
    ValueError
        If weights is empty or does not sum to 1.0.

    Notes
    -----
    Negative weights are rejected. The last bucket is always pinned to exactly
    ``1.0`` to avoid floating-point rounding gaps.

    Examples
    --------
    >>> weighted_split({"control": 0.9, "treatment": 0.1})
    {'control': {'lower': 0.0, 'upper': 0.9}, 'treatment': {'lower': 0.9, 'upper': 1.0}}
    """
    if not weights:
        raise ValueError("weights must not be empty")
    if any(v < 0 for v in weights.values()):
        negative = {k: v for k, v in weights.items() if v < 0}
        raise ValueError(f"weights must not be negative; got {negative}")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1.0; got {total}")
    result: SplitMap = {}
    lower = 0.0
    items = list(weights.items())
    for i, (name, weight) in enumerate(items):
        upper = 1.0 if i == len(items) - 1 else lower + weight
        result[name] = {"lower": lower, "upper": upper}
        lower = upper
    return result


def _splits_from_list(cohort_splits: list[CohortSplit]) -> SplitMap:
    """Convert a list of CohortSplit instances to a SplitMap dict."""
    return {s.name: {"lower": s.lower, "upper": s.upper} for s in cohort_splits}


SplitInput = SplitMap | list[CohortSplit]
"""Accepted split format for all public APIs: a SplitMap dict or list of CohortSplit."""


def _normalize_splits(splits: SplitInput) -> SplitMap:
    """Normalize SplitInput to SplitMap, converting list[CohortSplit] if needed.

    Parameters
    ----------
    splits : SplitInput
        Either a SplitMap dict or a list of CohortSplit instances.

    Returns
    -------
    SplitMap
        Normalized SplitMap dict.
    """
    if isinstance(splits, list):
        return _splits_from_list(splits)
    return splits


def _is_pandas_series(obj: object) -> TypeGuard[pd.Series]:
    """Return True if obj is a pd.Series or subclass.

    Uses sys.modules to avoid triggering an import. If pandas is not yet
    loaded, the object cannot be a pd.Series, so False is returned immediately.
    """
    _pd = sys.modules.get("pandas")
    if _pd is None:
        return False
    return isinstance(obj, _pd.Series)


def _is_polars_series(obj: object) -> TypeGuard[pl.Series]:
    """Return True if obj is a pl.Series or subclass.

    Uses sys.modules to avoid triggering an import. If polars is not yet
    loaded, the object cannot be a pl.Series, so False is returned immediately.
    """
    _pl = sys.modules.get("polars")
    if _pl is None:
        return False
    return isinstance(obj, _pl.Series)


def _is_pandas_frame(obj: object) -> TypeGuard[pd.DataFrame]:
    """Return True if obj is a pd.DataFrame or subclass.

    Uses sys.modules to avoid triggering an import. If pandas is not yet
    loaded, the object cannot be a pd.DataFrame, so False is returned immediately.
    """
    _pd = sys.modules.get("pandas")
    if _pd is None:
        return False
    return isinstance(obj, _pd.DataFrame)


def _is_polars_frame(obj: object) -> TypeGuard[pl.DataFrame]:
    """Return True if obj is a pl.DataFrame or subclass.

    Uses sys.modules to avoid triggering an import. If polars is not yet
    loaded, the object cannot be a pl.DataFrame, so False is returned immediately.
    """
    _pl = sys.modules.get("polars")
    if _pl is None:
        return False
    return isinstance(obj, _pl.DataFrame)


def _is_numpy_array(obj: object) -> TypeGuard[np.ndarray]:
    """Return True if obj is an np.ndarray or subclass.

    Uses sys.modules to avoid triggering an import. If numpy is not yet
    loaded, the object cannot be an ndarray, so False is returned immediately.
    """
    _np = sys.modules.get("numpy")
    if _np is None:
        return False
    return isinstance(obj, _np.ndarray)
