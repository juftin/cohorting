"""Tests for cohorting data models and type detection helpers."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from cohorting._models import (
    CohortBounds,
    SplitMap,
    _is_numpy_array,
    _is_pandas_frame,
    _is_pandas_series,
    _is_polars_frame,
    _is_polars_series,
    even_split,
    weighted_split,
)


def test_even_split_two_way() -> None:
    """even_split produces equal halves for two cohorts."""
    splits = even_split(["control", "treatment"])
    assert splits["control"] == {"lower": 0.0, "upper": 0.5}
    assert splits["treatment"] == {"lower": 0.5, "upper": 1.0}


def test_even_split_three_way_last_ends_at_one() -> None:
    """even_split last bucket always ends at exactly 1.0."""
    splits = even_split(["a", "b", "c"])
    assert splits["a"]["lower"] == 0.0
    assert splits["c"]["upper"] == 1.0


def test_even_split_single() -> None:
    """even_split with one name spans the full [0, 1) range."""
    splits = even_split(["all"])
    assert splits["all"] == {"lower": 0.0, "upper": 1.0}


def test_even_split_empty_raises() -> None:
    """even_split raises ValueError for an empty list."""
    with pytest.raises(ValueError, match="empty"):
        even_split([])


def test_even_split_is_valid() -> None:
    """even_split output passes validate_splits for any length."""
    from cohorting._validate import validate_splits

    for n in range(1, 6):
        validate_splits(even_split([f"cohort_{i}" for i in range(n)]))


def test_weighted_split_two_cohorts() -> None:
    """weighted_split divides [0, 1) by proportional weights."""
    splits = weighted_split({"control": 0.9, "treatment": 0.1})
    assert splits["control"] == {"lower": 0.0, "upper": 0.9}
    assert splits["treatment"]["lower"] == pytest.approx(0.9)
    assert splits["treatment"]["upper"] == 1.0


def test_weighted_split_last_pinned_to_one() -> None:
    """weighted_split last bucket always ends at exactly 1.0."""
    splits = weighted_split({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})
    assert splits["c"]["upper"] == 1.0


def test_weighted_split_empty_raises() -> None:
    """weighted_split raises ValueError for an empty dict."""
    with pytest.raises(ValueError, match="empty"):
        weighted_split({})


def test_weighted_split_wrong_sum_raises() -> None:
    """weighted_split raises ValueError when weights don't sum to 1.0."""
    with pytest.raises(ValueError, match="sum"):
        weighted_split({"a": 0.4, "b": 0.4})


def test_weighted_split_is_valid() -> None:
    """weighted_split output passes validate_splits."""
    from cohorting._validate import validate_splits

    validate_splits(weighted_split({"control": 0.7, "treatment": 0.3}))


def test_weighted_split_negative_weight_raises() -> None:
    """weighted_split raises ValueError for negative weights."""
    with pytest.raises(ValueError, match="negative"):
        weighted_split({"a": 1.5, "b": -0.5})


def test_cohort_bounds_construction() -> None:
    """CohortBounds is a TypedDict with lower and upper float fields."""
    bounds: CohortBounds = {"lower": 0.0, "upper": 0.5}
    assert bounds["lower"] == 0.0
    assert bounds["upper"] == 0.5


def test_split_map_construction() -> None:
    """SplitMap is a dict[str, CohortBounds]."""
    splits: SplitMap = {
        "control": {"lower": 0.0, "upper": 0.5},
        "treatment": {"lower": 0.5, "upper": 1.0},
    }
    assert splits["control"]["lower"] == 0.0
    assert splits["treatment"]["upper"] == 1.0


def test_is_pandas_series_false_for_plain_types() -> None:
    """_is_pandas_series returns False for non-pandas objects."""
    assert _is_pandas_series("hello") is False
    assert _is_pandas_series([1, 2, 3]) is False
    assert _is_pandas_series(42) is False


def test_is_polars_series_false_for_plain_types() -> None:
    """_is_polars_series returns False for non-polars objects."""
    assert _is_polars_series("hello") is False
    assert _is_polars_series([1, 2, 3]) is False


def test_is_pandas_series_true() -> None:
    """_is_pandas_series returns True for pd.Series."""
    assert _is_pandas_series(pd.Series(["a", "b"])) is True


def test_is_pandas_series_false_for_dataframe() -> None:
    """_is_pandas_series returns False for pd.DataFrame."""
    assert _is_pandas_series(pd.DataFrame({"a": [1]})) is False


def test_is_pandas_frame_true() -> None:
    """_is_pandas_frame returns True for pd.DataFrame."""
    assert _is_pandas_frame(pd.DataFrame({"a": [1]})) is True


def test_is_pandas_frame_false_for_series() -> None:
    """_is_pandas_frame returns False for pd.Series."""
    assert _is_pandas_frame(pd.Series([1, 2])) is False


def test_is_polars_series_true() -> None:
    """_is_polars_series returns True for pl.Series."""
    assert _is_polars_series(pl.Series("a", [1, 2, 3])) is True


def test_is_polars_series_false_for_dataframe() -> None:
    """_is_polars_series returns False for pl.DataFrame."""
    assert _is_polars_series(pl.DataFrame({"a": [1]})) is False


def test_is_polars_frame_true() -> None:
    """_is_polars_frame returns True for pl.DataFrame."""
    assert _is_polars_frame(pl.DataFrame({"a": [1]})) is True


def test_is_polars_frame_false_for_series() -> None:
    """_is_polars_frame returns False for pl.Series."""
    assert _is_polars_frame(pl.Series("a", [1])) is False


def test_is_pandas_series_true_for_subclass() -> None:
    """_is_pandas_series detects subclasses of pd.Series."""

    class MySeries(pd.Series):
        pass

    assert _is_pandas_series(MySeries(["a", "b"])) is True


def test_is_pandas_frame_true_for_subclass() -> None:
    """_is_pandas_frame detects subclasses of pd.DataFrame."""

    class MyFrame(pd.DataFrame):
        pass

    assert _is_pandas_frame(MyFrame({"a": [1]})) is True


def test_is_numpy_array_true_for_subclass() -> None:
    """_is_numpy_array detects subclasses of np.ndarray (e.g. MaskedArray)."""
    masked = np.ma.array([1, 2, 3], mask=[False, True, False])
    assert _is_numpy_array(masked) is True
