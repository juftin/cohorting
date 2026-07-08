"""Tests for split map validation."""

import pytest

from cohorting._models import SplitMap
from cohorting._validate import validate_splits


def test_valid_two_cohort_split() -> None:
    """Two contiguous cohorts spanning [0, 1) pass without error."""
    splits: SplitMap = {
        "control": {"lower": 0.0, "upper": 0.5},
        "treatment": {"lower": 0.5, "upper": 1.0},
    }
    validate_splits(splits)  # must not raise


def test_valid_three_cohort_split() -> None:
    """Three contiguous cohorts spanning [0, 1) pass without error."""
    splits: SplitMap = {
        "a": {"lower": 0.0, "upper": 0.33},
        "b": {"lower": 0.33, "upper": 0.66},
        "c": {"lower": 0.66, "upper": 1.0},
    }
    validate_splits(splits)  # must not raise


def test_valid_single_cohort() -> None:
    """A single cohort covering the full [0, 1) range is valid."""
    validate_splits({"all": {"lower": 0.0, "upper": 1.0}})


def test_empty_splits_raises() -> None:
    """Empty splits dict raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        validate_splits({})


def test_does_not_start_at_zero_raises() -> None:
    """Splits starting above 0.0 raise ValueError."""
    with pytest.raises(ValueError, match="0.0"):
        validate_splits({"a": {"lower": 0.1, "upper": 1.0}})


def test_does_not_end_at_one_raises() -> None:
    """Splits ending below 1.0 raise ValueError."""
    with pytest.raises(ValueError, match="1.0"):
        validate_splits({"a": {"lower": 0.0, "upper": 0.9}})


def test_gap_between_cohorts_raises() -> None:
    """A gap between adjacent cohorts raises ValueError mentioning 'gap'."""
    splits: SplitMap = {
        "control": {"lower": 0.0, "upper": 0.4},
        "treatment": {"lower": 0.5, "upper": 1.0},
    }
    with pytest.raises(ValueError, match="gap"):
        validate_splits(splits)


def test_overlap_between_cohorts_raises() -> None:
    """An overlap between adjacent cohorts raises ValueError mentioning 'overlap'."""
    splits: SplitMap = {
        "control": {"lower": 0.0, "upper": 0.6},
        "treatment": {"lower": 0.5, "upper": 1.0},
    }
    with pytest.raises(ValueError, match="overlap"):
        validate_splits(splits)


def test_error_message_names_cohorts() -> None:
    """Error messages include the names of the offending cohorts."""
    splits: SplitMap = {
        "control": {"lower": 0.0, "upper": 0.4},
        "treatment": {"lower": 0.5, "upper": 1.0},
    }
    with pytest.raises(ValueError, match="control"):
        validate_splits(splits)


def test_inverted_bounds_raises() -> None:
    """A cohort with lower > upper raises ValueError."""
    splits: SplitMap = {
        "a": {"lower": 0.0, "upper": 1.5},
        "b": {"lower": 1.5, "upper": 1.0},
    }
    with pytest.raises(ValueError, match="inverted"):
        validate_splits(splits)


def test_zero_width_bounds_raises() -> None:
    """A cohort with lower == upper (zero-width) raises ValueError."""
    splits: SplitMap = {
        "a": {"lower": 0.0, "upper": 1.0},
        "b": {"lower": 1.0, "upper": 1.0},
    }
    with pytest.raises(ValueError, match="zero-width"):
        validate_splits(splits)


def test_nan_bounds_raises() -> None:
    """A cohort with NaN bounds raises ValueError."""
    import math

    splits: SplitMap = {
        "a": {"lower": 0.0, "upper": math.nan},
        "b": {"lower": math.nan, "upper": 1.0},
    }
    with pytest.raises(ValueError, match="NaN"):
        validate_splits(splits)
