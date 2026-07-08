"""Smoke tests for the public API surface."""

from __future__ import annotations


def test_public_imports_available() -> None:
    """All public symbols are importable from the top-level package."""
    from cohorting import (
        CohortBounds,
        CohortSplit,
        Experiment,
        SplitInput,
        SplitMap,
        assign_cohorts,
        assign_cohorts_to_frame,
        even_split,
        hash_values,
        validate_splits,
        weighted_split,
    )

    assert CohortBounds is not None
    assert CohortSplit is not None
    assert Experiment is not None
    assert SplitInput is not None
    assert SplitMap is not None
    assert assign_cohorts is not None
    assert assign_cohorts_to_frame is not None
    assert even_split is not None
    assert hash_values is not None
    assert validate_splits is not None
    assert weighted_split is not None


def test_end_to_end_string() -> None:
    """End-to-end: hash and assign a single string via public API."""
    from cohorting import SplitMap, assign_cohorts, hash_values

    splits: SplitMap = {
        "control": {"lower": 0.0, "upper": 0.5},
        "treatment": {"lower": 0.5, "upper": 1.0},
    }
    h = hash_values("user_1", salt="exp")
    assert 0.0 <= h < 1.0
    cohort = assign_cohorts("user_1", splits=splits, salt="exp")
    assert cohort in {"control", "treatment"}


def test_end_to_end_experiment() -> None:
    """End-to-end: use Experiment class via public API."""
    from cohorting import Experiment

    exp = Experiment(
        name="my_experiment",
        salt="my_salt",
        splits={"a": {"lower": 0.0, "upper": 1.0}},
    )
    assert exp.assign("any_user") == "a"


def test_end_to_end_experiment_with_cohort_splits() -> None:
    """End-to-end: Experiment accepts list[CohortSplit] directly."""
    from cohorting import CohortSplit, Experiment

    exp = Experiment(
        name="my_experiment",
        salt="my_salt",
        splits=[
            CohortSplit(name="a", lower=0.0, upper=1.0),
        ],
    )
    assert exp.assign("any_user") == "a"
