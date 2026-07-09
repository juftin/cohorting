"""CodSpeed performance benchmarks for the cohorting hot paths.

These benchmarks exercise the performance-critical surface of the library:
deterministic hashing, single and batch cohort assignment via both the
functional API and the :class:`Experiment` class, and the vectorized
DataFrame / numpy backends.

Run locally with::

    codspeed run --mode simulation -- uv run pytest tests/test_benchmarks.py --codspeed
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from cohorting import (
    Experiment,
    assign_cohorts,
    assign_cohorts_to_frame,
    assign_orm,
    even_split,
    hash_values,
    validate_splits,
)

# A representative 3-way split and a batch of identifiers sized to reflect a
# realistic bulk-assignment workload without inflating benchmark runtime.
SPLITS = even_split(names=["control", "treatment", "holdout"])
IDS = [f"user_{i}" for i in range(10_000)]


@pytest.mark.benchmark
def test_hash_single() -> None:
    """Hash a single string identifier via the functional API."""
    hash_values("user_1234", salt="checkout")


@pytest.mark.benchmark
def test_hash_batch() -> None:
    """Hash a batch of string identifiers via the functional API."""
    hash_values(IDS, salt="checkout")


@pytest.mark.benchmark
def test_assign_single() -> None:
    """Assign a single identifier via the functional API (validates each call)."""
    assign_cohorts("user_1234", splits=SPLITS, salt="checkout")


@pytest.mark.benchmark
def test_assign_batch_list() -> None:
    """Assign a batch of identifiers via the functional API."""
    assign_cohorts(IDS, splits=SPLITS, salt="checkout")


@pytest.mark.benchmark
def test_experiment_construction() -> None:
    """Construct an Experiment (validates and sorts splits once)."""
    Experiment(name="checkout", splits=SPLITS)


@pytest.mark.benchmark
def test_experiment_assign_single() -> None:
    """Assign a single identifier through the Experiment class."""
    exp = Experiment(name="checkout", splits=SPLITS)
    exp.assign("user_1234")


@pytest.mark.benchmark
def test_experiment_assign_batch() -> None:
    """Assign a batch of identifiers through the Experiment class."""
    exp = Experiment(name="checkout", splits=SPLITS)
    exp.assign(IDS)


@pytest.mark.benchmark
def test_experiment_assign_batch_xxhash() -> None:
    """Assign a batch of identifiers using the xxhash backend."""
    exp = Experiment(name="checkout", splits=SPLITS, xxhash=True)
    exp.assign(IDS)


@pytest.mark.benchmark
def test_assign_numpy() -> None:
    """Assign a numpy array of identifiers via the functional API."""
    arr = np.array(IDS)
    assign_cohorts(arr, splits=SPLITS, salt="checkout")


@pytest.mark.benchmark
def test_assign_pandas_frame() -> None:
    """Add a cohort column to a pandas DataFrame."""
    df = pd.DataFrame({"user_id": IDS})
    assign_cohorts_to_frame(df, id_column="user_id", splits=SPLITS, salt="checkout")


@pytest.mark.benchmark
def test_assign_polars_frame() -> None:
    """Add a cohort column to a polars DataFrame."""
    df = pl.DataFrame({"user_id": IDS})
    assign_cohorts_to_frame(df, id_column="user_id", splits=SPLITS, salt="checkout")


@pytest.mark.benchmark
def test_assign_orm_batch() -> None:
    """Assign cohorts to a batch of model objects via attribute access."""

    class User:  # noqa: D106
        __slots__ = ("user_id",)

        def __init__(self, user_id: str) -> None:
            self.user_id = user_id

    users = [User(user_id=i) for i in IDS]
    assign_orm(users, id_field="user_id", splits=SPLITS, salt="checkout")


@pytest.mark.benchmark
def test_validate_splits() -> None:
    """Validate a split map spanning [0, 1)."""
    validate_splits(SPLITS)
