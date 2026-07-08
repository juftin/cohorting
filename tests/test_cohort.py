"""Tests for assign_cohorts."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from cohorting._cohort import assign_cohorts, assign_cohorts_to_frame, assign_orm
from cohorting._models import SplitMap

FIFTY_FIFTY: SplitMap = {
    "control": {"lower": 0.0, "upper": 0.5},
    "treatment": {"lower": 0.5, "upper": 1.0},
}

ALL_IN_A: SplitMap = {"a": {"lower": 0.0, "upper": 1.0}}


def test_assign_single_string_returns_string() -> None:
    """assign_cohorts on a single string returns a cohort name string."""
    result = assign_cohorts("user_1", splits=ALL_IN_A, salt="exp")
    assert isinstance(result, str)
    assert result == "a"


def test_assign_deterministic() -> None:
    """Same input always maps to the same cohort."""
    result1 = assign_cohorts("user_1", splits=FIFTY_FIFTY, salt="exp")
    result2 = assign_cohorts("user_1", splits=FIFTY_FIFTY, salt="exp")
    assert result1 == result2


def test_assign_salt_isolation() -> None:
    """Different salts can produce different cohort assignments."""
    all_a = [
        assign_cohorts(f"user_{i}", splits=FIFTY_FIFTY, salt="exp_a") for i in range(20)
    ]
    all_b = [
        assign_cohorts(f"user_{i}", splits=FIFTY_FIFTY, salt="exp_b") for i in range(20)
    ]
    assert all_a != all_b


def test_assign_list_returns_list() -> None:
    """assign_cohorts on a list returns a list of cohort name strings."""
    result = assign_cohorts(["a", "b", "c"], splits=ALL_IN_A, salt="exp")
    assert isinstance(result, list)
    assert all(v == "a" for v in result)


def test_assign_list_correct_length() -> None:
    """List output has the same length as input."""
    result = assign_cohorts(["x", "y", "z"], splits=FIFTY_FIFTY, salt="exp")
    assert len(result) == 3


def test_assign_numpy_returns_ndarray() -> None:
    """assign_cohorts on an ndarray returns an ndarray of strings."""
    np = pytest.importorskip("numpy")
    arr = np.array(["a", "b", "c"])
    result = assign_cohorts(arr, splits=ALL_IN_A, salt="exp")
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


def test_assign_numpy_all_valid_cohort_names() -> None:
    """All numpy output values are valid cohort names from the split map."""
    np = pytest.importorskip("numpy")
    arr = np.array([f"user_{i}" for i in range(50)])
    result = assign_cohorts(arr, splits=FIFTY_FIFTY, salt="exp")
    assert set(result).issubset({"control", "treatment"})


def test_assign_invalid_splits_raises() -> None:
    """assign_cohorts raises ValueError if splits are invalid."""
    bad_splits: SplitMap = {
        "a": {"lower": 0.0, "upper": 0.4},
        "b": {"lower": 0.5, "upper": 1.0},
    }
    with pytest.raises(ValueError, match="gap"):
        assign_cohorts("user_1", splits=bad_splits, salt="exp")


def test_assign_unsupported_type_raises() -> None:
    """assign_cohorts raises TypeError for unsupported input types."""
    with pytest.raises(TypeError, match="assign_cohorts expected"):
        assign_cohorts({"a": 1}, splits=ALL_IN_A, salt="exp")  # type: ignore[call-overload]


def test_assign_int_scalar() -> None:
    """assign_cohorts accepts an integer and returns a cohort name."""
    result = assign_cohorts(12345, splits=ALL_IN_A, salt="exp")
    assert isinstance(result, str)
    assert result == "a"


def test_assign_int_equals_string() -> None:
    """assign_cohorts(123) and assign_cohorts("123") return the same cohort."""
    assert assign_cohorts(123, splits=ALL_IN_A, salt="exp") == assign_cohorts(
        "123", splits=ALL_IN_A, salt="exp"
    )


def test_assign_int_list() -> None:
    """assign_cohorts accepts a list of ints."""
    result = assign_cohorts([1, 2, 3], splits=ALL_IN_A, salt="exp")
    assert isinstance(result, list)
    assert all(v == "a" for v in result)


def test_assign_pandas_series_returns_series() -> None:
    """assign_cohorts on pd.Series returns pd.Series of cohort names."""
    pd = pytest.importorskip("pandas")
    series = pd.Series(["a", "b", "c"], name="user_id")
    result = assign_cohorts(series, splits=ALL_IN_A, salt="exp")
    assert isinstance(result, pd.Series)
    assert result.name == "user_id"
    assert list(result) == ["a", "a", "a"]


def test_assign_pandas_series_preserves_index() -> None:
    """assign_cohorts preserves the pandas Series index."""
    pd = pytest.importorskip("pandas")
    series = pd.Series(["x", "y"], index=[10, 20])
    result = assign_cohorts(series, splits=ALL_IN_A, salt="exp")
    assert list(result.index) == [10, 20]


def test_assign_polars_series_returns_series() -> None:
    """assign_cohorts on pl.Series returns pl.Series of cohort names."""
    pl = pytest.importorskip("polars")
    series = pl.Series("user_id", ["a", "b", "c"])
    result = assign_cohorts(series, splits=ALL_IN_A, salt="exp")
    assert isinstance(result, pl.Series)
    assert result.name == "user_id"
    assert list(result) == ["a", "a", "a"]


def test_assign_frame_pandas_returns_dataframe() -> None:
    """assign_cohorts_to_frame on pd.DataFrame returns a pd.DataFrame."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"user_id": ["a", "b", "c"], "score": [1, 2, 3]})
    result = assign_cohorts_to_frame(
        df, id_column="user_id", splits=ALL_IN_A, salt="exp"
    )
    assert isinstance(result, pd.DataFrame)
    assert "cohort" in result.columns


def test_assign_frame_pandas_default_output_column() -> None:
    """Default output column name is 'cohort'."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"user_id": ["a"]})
    result = assign_cohorts_to_frame(
        df, id_column="user_id", splits=ALL_IN_A, salt="exp"
    )
    assert "cohort" in result.columns


def test_assign_frame_pandas_custom_output_column() -> None:
    """Custom output column name is respected."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"user_id": ["a"]})
    result = assign_cohorts_to_frame(
        df,
        id_column="user_id",
        splits=ALL_IN_A,
        salt="exp",
        output_column="experiment_arm",
    )
    assert "experiment_arm" in result.columns


def test_assign_frame_pandas_does_not_mutate() -> None:
    """assign_cohorts_to_frame does not modify the original DataFrame."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"user_id": ["a", "b"]})
    original_columns = list(df.columns)
    assign_cohorts_to_frame(df, id_column="user_id", splits=ALL_IN_A, salt="exp")
    assert list(df.columns) == original_columns


def test_assign_frame_polars_returns_dataframe() -> None:
    """assign_cohorts_to_frame on pl.DataFrame returns a pl.DataFrame."""
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"user_id": ["a", "b", "c"], "score": [1, 2, 3]})
    result = assign_cohorts_to_frame(
        df, id_column="user_id", splits=ALL_IN_A, salt="exp"
    )
    assert isinstance(result, pl.DataFrame)
    assert "cohort" in result.columns


def test_assign_frame_polars_does_not_mutate() -> None:
    """assign_cohorts_to_frame does not modify the original polars DataFrame."""
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"user_id": ["a", "b"]})
    original_columns = list(df.columns)
    assign_cohorts_to_frame(df, id_column="user_id", splits=ALL_IN_A, salt="exp")
    assert list(df.columns) == original_columns


def test_assign_frame_unsupported_type_raises() -> None:
    """assign_cohorts_to_frame raises TypeError for unsupported input types."""
    with pytest.raises(TypeError, match="assign_cohorts_to_frame expected"):
        assign_cohorts_to_frame(
            {"user_id": ["a"]},  # type: ignore[arg-type]
            id_column="user_id",
            splits=ALL_IN_A,
            salt="exp",
        )


# --- assign_orm ---


@dataclass
class _User:
    """Test user dataclass."""

    user_id: str
    name: str


@dataclass
class _IntUser:
    """Test user dataclass with integer id."""

    user_id: int
    name: str


def test_assign_orm_single() -> None:
    """assign_orm assigns a cohort from the given attribute."""
    user = _User(user_id="user_123", name="Alice")
    assert assign_orm(user, id_field="user_id", splits=ALL_IN_A, salt="exp") == "a"


def test_assign_orm_list() -> None:
    """assign_orm on a list returns a list of cohort names."""
    users = [_User(user_id="x", name="X"), _User(user_id="y", name="Y")]
    result = assign_orm(users, id_field="user_id", splits=ALL_IN_A, salt="exp")
    assert isinstance(result, list)
    assert all(v == "a" for v in result)


def test_assign_orm_int_field() -> None:
    """assign_orm works when the id field is an int."""
    user = _IntUser(user_id=123, name="Alice")
    result = assign_orm(user, id_field="user_id", splits=ALL_IN_A, salt="exp")
    assert result == assign_cohorts(123, splits=ALL_IN_A, salt="exp")


def test_assign_orm_matches_assign_cohorts() -> None:
    """assign_orm produces the same result as assign_cohorts with the same id."""
    user = _User(user_id="user_123", name="Alice")
    assert assign_orm(
        user, id_field="user_id", splits=ALL_IN_A, salt="exp"
    ) == assign_cohorts("user_123", splits=ALL_IN_A, salt="exp")


def test_assign_orm_missing_field_raises() -> None:
    """assign_orm raises AttributeError for an unknown field."""
    user = _User(user_id="user_123", name="Alice")
    with pytest.raises(AttributeError):
        assign_orm(user, id_field="nonexistent", splits=ALL_IN_A, salt="exp")


# --- non-deterministic mode ---


def test_assign_nondeterministic_returns_valid_cohort() -> None:
    """Non-deterministic assign returns a valid cohort name."""
    import cohorting

    cohorting.config.deterministic = False
    try:
        result = assign_cohorts(data="user_1", splits=FIFTY_FIFTY, salt="exp")
        assert result in {"control", "treatment"}
    finally:
        cohorting.config.deterministic = True


def test_assign_nondeterministic_differs_across_calls() -> None:
    """Two non-deterministic calls with the same input return different cohorts."""
    import cohorting

    cohorting.config.deterministic = False
    try:
        results = {
            assign_cohorts(data="user_1", splits=FIFTY_FIFTY, salt="exp")
            for _ in range(20)
        }
        assert len(results) == 2
    finally:
        cohorting.config.deterministic = True


def test_assign_nondeterministic_ignores_random_seed() -> None:
    """Non-deterministic hashing is unaffected by random.seed.

    A seeded PRNG would produce the same float every time the seed is reset.
    os.urandom ignores the seed, so resetting it does not repeat results.
    """
    import random

    import cohorting
    from cohorting import hash_values

    cohorting.config.deterministic = False
    try:
        random.seed(0)
        h1 = hash_values(data="user_1", salt="exp")
        random.seed(0)
        h2 = hash_values(data="user_1", salt="exp")
        # A seeded PRNG would give h1 == h2 after resetting to seed 0; os.urandom won't
        assert h1 != h2
    finally:
        cohorting.config.deterministic = True


def test_assign_deterministic_restored_after_toggle() -> None:
    """Deterministic results are correct after toggling back from non-deterministic."""
    import cohorting

    expected = assign_cohorts(data="user_1", splits=FIFTY_FIFTY, salt="exp")
    cohorting.config.deterministic = False
    cohorting.config.deterministic = True
    assert assign_cohorts(data="user_1", splits=FIFTY_FIFTY, salt="exp") == expected
