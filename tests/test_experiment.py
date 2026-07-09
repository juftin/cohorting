"""Tests for the Experiment class."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from cohorting._models import CohortSplit, SplitInput, SplitMap
from cohorting.experiment import Experiment

FIFTY_FIFTY: SplitMap = {
    "control": {"lower": 0.0, "upper": 0.5},
    "treatment": {"lower": 0.5, "upper": 1.0},
}

ALL_IN_A: SplitInput = {"a": {"lower": 0.0, "upper": 1.0}}


def test_experiment_constructs_with_dict_splits() -> None:
    """Experiment can be constructed directly with a SplitMap dict."""
    exp = Experiment(name="exp", salt="exp", splits=FIFTY_FIFTY)
    assert exp.name == "exp"
    assert exp.salt == "exp"
    assert exp.splits == FIFTY_FIFTY


def test_experiment_constructs_with_list_splits() -> None:
    """Experiment can be constructed with a list of CohortSplit instances."""
    splits: SplitInput = [
        CohortSplit(name="control", lower=0.0, upper=0.5),
        CohortSplit(name="treatment", lower=0.5, upper=1.0),
    ]
    exp = Experiment(name="exp", salt="exp", splits=splits)
    assert set(exp.splits.keys()) == {"control", "treatment"}


def test_experiment_validates_splits_on_construction() -> None:
    """Experiment raises ValueError on construction if splits are invalid."""
    bad_splits: SplitMap = {
        "a": {"lower": 0.0, "upper": 0.4},
        "b": {"lower": 0.5, "upper": 1.0},
    }
    with pytest.raises(ValueError, match="gap"):
        Experiment(name="exp", salt="exp", splits=bad_splits)


def test_experiment_hash_string() -> None:
    """Experiment.hash returns a float for a single string input."""
    exp = Experiment(name="exp", salt="exp", splits=FIFTY_FIFTY)
    result = exp.hash("user_123")
    assert isinstance(result, float)
    assert 0.0 <= result < 1.0


def test_experiment_hash_list() -> None:
    """Experiment.hash returns a list for list input."""
    exp = Experiment(name="exp", salt="exp", splits=FIFTY_FIFTY)
    result = exp.hash(["user_1", "user_2"])
    assert isinstance(result, list)
    assert len(result) == 2


def test_experiment_hash_numpy() -> None:
    """Experiment.hash returns an ndarray for ndarray input."""
    np = pytest.importorskip("numpy")
    exp = Experiment(name="exp", salt="exp", splits=FIFTY_FIFTY)
    arr = np.array(["user_1", "user_2"])
    result = exp.hash(arr)
    assert isinstance(result, np.ndarray)


def test_experiment_assign_string() -> None:
    """Experiment.assign returns a cohort name string for single string input."""
    exp = Experiment(name="exp", salt="exp", splits=ALL_IN_A)
    result = exp.assign("user_123")
    assert isinstance(result, str)
    assert result == "a"


def test_experiment_assign_list() -> None:
    """Experiment.assign returns a list of cohort names for list input."""
    exp = Experiment(name="exp", salt="exp", splits=ALL_IN_A)
    result = exp.assign(["x", "y", "z"])
    assert isinstance(result, list)
    assert all(v == "a" for v in result)


def test_experiment_assign_deterministic() -> None:
    """Experiment.assign is deterministic for the same identifier."""
    exp = Experiment(name="exp", salt="exp", splits=FIFTY_FIFTY)
    assert exp.assign("user_1") == exp.assign("user_1")


def test_experiment_assign_frame_pandas() -> None:
    """Experiment.assign_frame works with pd.DataFrame."""
    pd = pytest.importorskip("pandas")
    exp = Experiment(name="exp", salt="exp", splits=ALL_IN_A)
    df = pd.DataFrame({"user_id": ["a", "b"]})
    result = exp.assign_frame(df, id_column="user_id")
    assert isinstance(result, pd.DataFrame)
    assert "cohort" in result.columns


def test_experiment_assign_frame_custom_output_column() -> None:
    """Experiment.assign_frame respects custom output_column."""
    pd = pytest.importorskip("pandas")
    exp = Experiment(name="exp", salt="exp", splits=ALL_IN_A)
    df = pd.DataFrame({"user_id": ["a"]})
    result = exp.assign_frame(df, id_column="user_id", output_column="arm")
    assert "arm" in result.columns


def test_experiment_assign_frame_polars() -> None:
    """Experiment.assign_frame works with pl.DataFrame."""
    pl = pytest.importorskip("polars")
    exp = Experiment(name="exp", salt="exp", splits=ALL_IN_A)
    df = pl.DataFrame({"user_id": ["a", "b"]})
    result = exp.assign_frame(df, id_column="user_id")
    assert isinstance(result, pl.DataFrame)
    assert "cohort" in result.columns


def test_experiment_salt_defaults_to_name() -> None:
    """Experiment.salt defaults to name when not provided."""
    exp = Experiment(name="exp", splits=ALL_IN_A)
    assert exp.salt == "exp"


def test_experiment_explicit_salt_overrides_name() -> None:
    """Experiment.salt uses the explicit value when provided."""
    exp = Experiment(name="exp", salt="custom-salt", splits=ALL_IN_A)
    assert exp.salt == "custom-salt"


def test_experiment_equality() -> None:
    """Two Experiments with the same config are equal."""
    a = Experiment(name="exp", salt="exp", splits=ALL_IN_A)
    b = Experiment(name="exp", salt="exp", splits=ALL_IN_A)
    assert a == b


def test_experiment_repr() -> None:
    """Experiment repr includes name, salt, and splits."""
    exp = Experiment(name="exp", salt="exp", splits=ALL_IN_A)
    r = repr(exp)
    assert "exp" in r
    assert "splits" in r


def test_experiment_hash_golden_value() -> None:
    """Experiment.hash produces the xxh3_64 golden value."""
    exp = Experiment(name="exp", salt="exp", splits=ALL_IN_A)
    assert exp.hash("user_123") == pytest.approx(0.8382543487878418)


# --- ORM methods ---


@dataclass
class _User:
    user_id: str
    name: str


def test_assign_orm_single() -> None:
    """assign_orm assigns a cohort from the given attribute."""
    exp = Experiment(name="exp", splits=ALL_IN_A)
    user = _User(user_id="user_123", name="Alice")
    assert exp.assign_orm(user, id_field="user_id") == "a"


def test_assign_orm_list() -> None:
    """assign_orm on a list returns a list of cohort names."""
    exp = Experiment(name="exp", splits=ALL_IN_A)
    users = [_User(user_id="x", name="X"), _User(user_id="y", name="Y")]
    result = exp.assign_orm(users, id_field="user_id")
    assert isinstance(result, list)
    assert all(v == "a" for v in result)


def test_hash_orm_single() -> None:
    """hash_orm hashes the given attribute."""
    exp = Experiment(name="exp", salt="exp", splits=ALL_IN_A)
    user = _User(user_id="user_123", name="Alice")
    assert exp.hash_orm(user, id_field="user_id") == exp.hash("user_123")


def test_hash_orm_list() -> None:
    """hash_orm on a list returns a list of floats."""
    exp = Experiment(name="exp", splits=ALL_IN_A)
    users = [_User(user_id="x", name="X"), _User(user_id="y", name="Y")]
    result = exp.hash_orm(users, id_field="user_id")
    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


def test_assign_orm_missing_field_raises() -> None:
    """assign_orm raises AttributeError for an unknown field."""
    exp = Experiment(name="exp", splits=ALL_IN_A)
    user = _User(user_id="user_123", name="Alice")
    with pytest.raises(AttributeError):
        exp.assign_orm(user, id_field="nonexistent")


def test_experiment_hash_int_scalar() -> None:
    """Experiment.hash accepts an integer and returns a float."""
    exp = Experiment(name="exp", salt="exp", splits=ALL_IN_A)
    result = exp.hash(12345)
    assert isinstance(result, float)
    assert 0.0 <= result < 1.0


def test_experiment_hash_int_equals_string() -> None:
    """Experiment.hash(123) == Experiment.hash("123")."""
    exp = Experiment(name="exp", salt="exp", splits=ALL_IN_A)
    assert exp.hash(123) == exp.hash("123")


def test_experiment_assign_int_scalar() -> None:
    """Experiment.assign accepts an integer and returns a cohort name."""
    exp = Experiment(name="exp", salt="exp", splits=ALL_IN_A)
    assert exp.assign(12345) == "a"


def test_experiment_assign_int_equals_string() -> None:
    """Experiment.assign(123) and Experiment.assign("123") return the same cohort."""
    exp = Experiment(name="exp", salt="exp", splits=ALL_IN_A)
    assert exp.assign(123) == exp.assign("123")


def test_experiment_assign_int_list() -> None:
    """Experiment.assign accepts a list of ints."""
    exp = Experiment(name="exp", salt="exp", splits=ALL_IN_A)
    result = exp.assign([1, 2, 3])
    assert isinstance(result, list)
    assert all(v == "a" for v in result)


@dataclass
class _IntUser:
    user_id: int
    name: str


def test_assign_orm_int_field() -> None:
    """assign_orm works when the id field is an int."""
    exp = Experiment(name="exp", splits=ALL_IN_A)
    user = _IntUser(user_id=123, name="Alice")
    assert exp.assign_orm(user, id_field="user_id") == exp.assign(123)


def test_hash_orm_int_field() -> None:
    """hash_orm works when the id field is an int."""
    exp = Experiment(name="exp", splits=ALL_IN_A)
    user = _IntUser(user_id=123, name="Alice")
    assert exp.hash_orm(user, id_field="user_id") == exp.hash(123)


def test_experiment_cohorts_order() -> None:
    """Experiment.cohorts returns cohort names sorted by lower bound."""
    exp = Experiment(name="exp", salt="exp", splits=FIFTY_FIFTY)
    assert exp.cohorts == ["control", "treatment"]


def test_experiment_cohorts_three_way() -> None:
    """Experiment.cohorts returns all three cohort names in order."""
    splits: SplitMap = {
        "c": {"lower": 2 / 3, "upper": 1.0},
        "a": {"lower": 0.0, "upper": 1 / 3},
        "b": {"lower": 1 / 3, "upper": 2 / 3},
    }
    exp = Experiment(name="exp", splits=splits)
    assert exp.cohorts == ["a", "b", "c"]


def test_experiment_cohorts_single() -> None:
    """Experiment.cohorts returns a one-element list for a single cohort."""
    exp = Experiment(name="exp", splits=ALL_IN_A)
    assert exp.cohorts == ["a"]


def test_experiment_nondeterministic_returns_valid_cohort() -> None:
    """Experiment(deterministic=False) assigns a valid cohort on each call."""
    exp = Experiment(name="exp", splits=FIFTY_FIFTY, deterministic=False)
    result = exp.assign(data="user_1")
    assert result in {"control", "treatment"}


def test_experiment_nondeterministic_differs_across_calls() -> None:
    """Experiment(deterministic=False) produces different cohorts for the same input."""
    exp = Experiment(name="exp", splits=FIFTY_FIFTY, deterministic=False)
    results = {exp.assign(data="user_1") for _ in range(20)}
    assert len(results) == 2


def test_experiment_nondeterministic_hash_in_range() -> None:
    """Experiment(deterministic=False).hash returns a float in [0, 1)."""
    exp = Experiment(name="exp", splits=FIFTY_FIFTY, deterministic=False)
    h = exp.hash(data="user_1")
    assert isinstance(h, float)
    assert 0.0 <= h < 1.0


def test_experiment_nondeterministic_repr() -> None:
    """Experiment(deterministic=False) shows deterministic=False in repr."""
    exp = Experiment(name="exp", splits=FIFTY_FIFTY, deterministic=False)
    assert "deterministic=False" in repr(exp)


# --- cache=False fast path equivalence ---
#
# The use_cache=False branch in _dispatch_assign uses an inlined bisect rather
# than calling _lookup_cohort. These tests confirm the two paths produce
# identical assignments across every vectorised input type.

_IDS = [f"user_{i}" for i in range(200)]


def test_cache_false_matches_cache_true_list() -> None:
    """cache=False list path produces identical assignments to cache=True."""
    exp_no = Experiment(name="eq", splits=FIFTY_FIFTY, cache=False)
    exp_yes = Experiment(name="eq", splits=FIFTY_FIFTY, cache=True)
    assert exp_no.assign(_IDS) == exp_yes.assign(_IDS)


def test_cache_false_matches_cache_true_numpy() -> None:
    """cache=False ndarray lambda produces identical assignments to cache=True."""
    np = pytest.importorskip("numpy")
    arr = np.array(_IDS)
    exp_no = Experiment(name="eq", splits=FIFTY_FIFTY, cache=False)
    exp_yes = Experiment(name="eq", splits=FIFTY_FIFTY, cache=True)
    assert list(exp_no.assign(arr)) == list(exp_yes.assign(arr))


def test_cache_false_matches_cache_true_pandas() -> None:
    """cache=False pd.Series lambda produces identical assignments to cache=True."""
    pd = pytest.importorskip("pandas")
    series = pd.Series(_IDS)
    exp_no = Experiment(name="eq", splits=FIFTY_FIFTY, cache=False)
    exp_yes = Experiment(name="eq", splits=FIFTY_FIFTY, cache=True)
    assert list(exp_no.assign(series)) == list(exp_yes.assign(series))


def test_cache_false_matches_cache_true_polars() -> None:
    """cache=False pl.Series lambda produces identical assignments to cache=True."""
    pl = pytest.importorskip("polars")
    series = pl.Series(_IDS)
    exp_no = Experiment(name="eq", splits=FIFTY_FIFTY, cache=False)
    exp_yes = Experiment(name="eq", splits=FIFTY_FIFTY, cache=True)
    assert list(exp_no.assign(series)) == list(exp_yes.assign(series))


def test_experiment_deterministic_default_repr() -> None:
    """Experiment with default determinism does not show deterministic in repr."""
    exp = Experiment(name="exp", splits=FIFTY_FIFTY)
    assert "deterministic" not in repr(exp)


def test_experiment_nondeterministic_not_equal_to_deterministic() -> None:
    """Deterministic and non-deterministic Experiments with same config differ."""
    exp_det = Experiment(name="exp", splits=FIFTY_FIFTY)
    exp_rand = Experiment(name="exp", splits=FIFTY_FIFTY, deterministic=False)
    assert exp_det != exp_rand


# --- cache flag ---


def test_experiment_cache_default_off() -> None:
    """Experiment cache defaults to False."""
    exp = Experiment(name="exp", splits=FIFTY_FIFTY)
    assert exp._cache is False


def test_experiment_cache_returns_correct_cohort() -> None:
    """Experiment with cache=True returns the same cohort as cache=False."""
    exp_no_cache = Experiment(name="exp", splits=FIFTY_FIFTY)
    exp_cache = Experiment(name="exp", splits=FIFTY_FIFTY, cache=True)
    for uid in ["user_1", "user_2", "user_3"]:
        assert exp_no_cache.assign(data=uid) == exp_cache.assign(data=uid)


def test_experiment_cache_repr() -> None:
    """Experiment with cache=True shows cache=True in repr."""
    exp = Experiment(name="exp", splits=FIFTY_FIFTY, cache=True)
    assert "cache=True" in repr(exp)


def test_experiment_cache_default_not_in_repr() -> None:
    """Experiment with default cache=False does not show cache in repr."""
    exp = Experiment(name="exp", splits=FIFTY_FIFTY)
    assert "cache" not in repr(exp)


def test_experiment_cache_not_equal_to_nocache() -> None:
    """Experiments that differ only in cache flag are not equal."""
    exp_no_cache = Experiment(name="exp", splits=FIFTY_FIFTY)
    exp_cache = Experiment(name="exp", splits=FIFTY_FIFTY, cache=True)
    assert exp_no_cache != exp_cache
