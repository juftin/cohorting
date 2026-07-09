"""Tests for hash_values."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from cohorting._hash import hash_orm, hash_values


def test_hash_hashlib_known_value() -> None:
    """Default blake2b backend produces the expected deterministic float.

    Golden value: Blake2b512 (first 8 bytes as little-endian u64 / 2^64)
    over "user_123" + b"\\x00exp".
    """
    expected = 0.33078074404357033
    assert hash_values("user_123", salt="exp") == expected


def test_hash_xxhash_known_value() -> None:
    """xxhash backend produces the expected deterministic float.

    Golden value: xxh3_64 one-shot over "user_123" + b"\\x00exp", hash64 / 2^64.
    Both backends are compiled into the Rust extension — no importorskip needed.
    """
    from cohorting import config, hash_values

    config.xxhash = True
    try:
        result = hash_values("user_123", salt="exp")
    finally:
        config.xxhash = False

    assert result == 0.8382543487878418


def test_hash_single_string_returns_float() -> None:
    """hash_values on a string returns a float."""
    result = hash_values("user_123", salt="exp")
    assert isinstance(result, float)


def test_hash_single_string_in_range() -> None:
    """hash_values result is in [0, 1)."""
    result = hash_values("user_123", salt="exp")
    assert 0.0 <= result < 1.0


def test_hash_single_string_deterministic() -> None:
    """Same input and salt always produce the same hash."""
    assert hash_values("user_123", salt="exp") == hash_values("user_123", salt="exp")


def test_hash_salt_isolation() -> None:
    """Different salts produce different hashes for the same input."""
    assert hash_values("user_123", salt="exp_a") != hash_values(
        "user_123", salt="exp_b"
    )


def test_hash_list_returns_list() -> None:
    """hash_values on a list returns a list of floats."""
    result = hash_values(["a", "b", "c"], salt="exp")
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(v, float) for v in result)


def test_hash_list_values_in_range() -> None:
    """All list hash values are in [0, 1)."""
    result = hash_values(["a", "b", "c"], salt="exp")
    assert all(0.0 <= v < 1.0 for v in result)


def test_hash_list_consistent_with_scalar() -> None:
    """List hash results match individual scalar hashes."""
    result = hash_values(["x", "y"], salt="exp")
    assert result[0] == hash_values("x", salt="exp")
    assert result[1] == hash_values("y", salt="exp")


def test_hash_numpy_returns_ndarray() -> None:
    """hash_values on an ndarray returns an ndarray of floats."""
    np = pytest.importorskip("numpy")
    arr = np.array(["a", "b", "c"])
    result = hash_values(arr, salt="exp")
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert result.dtype == float


def test_hash_numpy_values_in_range() -> None:
    """All ndarray hash values are in [0, 1)."""
    np = pytest.importorskip("numpy")
    arr = np.array(["a", "b", "c"])
    result = hash_values(arr, salt="exp")
    assert np.all(result >= 0.0)
    assert np.all(result < 1.0)


def test_hash_numpy_consistent_with_scalar() -> None:
    """Numpy hash results match individual scalar hashes."""
    np = pytest.importorskip("numpy")
    arr = np.array(["x", "y"])
    result = hash_values(arr, salt="exp")
    assert result[0] == hash_values("x", salt="exp")
    assert result[1] == hash_values("y", salt="exp")


def test_hash_pandas_series_returns_series() -> None:
    """hash_values on a pd.Series returns a pd.Series."""
    pd = pytest.importorskip("pandas")
    series = pd.Series(["a", "b", "c"], name="user_id")
    result = hash_values(series, salt="exp")
    assert isinstance(result, pd.Series)
    assert result.name == "user_id"
    assert len(result) == 3


def test_hash_pandas_series_values_in_range() -> None:
    """All pandas hash values are in [0, 1)."""
    pd = pytest.importorskip("pandas")
    result = hash_values(pd.Series(["a", "b", "c"]), salt="exp")
    assert all(0.0 <= v < 1.0 for v in result)


def test_hash_pandas_series_preserves_index() -> None:
    """hash_values preserves the pandas Series index."""
    pd = pytest.importorskip("pandas")
    series = pd.Series(["a", "b"], index=[10, 20])
    result = hash_values(series, salt="exp")
    assert list(result.index) == [10, 20]


def test_hash_polars_series_returns_series() -> None:
    """hash_values on a pl.Series returns a pl.Series."""
    pl = pytest.importorskip("polars")
    series = pl.Series("user_id", ["a", "b", "c"])
    result = hash_values(series, salt="exp")
    assert isinstance(result, pl.Series)
    assert result.name == "user_id"
    assert len(result) == 3


def test_hash_polars_series_values_in_range() -> None:
    """All polars hash values are in [0, 1)."""
    pl = pytest.importorskip("polars")
    result = hash_values(pl.Series("x", ["a", "b", "c"]), salt="exp")
    assert all(0.0 <= v < 1.0 for v in result)


def test_hash_unsupported_type_raises() -> None:
    """Passing an unsupported type raises TypeError."""
    with pytest.raises(TypeError, match="hash_values expected"):
        hash_values({"a": 1}, salt="exp")  # type: ignore[call-overload]


def test_hash_int_scalar() -> None:
    """hash_values accepts an integer and returns a float in [0, 1)."""
    result = hash_values(12345, salt="exp")
    assert isinstance(result, float)
    assert 0.0 <= result < 1.0


def test_hash_int_equals_string() -> None:
    """hash_values(123) == hash_values("123") since int is str(int)-encoded."""
    assert hash_values(123, salt="exp") == hash_values("123", salt="exp")


def test_hash_int_list() -> None:
    """hash_values accepts a list of ints."""
    result = hash_values([1, 2, 3], salt="exp")
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(v, float) for v in result)


# --- hash_orm ---


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


def test_hash_orm_single() -> None:
    """hash_orm hashes the given attribute of a single object."""
    user = _User(user_id="user_123", name="Alice")
    assert hash_orm(user, id_field="user_id", salt="exp") == hash_values(
        "user_123", salt="exp"
    )


def test_hash_orm_list() -> None:
    """hash_orm on a list returns a list of floats."""
    users = [_User(user_id="x", name="X"), _User(user_id="y", name="Y")]
    result = hash_orm(users, id_field="user_id", salt="exp")
    assert isinstance(result, list)
    assert result == hash_values(["x", "y"], salt="exp")


def test_hash_orm_int_field() -> None:
    """hash_orm works when the id field is an int."""
    user = _IntUser(user_id=123, name="Alice")
    assert hash_orm(user, id_field="user_id", salt="exp") == hash_values(
        123, salt="exp"
    )


def test_hash_orm_missing_field_raises() -> None:
    """hash_orm raises AttributeError for an unknown field."""
    user = _User(user_id="user_123", name="Alice")
    with pytest.raises(AttributeError):
        hash_orm(user, id_field="nonexistent", salt="exp")


# --- non-deterministic mode ---


def test_hash_nondeterministic_scalar_in_range() -> None:
    """Non-deterministic hash returns a float in [0, 1)."""
    import cohorting

    cohorting.config.deterministic = False
    try:
        result = hash_values(data="user_1", salt="exp")
        assert isinstance(result, float)
        assert 0.0 <= result < 1.0
    finally:
        cohorting.config.deterministic = True


def test_hash_nondeterministic_differs_across_calls() -> None:
    """Two non-deterministic calls with the same input return different values."""
    import cohorting

    cohorting.config.deterministic = False
    try:
        r1 = hash_values(data="user_1", salt="exp")
        r2 = hash_values(data="user_1", salt="exp")
        assert r1 != r2
    finally:
        cohorting.config.deterministic = True


def test_hash_nondeterministic_ignores_random_seed() -> None:
    """Non-deterministic hash is unaffected by random.seed."""
    import random

    import cohorting

    cohorting.config.deterministic = False
    try:
        random.seed(0)
        r1 = hash_values(data="user_1", salt="exp")
        random.seed(0)
        r2 = hash_values(data="user_1", salt="exp")
        assert r1 != r2
    finally:
        cohorting.config.deterministic = True


def test_hash_nondeterministic_ignores_numpy_seed() -> None:
    """Non-deterministic hash is unaffected by numpy.random.seed."""
    np = pytest.importorskip("numpy")
    import cohorting

    cohorting.config.deterministic = False
    try:
        np.random.seed(0)
        r1 = hash_values(data="user_1", salt="exp")
        np.random.seed(0)
        r2 = hash_values(data="user_1", salt="exp")
        assert r1 != r2
    finally:
        cohorting.config.deterministic = True


def test_hash_nondeterministic_list() -> None:
    """Non-deterministic hash on a list returns floats in [0, 1)."""
    import cohorting

    cohorting.config.deterministic = False
    try:
        result = hash_values(data=["user_1", "user_2", "user_3"], salt="exp")
        assert isinstance(result, list)
        assert all(0.0 <= v < 1.0 for v in result)
    finally:
        cohorting.config.deterministic = True


def test_hash_deterministic_restored_after_toggle() -> None:
    """Deterministic results are correct after toggling back from non-deterministic."""
    import cohorting

    expected = hash_values(data="user_123", salt="exp")
    cohorting.config.deterministic = False
    cohorting.config.deterministic = True
    assert hash_values(data="user_123", salt="exp") == expected


# --- bool/int cache-key normalization ---


def test_hash_bool_true_equals_int_one() -> None:
    """hash_values(True) normalizes to int(1) and matches hash_values(1).

    bool is a subclass of int; True == 1 and hash(True) == hash(1), which means
    the LRU cache cannot distinguish them without explicit normalization. Without
    the fix, hash_values(True) returns the cached result for 1 (or vice versa),
    breaking determinism depending on call order.
    """
    from cohorting._hash import _cached_hash_single

    _cached_hash_single.cache_clear()
    h_int = hash_values(data=1, salt="exp")
    h_bool = hash_values(data=True, salt="exp")
    assert h_int == h_bool


def test_hash_bool_false_equals_int_zero() -> None:
    """hash_values(False) normalizes to int(0) and matches hash_values(0)."""
    from cohorting._hash import _cached_hash_single

    _cached_hash_single.cache_clear()
    assert hash_values(data=0, salt="exp") == hash_values(data=False, salt="exp")


# --- cache flag ---


def test_hash_cache_off_by_default() -> None:
    """config.cache is False by default."""
    import cohorting

    assert cohorting.config.cache is False


def test_hash_cache_on_returns_correct_value() -> None:
    """hash_values with config.cache=True returns the same deterministic value."""
    import cohorting

    expected = hash_values(data="user_123", salt="exp")
    cohorting.config.cache = True
    try:
        assert hash_values(data="user_123", salt="exp") == expected
    finally:
        cohorting.config.cache = False


def test_hash_cache_restored_after_toggle() -> None:
    """Results are unchanged after toggling cache on and back off."""
    import cohorting

    expected = hash_values(data="user_123", salt="exp")
    cohorting.config.cache = True
    cohorting.config.cache = False
    assert hash_values(data="user_123", salt="exp") == expected


# --- bool/int cache-key normalization ---


def test_hash_bool_cache_order_independent() -> None:
    """bool/int result is the same regardless of which is called first."""
    from cohorting._hash import _cached_hash_single

    _cached_hash_single.cache_clear()
    first_bool = hash_values(data=True, salt="exp")
    first_int = hash_values(data=1, salt="exp")

    _cached_hash_single.cache_clear()
    second_int = hash_values(data=1, salt="exp")
    second_bool = hash_values(data=True, salt="exp")

    assert first_bool == second_bool
    assert first_int == second_int
