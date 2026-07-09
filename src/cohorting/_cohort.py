"""Cohort assignment functions."""

from __future__ import annotations

import bisect
from functools import cache, lru_cache
from typing import TYPE_CHECKING, Any, cast, overload

import cohorting._hash as _hash_mod
from cohorting._core import (
    assign_single as _rust_assign_single,
)
from cohorting._core import (
    assign_strings as _rust_assign_strings,
)
from cohorting._core import (
    random_float as _rust_random_float,
)
from cohorting._core import (
    random_floats as _rust_random_floats,
)
from cohorting._models import (
    SplitInput,
    SplitMap,
    _is_numpy_array,
    _is_pandas_frame,
    _is_pandas_series,
    _is_polars_frame,
    _is_polars_series,
    _normalize_splits,
)
from cohorting._validate import validate_splits

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    import polars as pl

_SortedBounds = tuple[tuple[str, float, float], ...]
"""Pre-sorted cohort bounds: (name, lower, upper) tuples ordered by lower bound."""


def _splits_to_sorted_bounds(splits: SplitMap) -> _SortedBounds:
    """Convert a SplitMap to a sorted tuple for bisect-based assignment.

    Parameters
    ----------
    splits : SplitMap
        Cohort split map.

    Returns
    -------
    _SortedBounds
        Tuple of (name, lower, upper) sorted by lower bound.
    """
    return tuple(
        sorted(
            ((name, b["lower"], b["upper"]) for name, b in splits.items()),
            key=lambda t: t[1],
        )
    )


@cache
def _get_lower_bounds(sorted_bounds: _SortedBounds) -> tuple[float, ...]:
    """Extract lower bounds for bisect lookup, cached per unique experiment config.

    Parameters
    ----------
    sorted_bounds : _SortedBounds
        Pre-sorted cohort bounds.

    Returns
    -------
    tuple[float, ...]
        Lower bound values in sorted order.
    """
    return tuple(b[1] for b in sorted_bounds)


@cache
def _get_cohort_names(sorted_bounds: _SortedBounds) -> tuple[str, ...]:
    """Extract cohort names in sorted order, cached per unique experiment config.

    Parameters
    ----------
    sorted_bounds : _SortedBounds
        Pre-sorted cohort bounds.

    Returns
    -------
    tuple[str, ...]
        Cohort names in sorted order.
    """
    return tuple(b[0] for b in sorted_bounds)


@lru_cache(maxsize=65_536)
def _cached_assign_single(
    x: str, sep_salt: bytes, use_xxhash: bool, sorted_bounds: _SortedBounds
) -> str:
    """Cached hash + assign via Rust.

    Parameters
    ----------
    x : str
        Identifier.
    sep_salt : bytes
        Pre-encoded b"\\x00" + salt bytes.
    use_xxhash : bool
        True for xxhash backend.
    sorted_bounds : _SortedBounds
        Pre-sorted cohort bounds.

    Returns
    -------
    str
        Cohort name.
    """
    lowers = _get_lower_bounds(sorted_bounds)
    names = _get_cohort_names(sorted_bounds)
    return _rust_assign_single(x, sep_salt, use_xxhash, list(names), list(lowers))


def _dispatch_assign(
    data: Any,
    *,
    sep_salt: bytes,
    sorted_bounds: _SortedBounds,
    use_xxhash: bool,
    use_deterministic: bool,
    use_cache: bool,
) -> Any:
    """Apply cohort assignment to all supported data types.

    Parameters
    ----------
    data : str | int | list[str | int] | np.ndarray | pd.Series | pl.Series
        Input identifiers. Integers are accepted alongside strings.
    sep_salt : bytes
        Pre-encoded b"\\x00" + salt bytes. Unused when ``use_deterministic=False``.
    sorted_bounds : _SortedBounds
        Pre-sorted, validated cohort bounds.
    use_xxhash : bool
        True for xxhash; False for hashlib. Unused when ``use_deterministic=False``.
    use_deterministic : bool
        False to bypass hashing and assign each identifier to a randomly chosen cohort
        using OS entropy.
    use_cache : bool
        True to use LRU-cached assign functions.

    Returns
    -------
    str | list[str] | np.ndarray | pd.Series | pl.Series
        Cohort name(s). Return type mirrors input type.

    Raises
    ------
    TypeError
        If data is not one of the supported input types.
    """
    if not use_deterministic:
        _lowers = _get_lower_bounds(sorted_bounds)
        _names_list = list(_get_cohort_names(sorted_bounds))

        def _random_assign(hash_val: float) -> str:
            idx = bisect.bisect_right(_lowers, hash_val) - 1
            return _names_list[idx]

        if isinstance(data, (str, int)):
            return _random_assign(_rust_random_float())
        if isinstance(data, list):
            return [_random_assign(_rust_random_float()) for _ in data]
        if _is_numpy_array(data):
            import numpy as _np

            floats = _rust_random_floats(data.size)
            names_arr = _np.array(_names_list)
            indices = _np.searchsorted(_lowers, floats, side="right") - 1
            return names_arr[indices].reshape(data.shape)
        if _is_pandas_series(data):
            import numpy as _np
            import pandas as _pd

            floats = _rust_random_floats(len(data))
            names_arr = _np.array(_names_list)
            indices = _np.searchsorted(_lowers, floats, side="right") - 1
            return _pd.Series(names_arr[indices], index=data.index, name=data.name)
        if _is_polars_series(data):
            import polars as _pl

            floats = _rust_random_floats(len(data))
            result = [_names_list[bisect.bisect_right(_lowers, f) - 1] for f in floats]
            return _pl.Series(name=data.name, values=result)
        raise TypeError(
            f"assign_cohorts expected str, list, np.ndarray, pd.Series, or pl.Series; "
            f"got {type(data).__name__}"
        )

    lowers = _get_lower_bounds(sorted_bounds)
    names = _get_cohort_names(sorted_bounds)

    if isinstance(data, (str, int)):
        norm: str | int = int(data) if isinstance(data, bool) else data
        norm_str = str(norm)
        if use_cache:
            return _cached_assign_single(norm_str, sep_salt, use_xxhash, sorted_bounds)
        return _rust_assign_single(
            norm_str, sep_salt, use_xxhash, list(names), list(lowers)
        )

    if isinstance(data, list):
        norm_list = [str(int(x) if isinstance(x, bool) else x) for x in data]
        if use_cache:
            return [
                _cached_assign_single(x, sep_salt, use_xxhash, sorted_bounds)
                for x in norm_list
            ]
        return _rust_assign_strings(
            norm_list, sep_salt, use_xxhash, list(names), list(lowers)
        )

    if _is_numpy_array(data):
        str_ids = [str(int(x) if isinstance(x, bool) else x) for x in data.flat]
        result = _rust_assign_strings(
            str_ids, sep_salt, use_xxhash, list(names), list(lowers)
        )
        import numpy as _np

        return _np.array(result).reshape(data.shape)

    if _is_pandas_series(data):
        import pandas as _pd

        str_ids = [str(int(x) if isinstance(x, bool) else x) for x in data]
        result = _rust_assign_strings(
            str_ids, sep_salt, use_xxhash, list(names), list(lowers)
        )
        return _pd.Series(result, index=data.index, name=data.name)

    if _is_polars_series(data):
        import polars as _pl

        str_ids = [str(int(x) if isinstance(x, bool) else x) for x in data]
        result = _rust_assign_strings(
            str_ids, sep_salt, use_xxhash, list(names), list(lowers)
        )
        return _pl.Series(name=data.name, values=result)

    raise TypeError(
        f"assign_cohorts expected str, list, np.ndarray, pd.Series, or pl.Series; "
        f"got {type(data).__name__}"
    )


@overload
def assign_cohorts(data: str, *, splits: SplitInput, salt: str) -> str: ...


@overload
def assign_cohorts(data: int, *, splits: SplitInput, salt: str) -> str: ...


@overload
def assign_cohorts(data: list[str], *, splits: SplitInput, salt: str) -> list[str]: ...


@overload
def assign_cohorts(data: list[int], *, splits: SplitInput, salt: str) -> list[str]: ...


@overload
def assign_cohorts(
    data: npt.NDArray[Any], *, splits: SplitInput, salt: str
) -> npt.NDArray[np.str_]: ...


@overload
def assign_cohorts(data: pd.Series, *, splits: SplitInput, salt: str) -> pd.Series: ...


@overload
def assign_cohorts(data: pl.Series, *, splits: SplitInput, salt: str) -> pl.Series: ...


def assign_cohorts(data: Any, *, splits: SplitInput, salt: str) -> Any:
    """Assign cohort names to identifiers using reproducible hashing.

    Normalizes and validates splits on every call. For repeated assignments
    against the same experiment configuration, prefer :class:`cohorting.Experiment`
    — it validates and sorts splits once at construction for better performance.

    Integer identifiers are converted to their decimal string representation before
    hashing, so ``assign_cohorts(123, ...)`` and ``assign_cohorts("123", ...)``
    produce the same cohort.

    Parameters
    ----------
    data : str | int | list[str] | list[int] | np.ndarray | pd.Series | pl.Series
        Identifier(s) to assign. Strings and integers are both accepted.
    splits : SplitInput
        Cohort splits as a SplitMap dict or a list of CohortSplit instances.
        Must span exactly [0, 1) with no gaps or overlaps.
    salt : str
        Appended to each identifier before hashing. Use a unique experiment name.

    Returns
    -------
    str | list[str] | np.ndarray | pd.Series | pl.Series
        Cohort name(s). Return type mirrors the input type.

    Raises
    ------
    ValueError
        If splits are invalid (delegated to validate_splits).
    TypeError
        If data is not one of the supported input types.

    Notes
    -----
    For repeated assignments against the same experiment configuration, prefer
    :class:`cohorting.Experiment` — it validates and sorts splits once at
    construction rather than on every call.

    Examples
    --------
    >>> from cohorting import assign_cohorts, even_split
    >>> splits = even_split(names=["control", "treatment"])
    >>> assign_cohorts(data="user_1", splits=splits, salt="exp")
    'control'
    >>> assign_cohorts(data=123, splits=splits, salt="exp") == assign_cohorts(
    ...     data="123", splits=splits, salt="exp"
    ... )
    True
    >>> assign_cohorts(data=["u1", "u2", "u3"], splits=splits, salt="exp")
    ['treatment', 'treatment', 'control']
    """
    split_map = _normalize_splits(splits)
    validate_splits(split_map)
    sep_salt = (
        b"\x00" + salt.encode()
    )  # \x00 separates identifier from salt; see Experiment.__init__
    sorted_bounds = _splits_to_sorted_bounds(split_map)
    return _dispatch_assign(
        data,
        sep_salt=sep_salt,
        sorted_bounds=sorted_bounds,
        use_xxhash=_hash_mod._USE_XXHASH,
        use_deterministic=_hash_mod._USE_DETERMINISTIC,
        use_cache=_hash_mod._USE_CACHE,
    )


def assign_cohorts_to_frame(
    df: pd.DataFrame | pl.DataFrame,
    *,
    id_column: str,
    splits: SplitInput,
    salt: str,
    output_column: str = "cohort",
) -> pd.DataFrame | pl.DataFrame:
    """Add a cohort assignment column to a pandas or polars DataFrame.

    Normalizes and validates splits on every call. For repeated assignments,
    prefer :meth:`cohorting.Experiment.assign_frame` — it validates once at
    construction.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame
        Input DataFrame. Not mutated.
    id_column : str
        Name of the column containing string identifiers to hash.
    splits : SplitInput
        Cohort splits as a SplitMap dict or a list of CohortSplit instances.
        Must span exactly [0, 1) with no gaps or overlaps.
    salt : str
        Appended to each identifier before hashing. Use a unique experiment name.
    output_column : str, optional
        Name of the column to add with cohort assignments. Default is "cohort".

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        A copy of the input DataFrame with the cohort column added.

    Raises
    ------
    ValueError
        If splits are invalid (delegated to validate_splits).
    TypeError
        If df is not a pandas or polars DataFrame.

    Notes
    -----
    Both pandas and polars DataFrames are supported. The input DataFrame is not
    mutated; a new DataFrame with the cohort column appended is returned.

    Examples
    --------
    >>> import pandas as pd
    >>> from cohorting import assign_cohorts_to_frame, even_split
    >>> splits = even_split(names=["control", "treatment"])
    >>> df = pd.DataFrame({"user_id": ["user_1", "user_2", "user_3"]})
    >>> assign_cohorts_to_frame(df, id_column="user_id", splits=splits, salt="exp")
      user_id     cohort
    0  user_1    control
    1  user_2    control
    2  user_3  treatment
    """
    if _is_pandas_frame(df):
        import pandas as _pd

        pd_cohorts: _pd.Series = assign_cohorts(df[id_column], splits=splits, salt=salt)
        return df.assign(**{output_column: pd_cohorts})

    if _is_polars_frame(df):
        import polars as _pl

        pl_cohorts: _pl.Series = assign_cohorts(df[id_column], splits=splits, salt=salt)
        return df.with_columns(pl_cohorts.alias(output_column))

    raise TypeError(
        f"assign_cohorts_to_frame expected pd.DataFrame or pl.DataFrame; "
        f"got {type(df).__name__}"
    )


def assign_orm(
    obj: object | list[object],
    *,
    id_field: str,
    splits: SplitInput,
    salt: str,
) -> str | list[str]:
    """Assign a cohort to a dataclass, Pydantic model, or plain object.

    Functional counterpart to :meth:`cohorting.Experiment.assign_orm`. Normalizes,
    validates, and re-encodes splits on every call; prefer
    :class:`cohorting.Experiment` for repeated calls against the same experiment
    configuration.

    Parameters
    ----------
    obj : object | list[object]
        A single model instance or a list of model instances.
    id_field : str
        Attribute name to read the identifier from. The attribute may be a
        ``str`` or ``int``.
    splits : SplitInput
        Cohort splits as a SplitMap dict or a list of CohortSplit instances.
        Must span exactly [0, 1) with no gaps or overlaps.
    salt : str
        Appended to the identifier before hashing. Use a unique experiment name.

    Returns
    -------
    str | list[str]
        Cohort name(s).

    Raises
    ------
    AttributeError
        If obj does not have the given attribute.
    ValueError
        If splits are invalid (delegated to validate_splits).

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> from cohorting import assign_cohorts, assign_orm, even_split
    >>> @dataclass
    ... class User:
    ...     user_id: str
    >>> splits = even_split(names=["control", "treatment"])
    >>> assign_orm(User(user_id="user_1"), id_field="user_id",
    ...             splits=splits, salt="exp")
    'control'
    >>> assign_orm(User(user_id="user_1"), id_field="user_id",
    ...             splits=splits, salt="exp") == assign_cohorts(
    ...                 data="user_1", splits=splits, salt="exp"
    ...             )
    True
    """
    if isinstance(obj, list):
        return assign_cohorts(
            [cast(str, getattr(o, id_field)) for o in obj], splits=splits, salt=salt
        )
    return assign_cohorts(cast(str, getattr(obj, id_field)), splits=splits, salt=salt)
