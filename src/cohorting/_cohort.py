"""Cohort assignment functions."""

from __future__ import annotations

import bisect
from functools import cache, lru_cache
from typing import TYPE_CHECKING, Any, cast, overload

import cohorting._hash as _hash_mod
from cohorting._hash import (
    _compute_hashlib,
    _compute_xxhash,
    _hash_hashlib,
    _hash_xxhash,
    _random_float,
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


def _lookup_cohort(hash_val: float, sorted_bounds: _SortedBounds) -> str:
    """Map a hash float to a cohort name via bisect.

    Parameters
    ----------
    hash_val : float
        Value in [0, 1) produced by a hash function.
    sorted_bounds : _SortedBounds
        Pre-sorted cohort bounds.

    Returns
    -------
    str
        Cohort name.
    """
    lowers = _get_lower_bounds(sorted_bounds)
    # bisect_right returns the insertion point after any equal values, so
    # subtracting 1 gives the rightmost bucket whose lower bound is ≤ hash_val.
    return sorted_bounds[bisect.bisect_right(lowers, hash_val) - 1][0]


@lru_cache(maxsize=65_536)
def _assign_hashlib(x: str | int, sep_salt: bytes, sorted_bounds: _SortedBounds) -> str:
    """Hash + assign using the hashlib backend, cached.

    Caches the full ``(identifier, salt, bounds) → cohort`` result so repeat
    callers skip both the hash computation and the bisect lookup in one step.

    Parameters
    ----------
    x : str | int
        Identifier.
    sep_salt : bytes
        Pre-encoded b"\\x00" + salt bytes.
    sorted_bounds : _SortedBounds
        Pre-sorted cohort bounds.

    Returns
    -------
    str
        Cohort name.
    """
    return _lookup_cohort(_hash_hashlib(x, sep_salt), sorted_bounds)


@lru_cache(maxsize=65_536)
def _assign_xxhash(x: str | int, sep_salt: bytes, sorted_bounds: _SortedBounds) -> str:
    """Hash + assign using the xxhash backend, cached.

    Parameters
    ----------
    x : str | int
        Identifier.
    sep_salt : bytes
        Pre-encoded b"\\x00" + salt bytes.
    sorted_bounds : _SortedBounds
        Pre-sorted cohort bounds.

    Returns
    -------
    str
        Cohort name.
    """
    return _lookup_cohort(_hash_xxhash(x, sep_salt), sorted_bounds)


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
        using OS entropy, immune to any Python- or NumPy-level random seed.
    use_cache : bool
        True to use LRU-cached assign functions. False bypasses both the assign cache
        and the underlying hash cache.

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
        if isinstance(data, (str, int)):
            return _lookup_cohort(_random_float(), sorted_bounds)
        if isinstance(data, list):
            return [_lookup_cohort(_random_float(), sorted_bounds) for _ in data]
        if _is_numpy_array(data):
            import numpy as np

            return np.vectorize(
                lambda _: _lookup_cohort(_random_float(), sorted_bounds),
                otypes=[str],
            )(data)
        if _is_pandas_series(data):
            import numpy as np
            import pandas as pd

            return pd.Series(
                data=np.vectorize(
                    lambda _: _lookup_cohort(_random_float(), sorted_bounds),
                    otypes=[str],
                )(data.to_numpy()),
                index=data.index,
                name=data.name,
            )
        if _is_polars_series(data):
            import polars as pl

            return data.map_elements(
                lambda _: _lookup_cohort(_random_float(), sorted_bounds),
                return_dtype=pl.String,
            )
        raise TypeError(
            f"assign_cohorts expected str, list, np.ndarray, pd.Series, or pl.Series; "
            f"got {type(data).__name__}"
        )

    if not use_cache:
        compute_fn = _compute_xxhash if use_xxhash else _compute_hashlib
        lowers = _get_lower_bounds(sorted_bounds)
        _bisect = bisect.bisect_right
        if isinstance(data, (str, int)):
            norm: str | int = int(data) if isinstance(data, bool) else data
            return sorted_bounds[_bisect(lowers, compute_fn(norm, sep_salt)) - 1][0]
        if isinstance(data, list):
            return [
                sorted_bounds[
                    _bisect(
                        lowers,
                        compute_fn(int(x) if isinstance(x, bool) else x, sep_salt),
                    )
                    - 1
                ][0]
                for x in data
            ]
        if _is_numpy_array(data):
            import numpy as np

            return np.vectorize(
                lambda x: sorted_bounds[_bisect(lowers, compute_fn(x, sep_salt)) - 1][
                    0
                ],
                otypes=[str],
            )(data)
        if _is_pandas_series(data):
            import numpy as np
            import pandas as pd

            return pd.Series(
                data=np.vectorize(
                    lambda x: sorted_bounds[
                        _bisect(lowers, compute_fn(x, sep_salt)) - 1
                    ][0],
                    otypes=[str],
                )(data.to_numpy()),
                index=data.index,
                name=data.name,
            )
        if _is_polars_series(data):
            import polars as pl

            return data.map_elements(
                lambda x: sorted_bounds[_bisect(lowers, compute_fn(x, sep_salt)) - 1][
                    0
                ],
                return_dtype=pl.String,
            )
        raise TypeError(
            f"assign_cohorts expected str, list, np.ndarray, pd.Series, or pl.Series; "
            f"got {type(data).__name__}"
        )

    assign_fn = _assign_xxhash if use_xxhash else _assign_hashlib

    if isinstance(data, (str, int)):
        # bool is a subclass of int; True == 1 == hash(True), so without
        # normalization the LRU cache cannot distinguish True from 1.
        norm2: str | int = int(data) if isinstance(data, bool) else data
        return assign_fn(norm2, sep_salt, sorted_bounds)

    if isinstance(data, list):
        return [
            assign_fn(int(x) if isinstance(x, bool) else x, sep_salt, sorted_bounds)
            for x in data
        ]

    if _is_numpy_array(data):
        import numpy as np

        return np.vectorize(
            lambda x: assign_fn(x, sep_salt, sorted_bounds),
            otypes=[str],
        )(data)

    if _is_pandas_series(data):
        import numpy as np
        import pandas as pd

        return pd.Series(
            np.vectorize(
                lambda x: assign_fn(x, sep_salt, sorted_bounds),
                otypes=[str],
            )(data.to_numpy()),
            index=data.index,
            name=data.name,
        )

    if _is_polars_series(data):
        import polars as pl

        return data.map_elements(
            lambda x: assign_fn(x, sep_salt, sorted_bounds),
            return_dtype=pl.String,
        )

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
