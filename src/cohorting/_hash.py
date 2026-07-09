"""Hash function for producing deterministic floats from string identifiers."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast, overload

from cohorting._core import (
    hash_single as _rust_hash_single,
)
from cohorting._core import (
    hash_strings as _rust_hash_strings,
)
from cohorting._core import (
    random_float as _rust_random_float,
)
from cohorting._core import (
    random_floats as _rust_random_floats,
)
from cohorting._models import _is_numpy_array, _is_pandas_series, _is_polars_series

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    import polars as pl

_INV_2_64: float = 1.0 / (2**64)
"""Reciprocal of 2^64; multiplied instead of dividing to produce float in [0, 1)."""

_USE_XXHASH: bool = False
_USE_DETERMINISTIC: bool = True
_USE_CACHE: bool = False

if os.environ.get("COHORTING_HASH_BACKEND", "").lower() == "xxhash":
    _USE_XXHASH = True


@lru_cache(maxsize=65_536)
def _cached_hash_single(x: str, sep_salt: bytes, use_xxhash: bool) -> float:
    """Cached wrapper around the Rust hash_single."""
    return _rust_hash_single(x, sep_salt, use_xxhash)


def _random_float() -> float:
    """Return a random float in [0, 1) from OS entropy."""
    return _rust_random_float()


def _dispatch_hash(
    data: Any,
    *,
    sep_salt: bytes,
    use_xxhash: bool,
    use_deterministic: bool,
    use_cache: bool,
) -> Any:
    """Apply the selected hash backend to all supported data types.

    Parameters
    ----------
    data : str | int | list[str | int] | np.ndarray | pd.Series | pl.Series
        Input identifiers. Integers are accepted alongside strings.
    sep_salt : bytes
        Pre-encoded b"\\x00" + salt bytes. Unused when ``use_deterministic=False``.
    use_xxhash : bool
        True for xxhash; False for hashlib. Unused when ``use_deterministic=False``.
    use_deterministic : bool
        False to bypass hashing entirely and return OS-entropy random floats.
    use_cache : bool
        True to use LRU-cached hash functions.

    Returns
    -------
    float | list[float] | np.ndarray | pd.Series | pl.Series
        Hashed values in [0, 1). Return type mirrors input type.

    Raises
    ------
    TypeError
        If data is not one of the supported input types.
    """
    if not use_deterministic:
        if isinstance(data, (str, int)):
            return _rust_random_float()
        if isinstance(data, list):
            return _rust_random_floats(len(data))
        if _is_numpy_array(data):
            import numpy as np

            raw_arr = np.frombuffer(os.urandom(data.size * 8), dtype=np.uint64)
            return (raw_arr.astype(np.float64) * _INV_2_64).reshape(data.shape)
        if _is_pandas_series(data):
            import numpy as np
            import pandas as pd

            raw_arr = np.frombuffer(os.urandom(len(data) * 8), dtype=np.uint64)
            return pd.Series(
                data=raw_arr.astype(np.float64) * _INV_2_64,
                index=data.index,
                name=data.name,
            )
        if _is_polars_series(data):
            import polars as pl

            n = len(data)
            raw_bytes = os.urandom(n * 8)
            return pl.Series(
                name=data.name,
                values=[
                    int.from_bytes(raw_bytes[i * 8 : (i + 1) * 8], "little") * _INV_2_64
                    for i in range(n)
                ],
            )
        raise TypeError(
            f"hash_values expected str, list, np.ndarray, pd.Series, or pl.Series; "
            f"got {type(data).__name__}"
        )

    if isinstance(data, (str, int)):
        norm: str | int = int(data) if isinstance(data, bool) else data
        norm_str = str(norm)
        if use_cache:
            return _cached_hash_single(norm_str, sep_salt, use_xxhash)
        return _rust_hash_single(norm_str, sep_salt, use_xxhash)

    if isinstance(data, list):
        norm_list = [str(int(x) if isinstance(x, bool) else x) for x in data]
        if use_cache:
            return [_cached_hash_single(x, sep_salt, use_xxhash) for x in norm_list]
        return _rust_hash_strings(norm_list, sep_salt, use_xxhash)

    if _is_numpy_array(data):
        import numpy as np

        flat = data.flatten()
        str_ids = [str(int(x) if isinstance(x, bool) else x) for x in flat]
        result_flat = _rust_hash_strings(str_ids, sep_salt, use_xxhash)
        return np.array(result_flat, dtype=np.float64).reshape(data.shape)

    if _is_pandas_series(data):
        import pandas as pd

        str_ids = [str(int(x) if isinstance(x, bool) else x) for x in data]
        result = _rust_hash_strings(str_ids, sep_salt, use_xxhash)
        return pd.Series(result, index=data.index, name=data.name)

    if _is_polars_series(data):
        import polars as pl

        str_ids = [str(int(x) if isinstance(x, bool) else x) for x in data]
        result = _rust_hash_strings(str_ids, sep_salt, use_xxhash)
        return pl.Series(name=data.name, values=result)

    raise TypeError(
        f"hash_values expected str, list, np.ndarray, pd.Series, or pl.Series; "
        f"got {type(data).__name__}"
    )


@overload
def hash_values(data: str, *, salt: str) -> float: ...


@overload
def hash_values(data: int, *, salt: str) -> float: ...


@overload
def hash_values(data: list[str], *, salt: str) -> list[float]: ...


@overload
def hash_values(data: list[int], *, salt: str) -> list[float]: ...


@overload
def hash_values(data: npt.NDArray[Any], *, salt: str) -> npt.NDArray[np.float64]: ...


@overload
def hash_values(data: pd.Series, *, salt: str) -> pd.Series: ...


@overload
def hash_values(data: pl.Series, *, salt: str) -> pl.Series: ...


def hash_values(data: Any, *, salt: str) -> Any:
    """Hash identifiers to deterministic floats in [0, 1).

    The default backend is blake2b (always available, compiled in Rust).
    To use the faster xxhash backend, set ``COHORTING_HASH_BACKEND=xxhash`` or call
    ``cohorting.config.xxhash = True`` before hashing. For high-throughput use
    cases, prefer :class:`cohorting.Experiment` — it computes the salt bytes once
    at construction rather than on every call.

    Integer identifiers are converted to their decimal string representation before
    hashing, so ``hash_values(123, salt="exp")`` and
    ``hash_values("123", salt="exp")`` produce the same value.

    Parameters
    ----------
    data : str | int | list[str] | list[int] | np.ndarray | pd.Series | pl.Series
        Identifier(s) to hash. Strings and integers are both accepted.
    salt : str
        Appended to each identifier before hashing. Use a unique
        experiment name to ensure assignment isolation between experiments.

    Returns
    -------
    float | list[float] | np.ndarray | pd.Series | pl.Series
        Hashed values in [0, 1). Return type mirrors the input type.

    Raises
    ------
    TypeError
        If data is not one of the supported input types.

    Notes
    -----
    :class:`cohorting.Experiment` is faster for repeated calls against the same
    salt because it pre-encodes the salt bytes at construction instead of
    re-encoding them on every call. Setting ``cohorting.config.cache = True``
    enables LRU caching so repeated identical identifiers are looked up rather
    than re-hashed.

    Examples
    --------
    >>> from cohorting import hash_values
    >>> hash_values(data="user_1", salt="exp")  # doctest: +ELLIPSIS
    0...
    >>> hash_values(data=123, salt="exp") == hash_values(data="123", salt="exp")
    True
    >>> len(hash_values(data=["u1", "u2", "u3"], salt="exp"))
    3
    """
    return _dispatch_hash(
        data,
        sep_salt=b"\x00"
        + salt.encode(),  # \x00 separates identifier from salt; see Experiment.__init__
        use_xxhash=_USE_XXHASH,
        use_deterministic=_USE_DETERMINISTIC,
        use_cache=_USE_CACHE,
    )


def hash_orm(
    obj: object | list[object], *, id_field: str, salt: str
) -> float | list[float]:
    """Hash the id_field attribute of a dataclass, Pydantic model, or plain object.

    Functional counterpart to :meth:`cohorting.Experiment.hash_orm`. Normalizes and
    re-encodes the salt on every call; prefer :class:`cohorting.Experiment` for
    repeated calls against the same experiment configuration.

    Parameters
    ----------
    obj : object | list[object]
        A single model instance or a list of model instances.
    id_field : str
        Attribute name to read the identifier from. The attribute may be a
        ``str`` or ``int``.
    salt : str
        Appended to the identifier before hashing. Use a unique experiment name.

    Returns
    -------
    float | list[float]
        Hash value(s) >= 0, < 1.

    Raises
    ------
    AttributeError
        If obj does not have the given attribute.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> from cohorting import hash_values
    >>> from cohorting._hash import hash_orm
    >>> @dataclass
    ... class User:
    ...     user_id: str
    >>> hash_values(data="user_1", salt="exp") == hash_orm(
    ...     obj=User(user_id="user_1"), id_field="user_id", salt="exp"
    ... )
    True
    """
    if isinstance(obj, list):
        return hash_values([cast(str, getattr(o, id_field)) for o in obj], salt=salt)
    return hash_values(cast(str, getattr(obj, id_field)), salt=salt)
