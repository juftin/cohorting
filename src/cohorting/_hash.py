"""Hash function for producing deterministic floats from string identifiers."""

from __future__ import annotations

import hashlib as _hashlib
import os
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast, overload

from cohorting._models import _is_numpy_array, _is_pandas_series, _is_polars_series

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    import polars as pl

_INV_2_64: float = 1.0 / (2**64)
"""Reciprocal of 2^64; multiplied instead of dividing to produce float in [0, 1)."""

# Per-backend vectorized wrappers. Lazily initialized, one per (backend, cache) pair.
_VEC_HASH_HASHLIB: Any = None
_VEC_HASH_XXHASH: Any = None
_VEC_HASH_HASHLIB_NOCACHE: Any = None
_VEC_HASH_XXHASH_NOCACHE: Any = None

# Opt-in xxhash backend. Set COHORTING_HASH_BACKEND=xxhash to enable.
# Deliberately not auto-detected: just having xxhash installed should not silently
# change hash outputs in existing deployments.
_xxhash_mod: Any = None
_USE_XXHASH: bool = False
_USE_DETERMINISTIC: bool = True
_USE_CACHE: bool = False

if os.environ.get("COHORTING_HASH_BACKEND", "").lower() == "xxhash":
    try:
        import xxhash as _xxhash_mod

        _USE_XXHASH = True
    except ImportError:
        warnings.warn(
            "COHORTING_HASH_BACKEND=xxhash requested but xxhash is not installed; "
            "falling back to hashlib. Install cohorting[xxhash] to enable xxhash.",
            ImportWarning,
            stacklevel=2,
        )


def _ensure_xxhash() -> None:
    """Load the xxhash module on demand, raising ImportError if not installed.

    Raises
    ------
    ImportError
        If xxhash is not installed.
    """
    global _xxhash_mod
    if _xxhash_mod is None:
        try:
            import xxhash as _xxhash_import

            _xxhash_mod = _xxhash_import
        except ImportError as exc:
            raise ImportError(
                "xxhash is not installed. Install: pip install 'cohorting[xxhash]'"
            ) from exc


def _compute_hashlib(x: str | int, sep_salt: bytes) -> float:
    """Hash using hashlib blake2b — the raw computation, never cached.

    Parameters
    ----------
    x : str | int
        Identifier to hash. Integers are converted to their decimal string
        representation before hashing, so ``123`` and ``"123"`` produce the
        same value.
    sep_salt : bytes
        Pre-encoded b"\\x00" + salt bytes.

    Returns
    -------
    float
        Deterministic float in [0, 1).

    Notes
    -----
    ``digest_size=8`` produces exactly 8 bytes (64 bits). ``int.from_bytes``
    with ``"little"`` interprets those bytes as a little-endian unsigned 64-bit
    integer, giving a value in ``[0, 2**64)``. Multiplying by ``_INV_2_64``
    maps that range to ``[0, 1)`` with near-uniform distribution.
    """
    h = _hashlib.blake2b(digest_size=8)
    h.update(x.encode() if isinstance(x, str) else str(x).encode())
    h.update(sep_salt)
    return int.from_bytes(h.digest(), "little") * _INV_2_64


@lru_cache(maxsize=65_536)
def _hash_hashlib(x: str | int, sep_salt: bytes) -> float:
    """Cached wrapper around _compute_hashlib.

    Parameters
    ----------
    x : str | int
        Identifier to hash.
    sep_salt : bytes
        Pre-encoded b"\\x00" + salt bytes.

    Returns
    -------
    float
        Deterministic float in [0, 1).
    """
    return _compute_hashlib(x, sep_salt)


def _compute_xxhash(x: str | int, sep_salt: bytes) -> float:
    """Hash using xxhash xxh3_64 — the raw computation, never cached.

    Parameters
    ----------
    x : str | int
        Identifier to hash. Integers are converted to their decimal string
        representation before hashing, so ``123`` and ``"123"`` produce the
        same value.
    sep_salt : bytes
        Pre-encoded b"\\x00" + salt bytes.

    Returns
    -------
    float
        Deterministic float in [0, 1).
    """
    h = _xxhash_mod.xxh3_64()
    h.update(x.encode() if isinstance(x, str) else str(x).encode())
    h.update(sep_salt)
    return h.intdigest() * _INV_2_64


@lru_cache(maxsize=65_536)
def _hash_xxhash(x: str | int, sep_salt: bytes) -> float:
    """Cached wrapper around _compute_xxhash.

    Parameters
    ----------
    x : str | int
        Identifier to hash.
    sep_salt : bytes
        Pre-encoded b"\\x00" + salt bytes.

    Returns
    -------
    float
        Deterministic float in [0, 1).
    """
    return _compute_xxhash(x, sep_salt)


def _hash_single(x: str | int, sep_salt: bytes) -> float:
    """Route to the active global backend.

    Parameters
    ----------
    x : str | int
        Identifier to hash.
    sep_salt : bytes
        Pre-encoded b"\\x00" + salt bytes.

    Returns
    -------
    float
        Deterministic float in [0, 1).
    """
    return _hash_xxhash(x, sep_salt) if _USE_XXHASH else _hash_hashlib(x, sep_salt)


def _random_float() -> float:
    """Return a random float in [0, 1) sourced from OS entropy.

    Uses ``os.urandom`` directly so the result is immune to any Python-level or
    NumPy-level random seed (``random.seed``, ``numpy.random.seed``, etc.).
    """
    return int.from_bytes(os.urandom(8), "little") * _INV_2_64


def _get_vec_hash(use_xxhash: bool, use_cache: bool) -> Any:
    """Return a lazily-initialized np.vectorize wrapper for backend and cache mode."""
    global \
        _VEC_HASH_HASHLIB, \
        _VEC_HASH_XXHASH, \
        _VEC_HASH_HASHLIB_NOCACHE, \
        _VEC_HASH_XXHASH_NOCACHE
    import numpy as np

    if use_xxhash:
        if use_cache:
            if _VEC_HASH_XXHASH is None:
                _VEC_HASH_XXHASH = np.vectorize(_hash_xxhash, otypes=[float])
            return _VEC_HASH_XXHASH
        if _VEC_HASH_XXHASH_NOCACHE is None:
            _VEC_HASH_XXHASH_NOCACHE = np.vectorize(_compute_xxhash, otypes=[float])
        return _VEC_HASH_XXHASH_NOCACHE
    if use_cache:
        if _VEC_HASH_HASHLIB is None:
            _VEC_HASH_HASHLIB = np.vectorize(_hash_hashlib, otypes=[float])
        return _VEC_HASH_HASHLIB
    if _VEC_HASH_HASHLIB_NOCACHE is None:
        _VEC_HASH_HASHLIB_NOCACHE = np.vectorize(_compute_hashlib, otypes=[float])
    return _VEC_HASH_HASHLIB_NOCACHE


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
        False to bypass hashing entirely and return OS-entropy random floats,
        immune to any Python- or NumPy-level random seed.
    use_cache : bool
        True to use LRU-cached hash functions. Useful when the same identifiers
        appear repeatedly. False (default) avoids cache overhead for single-pass
        workloads over unique identifiers.

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
            return _random_float()
        if isinstance(data, list):
            raw_bytes = os.urandom(len(data) * 8)
            return [
                int.from_bytes(raw_bytes[i * 8 : (i + 1) * 8], "little") * _INV_2_64
                for i in range(len(data))
            ]
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

    hash_fn = (
        (_hash_xxhash if use_xxhash else _hash_hashlib)
        if use_cache
        else (_compute_xxhash if use_xxhash else _compute_hashlib)
    )

    if isinstance(data, (str, int)):
        # bool is a subclass of int; True == 1 == hash(True), so without
        # normalization the LRU cache cannot distinguish True from 1.
        norm: str | int = int(data) if isinstance(data, bool) else data
        return hash_fn(norm, sep_salt)

    if isinstance(data, list):
        return [hash_fn(int(x) if isinstance(x, bool) else x, sep_salt) for x in data]

    if _is_numpy_array(data):
        return _get_vec_hash(use_xxhash, use_cache)(data, sep_salt)

    if _is_pandas_series(data):
        import pandas as pd

        return pd.Series(
            _get_vec_hash(use_xxhash, use_cache)(data.to_numpy(), sep_salt),
            index=data.index,
            name=data.name,
        )

    if _is_polars_series(data):
        import polars as pl

        return data.map_elements(
            lambda x: hash_fn(x, sep_salt), return_dtype=pl.Float64
        )

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

    The default backend is hashlib blake2b (always available). To use the faster
    xxhash backend, set ``COHORTING_HASH_BACKEND=xxhash`` or call
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
