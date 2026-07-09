"""Class-based API for experiment cohorting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, overload

from cohorting._cohort import (
    _dispatch_assign,
    _splits_to_sorted_bounds,
)
from cohorting._hash import _dispatch_hash
from cohorting._models import (
    SplitInput,
    _is_pandas_frame,
    _is_polars_frame,
    _normalize_splits,
)
from cohorting._validate import validate_splits

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    import polars as pl


class Experiment:
    """Class-based API for reproducible experiment cohorting.

    Bundles name, salt, and splits into a single object. Splits are validated
    and sorted once at construction — subsequent :meth:`assign` and :meth:`hash`
    calls skip re-validation and re-sorting for better performance compared to the
    functional API.

    Attributes
    ----------
    name : str
        Experiment name.
    salt : str
        Salt appended to each identifier before hashing.
    splits : SplitMap
        Normalized cohort split map spanning exactly [0, 1).

    Examples
    --------
    >>> from cohorting import Experiment, even_split
    >>> exp = Experiment(
    ...     name="checkout",
    ...     splits=even_split(names=["control", "treatment"]),
    ... )
    >>> exp.assign(data="user_1")
    'control'
    >>> exp.cohorts
    ['control', 'treatment']
    """

    __slots__: tuple[str, ...] = (
        "_cache",
        "_deterministic",
        "_sep_salt",
        "_sorted_bounds",
        "_xxhash",
        "name",
        "salt",
        "splits",
    )

    def __init__(
        self,
        *,
        name: str,
        salt: str | None = None,
        splits: SplitInput,
        xxhash: bool = False,
        deterministic: bool = True,
        cache: bool = False,
    ) -> None:
        """Construct an Experiment and validate splits.

        Parameters
        ----------
        name : str
            Experiment name.
        salt : str, optional
            Salt appended to identifiers before hashing. Defaults to ``name``
            when omitted — the common case where the experiment name is
            sufficient to isolate assignments across experiments.
        splits : SplitInput
            Cohort splits as a SplitMap dict or list of CohortSplit instances.
            Must span exactly [0, 1) with no gaps or overlaps.
        xxhash : bool, optional
            Use the xxhash backend for this experiment. Requires
            ``cohorting[xxhash]`` to be installed. Defaults to False (hashlib).
        deterministic : bool, optional
            When False, bypass hashing entirely and assign each identifier to a
            randomly chosen cohort using OS entropy (``os.urandom``). The result
            is immune to any Python- or NumPy-level random seed. Defaults to True.
        cache : bool, optional
            When True, cache hash and assign results keyed on the identifier and
            salt. Useful when the same identifiers recur (e.g. a web server
            processing repeat visitors). Defaults to False.

        Raises
        ------
        ValueError
            If splits are invalid (gap, overlap, or wrong bounds).
        ImportError
            If ``xxhash=True`` and xxhash is not installed.
        """
        self.name = name
        self.salt = salt if salt is not None else name
        self.splits = _normalize_splits(splits)
        validate_splits(self.splits)
        self._xxhash = xxhash
        self._deterministic = deterministic
        self._cache = cache
        # \x00 byte separates identifier from salt, preventing a collision
        # between identifier "foob" + salt "ar" and identifier "foo" + salt "bar".
        self._sep_salt = b"\x00" + self.salt.encode()
        self._sorted_bounds = _splits_to_sorted_bounds(self.splits)

    @property
    def cohorts(self) -> list[str]:
        """Return cohort names ordered by lower bound.

        Returns
        -------
        list[str]
            Cohort names sorted by ascending lower bound.
        """
        return [name for name, _lower, _upper in self._sorted_bounds]

    def __repr__(self) -> str:
        """Return a readable representation of the Experiment."""
        parts = [
            f"name={self.name!r}",
            f"salt={self.salt!r}",
            f"splits={self.splits!r}",
        ]
        if self._xxhash:
            parts.append("xxhash=True")
        if not self._deterministic:
            parts.append("deterministic=False")
        if self._cache:
            parts.append("cache=True")
        return f"Experiment({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        """Return True if both experiments have equal configuration.

        Compares name, salt, splits, backend, determinism, and cache.
        """
        if not isinstance(other, Experiment):
            return NotImplemented
        return (
            self.name,
            self.salt,
            self.splits,
            self._xxhash,
            self._deterministic,
            self._cache,
        ) == (
            other.name,
            other.salt,
            other.splits,
            other._xxhash,
            other._deterministic,
            other._cache,
        )

    @overload
    def hash(self, data: str) -> float: ...

    @overload
    def hash(self, data: int) -> float: ...

    @overload
    def hash(self, data: list[str]) -> list[float]: ...

    @overload
    def hash(self, data: list[int]) -> list[float]: ...

    @overload
    def hash(self, data: npt.NDArray[Any]) -> npt.NDArray[np.float64]: ...

    @overload
    def hash(self, data: pd.Series) -> pd.Series: ...

    @overload
    def hash(self, data: pl.Series) -> pl.Series: ...

    def hash(self, data: Any) -> Any:
        """Hash identifiers to floats in [0, 1).

        Integer identifiers are converted to their decimal string representation
        before hashing, so ``exp.hash(123)`` and ``exp.hash("123")`` produce the
        same value.

        Parameters
        ----------
        data : str | int | list[str] | list[int] | np.ndarray | pd.Series | pl.Series
            Identifier(s) to hash. Strings and integers are both accepted.

        Returns
        -------
        float | list[float] | np.ndarray | pd.Series | pl.Series
            Hash values in [0, 1). Return type mirrors input type.

        Examples
        --------
        >>> from cohorting import Experiment, even_split
        >>> splits = even_split(names=["control", "treatment"])
        >>> exp = Experiment(name="checkout", splits=splits)
        >>> exp.hash(data="user_1")  # doctest: +ELLIPSIS
        0...
        >>> 0.0 <= exp.hash(data="user_1") < 1.0
        True
        """
        return _dispatch_hash(
            data,
            sep_salt=self._sep_salt,
            use_xxhash=self._xxhash,
            use_deterministic=self._deterministic,
            use_cache=self._cache,
        )

    @overload
    def assign(self, data: str) -> str: ...

    @overload
    def assign(self, data: int) -> str: ...

    @overload
    def assign(self, data: list[str]) -> list[str]: ...

    @overload
    def assign(self, data: list[int]) -> list[str]: ...

    @overload
    def assign(self, data: npt.NDArray[Any]) -> npt.NDArray[np.str_]: ...

    @overload
    def assign(self, data: pd.Series) -> pd.Series: ...

    @overload
    def assign(self, data: pl.Series) -> pl.Series: ...

    def assign(self, data: Any) -> Any:
        """Assign cohort names to identifiers.

        Skips split validation and sorting — both are done once at construction.
        Integer identifiers are converted to their decimal string representation
        before hashing, so ``exp.assign(123)`` and ``exp.assign("123")`` return
        the same cohort.

        Parameters
        ----------
        data : str | int | list[str] | list[int] | np.ndarray | pd.Series | pl.Series
            Identifier(s) to assign. Strings and integers are both accepted.

        Returns
        -------
        str | list[str] | np.ndarray | pd.Series | pl.Series
            Cohort name(s). Return type mirrors input type.

        Examples
        --------
        >>> from cohorting import Experiment, even_split
        >>> splits = even_split(names=["control", "treatment"])
        >>> exp = Experiment(name="checkout", splits=splits)
        >>> exp.assign(data="user_1")
        'control'
        >>> exp.assign(data=["user_1", "user_2", "user_3"])
        ['control', 'treatment', 'control']
        """
        return _dispatch_assign(
            data,
            sep_salt=self._sep_salt,
            sorted_bounds=self._sorted_bounds,
            use_xxhash=self._xxhash,
            use_deterministic=self._deterministic,
            use_cache=self._cache,
        )

    @overload
    def assign_frame(
        self,
        df: pd.DataFrame,
        *,
        id_column: str,
        output_column: str = ...,
    ) -> pd.DataFrame: ...

    @overload
    def assign_frame(
        self,
        df: pl.DataFrame,
        *,
        id_column: str,
        output_column: str = ...,
    ) -> pl.DataFrame: ...

    def assign_frame(
        self,
        df: pd.DataFrame | pl.DataFrame,
        *,
        id_column: str,
        output_column: str = "cohort",
    ) -> pd.DataFrame | pl.DataFrame:
        """Add a cohort assignment column to a DataFrame.

        Skips split validation and sorting — both are done once at construction.

        Parameters
        ----------
        df : pd.DataFrame | pl.DataFrame
            Input DataFrame. Not mutated.
        id_column : str
            Name of the column containing string identifiers to hash.
        output_column : str, optional
            Name of the column to add. Default is "cohort".

        Returns
        -------
        pd.DataFrame | pl.DataFrame
            A copy of the input DataFrame with the cohort column added.

        Raises
        ------
        TypeError
            If df is not a pandas or polars DataFrame.

        Notes
        -----
        Both pandas and polars DataFrames are supported. The input DataFrame is
        not mutated; a new DataFrame with the cohort column appended is returned.

        Examples
        --------
        >>> import pandas as pd
        >>> from cohorting import Experiment, even_split
        >>> splits = even_split(names=["control", "treatment"])
        >>> exp = Experiment(name="checkout", splits=splits)
        >>> df = pd.DataFrame({"user_id": ["user_1", "user_2", "user_3"]})
        >>> exp.assign_frame(df, id_column="user_id")
          user_id     cohort
        0  user_1    control
        1  user_2  treatment
        2  user_3    control
        """
        if _is_pandas_frame(df):
            import pandas as _pd

            pd_cohorts: _pd.Series = self.assign(df[id_column])
            return df.assign(**{output_column: pd_cohorts})

        if _is_polars_frame(df):
            import polars as _pl

            pl_cohorts: _pl.Series = self.assign(df[id_column])
            return df.with_columns(pl_cohorts.alias(output_column))

        raise TypeError(
            f"assign_frame expected pd.DataFrame or pl.DataFrame; "
            f"got {type(df).__name__}"
        )

    def hash_orm(
        self, obj: object | list[object], *, id_field: str
    ) -> float | list[float]:
        """Hash the id_field attribute of a dataclass, pydantic model, or plain object.

        Parameters
        ----------
        obj : object | list[object]
            A single model instance or a list of model instances.
        id_field : str
            Attribute name to read the string identifier from.

        Returns
        -------
        float | list[float]
            Hash value(s) in [0, 1).

        Raises
        ------
        AttributeError
            If obj does not have the given attribute.

        Examples
        --------
        >>> from dataclasses import dataclass
        >>> from cohorting import Experiment, even_split
        >>> @dataclass
        ... class User:
        ...     user_id: str
        >>> splits = even_split(names=["control", "treatment"])
        >>> exp = Experiment(name="checkout", splits=splits)
        >>> exp.hash_orm(
        ...     User(user_id="user_1"), id_field="user_id"
        ... )  # doctest: +ELLIPSIS
        0...
        >>> 0.0 <= exp.hash_orm(
        ...     User(user_id="user_1"), id_field="user_id"
        ... ) < 1.0
        True
        """
        if isinstance(obj, list):
            return self.hash([cast(str, getattr(o, id_field)) for o in obj])
        return self.hash(cast(str, getattr(obj, id_field)))

    def assign_orm(
        self, obj: object | list[object], *, id_field: str
    ) -> str | list[str]:
        """Assign a cohort to a dataclass, pydantic model, or plain object.

        Reads ``id_field`` from the object and assigns based on its hash value.

        Parameters
        ----------
        obj : object | list[object]
            A single model instance or a list of model instances.
        id_field : str
            Attribute name to read the string identifier from.

        Returns
        -------
        str | list[str]
            Cohort name(s).

        Raises
        ------
        AttributeError
            If obj does not have the given attribute.

        Examples
        --------
        >>> from dataclasses import dataclass
        >>> from cohorting import Experiment, even_split
        >>> @dataclass
        ... class User:
        ...     user_id: str
        >>> splits = even_split(names=["control", "treatment"])
        >>> exp = Experiment(name="checkout", splits=splits)
        >>> exp.assign_orm(User(user_id="user_1"), id_field="user_id")
        'control'
        >>> exp.assign_orm(
        ...     [User(user_id="user_1"), User(user_id="user_2")],
        ...     id_field="user_id",
        ... )
        ['control', 'treatment']
        """
        if isinstance(obj, list):
            return self.assign([cast(str, getattr(o, id_field)) for o in obj])
        return self.assign(cast(str, getattr(obj, id_field)))
