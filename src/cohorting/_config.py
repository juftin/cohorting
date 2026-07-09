"""Runtime configuration for the cohorting library."""

from __future__ import annotations


class _Config:
    """Runtime configuration object. Access via ``cohorting.config``.

    Controls the hash backend, determinism mode, and caching used for all
    cohort assignments. The default backend is `hashlib` `blake2b` (stdlib,
    always available). Switch to `xxhash` for higher throughput when
    ``cohorting[xxhash]`` is installed.

    .. warning::
        ``cohorting.config`` is a process-wide singleton — mutations are **not
        thread-safe**. Changing any setting from one thread affects all threads
        immediately, and ``cache_clear()`` called during a concurrent request can
        return a stale un-cached result to another thread mid-flight.

        For multi-threaded code, set config once at process startup before any
        requests are served, or use :class:`cohorting.Experiment` instead —
        ``Experiment`` captures its settings at construction and never reads the
        global config again.

        Also note: switching backends or determinism modes changes all hash outputs.
        Keep settings consistent across all processes in a deployment.

    Examples
    --------
    >>> import cohorting
    >>> cohorting.config.xxhash = True        # opt in to xxhash backend
    >>> cohorting.config.xxhash = False       # revert to hashlib
    >>> cohorting.config.deterministic = False  # random assignment via os.urandom
    >>> cohorting.config.deterministic = True   # revert to deterministic hashing
    >>> cohorting.config.cache = True         # enable LRU caching for repeated IDs
    >>> cohorting.config.cache = False        # disable caching (default)
    """

    @property
    def xxhash(self) -> bool:
        """Whether the xxhash backend is currently active."""
        import cohorting._hash as _hash_mod

        return _hash_mod._USE_XXHASH

    @xxhash.setter
    def xxhash(self, enable: bool) -> None:
        """Enable or disable the xxhash backend.

        Parameters
        ----------
        enable : bool
            True to switch to xxhash; False to revert to blake2b. Clears the
            hash and assign caches on every call to prevent mixed-backend results.

        Notes
        -----
        Both blake2b and xxhash backends are compiled into the Rust extension.
        No extra Python package is required for xxhash.
        """
        import cohorting._cohort as _cohort_mod
        import cohorting._hash as _hash_mod

        _hash_mod._USE_XXHASH = enable
        _hash_mod._cached_hash_single.cache_clear()
        _cohort_mod._cached_assign_single.cache_clear()

    @property
    def deterministic(self) -> bool:
        """Whether hashing is deterministic (default True).

        When False, ``hash_values`` and ``assign_cohorts`` ignore the identifier
        and salt entirely, returning values drawn from ``os.urandom``. The result
        is immune to any Python- or NumPy-level random seed.
        """
        import cohorting._hash as _hash_mod

        return _hash_mod._USE_DETERMINISTIC

    @deterministic.setter
    def deterministic(self, enable: bool) -> None:
        """Enable or disable deterministic hashing.

        Parameters
        ----------
        enable : bool
            False to bypass hashing and return random floats from ``os.urandom``.
            Clears the hash and assign caches when called so that stale
            deterministic results cannot be returned after a mode switch.
        """
        import cohorting._cohort as _cohort_mod
        import cohorting._hash as _hash_mod

        _hash_mod._USE_DETERMINISTIC = enable
        _hash_mod._cached_hash_single.cache_clear()
        _cohort_mod._cached_assign_single.cache_clear()

    @property
    def cache(self) -> bool:
        """Whether LRU caching of hash/assign results is enabled (default False).

        When True, repeated calls with the same identifier and salt return a cached
        result in O(1) without recomputing the hash. Useful for workloads where the
        same user IDs appear many times (e.g. a web server handling repeat visitors).

        When False (the default), each call computes the hash directly. This avoids
        the dict-lookup overhead and cache-eviction churn that makes caching a net
        loss for single-pass batch jobs over large sets of unique identifiers.
        """
        import cohorting._hash as _hash_mod

        return _hash_mod._USE_CACHE

    @cache.setter
    def cache(self, enable: bool) -> None:
        """Enable or disable LRU caching of hash and assign results.

        Parameters
        ----------
        enable : bool
            True to cache results keyed on (identifier, salt); False to compute
            every call fresh. Clears existing caches when called.
        """
        import cohorting._cohort as _cohort_mod
        import cohorting._hash as _hash_mod

        _hash_mod._USE_CACHE = enable
        _hash_mod._cached_hash_single.cache_clear()
        _cohort_mod._cached_assign_single.cache_clear()


config: _Config = _Config()
"""Runtime configuration.

Use ``config.xxhash``, ``config.deterministic``, and ``config.cache``
to control hashing.
"""
