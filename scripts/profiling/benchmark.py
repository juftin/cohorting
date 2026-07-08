"""Benchmark cohorting assignment throughput across data types, backends, and sizes.

Run with:
    uv run python scripts/profiling/benchmark.py

Reports wall-clock time (ms) and throughput (M items/sec) for each combination
of input type, backend, and cache setting. Optional dependencies (numpy, pandas,
polars, xxhash) are skipped gracefully if not installed.

Results shape:
  - list[str]     — pure-Python path; measures hashing + Python list overhead
  - np.ndarray    — np.vectorize path; each element hashed via Python callback
  - pd.Series     — same as ndarray, with index/name wrapping
  - pl.Series     — polars map_elements path

Cache notes:
  - cache=False  results are measured cold (no LRU entries for these identifiers)
  - cache=True   results are measured warm (cache pre-populated before timing)
"""

from __future__ import annotations

import timeit
from typing import Any

from cohorting import Experiment, even_split

# --- Configuration -----------------------------------------------------------

SIZES: list[int] = [1_000, 10_000, 100_000, 1_000_000]
REPEATS: int = 5  # timeit repetitions; minimum is reported

# --- Helpers -----------------------------------------------------------------


def _make_ids(n: int) -> list[str]:
    """Return a list of n string identifiers."""
    return [f"user_{i}" for i in range(n)]


def _make_exp(*, cache: bool, xxhash: bool) -> Experiment:
    """Construct a two-cohort experiment with the given backend settings."""
    return Experiment(
        name="bench",
        splits=even_split(names=["control", "treatment"]),
        cache=cache,
        xxhash=xxhash,
    )


def _time(fn: Any, n: int, *, warm_cache: bool = False) -> tuple[float, float]:
    """Return (best_seconds, M_items_per_sec) for a single benchmark call.

    Parameters
    ----------
    fn : callable
        Zero-argument callable that processes ``n`` identifiers in one call.
    n : int
        Number of identifiers processed per call (used to compute throughput).
    warm_cache : bool
        If True, call ``fn`` once before timing to populate any LRU caches.

    Returns
    -------
    tuple[float, float]
        ``(best_time_seconds, throughput_M_per_sec)``
    """
    if warm_cache:
        fn()
    best = min(timeit.repeat(fn, number=1, repeat=REPEATS))
    return best, n / best / 1_000_000


def _row(label: str, seconds: float, throughput: float) -> None:
    """Print a formatted benchmark result row."""
    print(f"    {label:<32} {seconds * 1000:8.1f} ms   {throughput:7.2f} M/s")


# --- Benchmark scenarios ------------------------------------------------------


def bench_list(exp: Experiment, ids: list[str], *, warm: bool) -> None:
    """Benchmark list[str] input."""
    secs, tput = _time(lambda: exp.assign(ids), len(ids), warm_cache=warm)
    _row("list[str]", secs, tput)


def bench_numpy(exp: Experiment, ids: list[str], *, warm: bool) -> None:
    """Benchmark np.ndarray input; skip if numpy is not installed."""
    try:
        import numpy as np
    except ImportError:
        print("    np.ndarray                       (numpy not installed)")
        return
    arr = np.array(ids)
    secs, tput = _time(lambda: exp.assign(arr), len(ids), warm_cache=warm)
    _row("np.ndarray", secs, tput)


def bench_pandas(exp: Experiment, ids: list[str], *, warm: bool) -> None:
    """Benchmark pd.Series input; skip if pandas is not installed."""
    try:
        import pandas as pd
    except ImportError:
        print("    pd.Series                        (pandas not installed)")
        return
    series = pd.Series(ids)
    secs, tput = _time(lambda: exp.assign(series), len(ids), warm_cache=warm)
    _row("pd.Series", secs, tput)


def bench_polars(exp: Experiment, ids: list[str], *, warm: bool) -> None:
    """Benchmark pl.Series input; skip if polars is not installed."""
    try:
        import polars as pl
    except ImportError:
        print("    pl.Series                        (polars not installed)")
        return
    series = pl.Series(ids)
    secs, tput = _time(lambda: exp.assign(series), len(ids), warm_cache=warm)
    _row("pl.Series", secs, tput)


def _bench_scenario(label: str, *, cache: bool, xxhash: bool, ids: list[str]) -> None:
    """Run all four input types for one (cache, backend) scenario.

    Parameters
    ----------
    label : str
        Human-readable scenario description printed as a sub-header.
    cache : bool
        Whether to enable LRU caching on the experiment.
    xxhash : bool
        Whether to use the xxhash backend.
    ids : list[str]
        Identifiers to assign.
    """
    print(f"\n  [{label}]")
    print(f"    {'input type':<32} {'time':>12}   {'throughput':>10}")
    print(f"    {'-' * 32} {'-' * 12}   {'-' * 10}")
    try:
        exp = _make_exp(cache=cache, xxhash=xxhash)
    except ImportError:
        print("    (xxhash not installed — skipping)")
        return
    warm = cache
    bench_list(exp, ids, warm=warm)
    bench_numpy(exp, ids, warm=warm)
    bench_pandas(exp, ids, warm=warm)
    bench_polars(exp, ids, warm=warm)


# --- Entry point -------------------------------------------------------------


def run() -> None:
    """Run the full benchmark suite and print results to stdout."""
    scenarios = [
        ("hashlib  cache=False", False, False),
        ("hashlib  cache=True (warm)", True, False),
        ("xxhash   cache=False", False, True),
        ("xxhash   cache=True (warm)", True, True),
    ]

    for n in SIZES:
        print(f"\n{'=' * 62}")
        print(f"  N = {n:>10,} identifiers")
        print(f"{'=' * 62}")
        ids = _make_ids(n)
        for label, cache, xxhash in scenarios:
            _bench_scenario(label, cache=cache, xxhash=xxhash, ids=ids)


if __name__ == "__main__":
    run()
