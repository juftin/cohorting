"""Profile cohorting assignment with cProfile to find call-level hotspots.

Run with:
    uv run python scripts/profiling/hotspots.py

Profiles Experiment.assign() on each supported input type with N=100_000
identifiers for both the hashlib and xxhash backends. Prints the top functions
by cumulative time and saves .prof files to scripts/profiling/results/ for
inspection with snakeviz or pstats.

    snakeviz scripts/profiling/results/list_hashlib.prof

Adjust N, TOP_N, or SORT_BY at the top of this file to change what gets profiled.

Note: this file is named ``hotspots.py`` (not ``profile.py``) to avoid
shadowing the stdlib ``profile`` module that ``cProfile`` imports internally.
"""

from __future__ import annotations

import cProfile
import io
import pstats
from collections.abc import Callable
from pathlib import Path

from cohorting import Experiment, even_split

# --- Configuration -----------------------------------------------------------

N: int = 100_000
TOP_N: int = 20
SORT_BY: str = "cumulative"  # cumulative | tottime | ncalls | pcalls
RESULTS_DIR: Path = Path(__file__).parent / "results"

# --- Helpers -----------------------------------------------------------------


def _make_ids(n: int) -> list[str]:
    """Return n string identifiers."""
    return [f"user_{i}" for i in range(n)]


def _make_exp() -> Experiment:
    """Construct a two-cohort experiment with cache disabled."""
    return Experiment(
        name="profile",
        splits=even_split(names=["control", "treatment"]),
        cache=False,
    )


def _print_stats(pr: cProfile.Profile, label: str, top_n: int = TOP_N) -> None:
    """Print the top ``top_n`` functions sorted by ``SORT_BY``.

    Parameters
    ----------
    pr : cProfile.Profile
        Completed profiler instance.
    label : str
        Header label printed above the stats table.
    top_n : int
        Number of functions to show.
    """
    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats(SORT_BY)
    ps.print_stats(top_n)
    print(f"\n{'=' * 72}")
    print(f"  {label}  (top {top_n} by {SORT_BY})")
    print(f"{'=' * 72}")
    # pstats prepends several header lines; skip to the data
    lines = buf.getvalue().splitlines()
    for i, line in enumerate(lines):
        if "ncalls" in line:
            print("\n".join(lines[i:]))
            break


def _save_prof(pr: cProfile.Profile, name: str) -> Path:
    """Dump profiling data to a .prof file and return the path.

    Parameters
    ----------
    pr : cProfile.Profile
        Completed profiler instance.
    name : str
        Stem of the output file (e.g. ``"list_hashlib"``).

    Returns
    -------
    Path
        Path to the saved .prof file.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.prof"
    pr.dump_stats(path)
    return path


def _profile(label: str, file_stem: str, fn: Callable[[], object]) -> None:
    """Run cProfile on ``fn``, print top functions, and save a .prof file.

    Parameters
    ----------
    label : str
        Human-readable description of what is being profiled.
    file_stem : str
        Stem used for the .prof output file.
    fn : Callable[[], object]
        Zero-argument callable to profile.
    """
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    _print_stats(pr, label)
    path = _save_prof(pr, file_stem)
    print(f"\n  → saved: {path}")


# --- Scenarios ---------------------------------------------------------------


def profile_list(exp: Experiment, ids: list[str], backend: str) -> None:
    """Profile list[str] assignment for the given backend.

    Parameters
    ----------
    exp : Experiment
        Pre-constructed experiment for the target backend.
    ids : list[str]
        Identifiers to assign.
    backend : str
        Backend label used in output (e.g. ``"hashlib"`` or ``"xxhash"``).
    """
    _profile(
        label=f"list[str]  n={N:,}  {backend}  cache=False",
        file_stem=f"list_{backend}",
        fn=lambda: exp.assign(ids),
    )


def profile_numpy(exp: Experiment, ids: list[str], backend: str) -> None:
    """Profile np.ndarray assignment; skip if numpy is not installed.

    Parameters
    ----------
    exp : Experiment
        Pre-constructed experiment for the target backend.
    ids : list[str]
        Identifiers to assign.
    backend : str
        Backend label used in output and file stem.
    """
    try:
        import numpy as np
    except ImportError:
        print(f"\nnumpy not installed — skipping np.ndarray ({backend})")
        return
    arr = np.array(ids)
    _profile(
        label=f"np.ndarray  n={N:,}  {backend}  cache=False",
        file_stem=f"numpy_{backend}",
        fn=lambda: exp.assign(arr),
    )


def profile_pandas(exp: Experiment, ids: list[str], backend: str) -> None:
    """Profile pd.Series assignment; skip if pandas is not installed.

    Parameters
    ----------
    exp : Experiment
        Pre-constructed experiment for the target backend.
    ids : list[str]
        Identifiers to assign.
    backend : str
        Backend label used in output and file stem.
    """
    try:
        import pandas as pd
    except ImportError:
        print(f"\npandas not installed — skipping pd.Series ({backend})")
        return
    series = pd.Series(ids)
    _profile(
        label=f"pd.Series  n={N:,}  {backend}  cache=False",
        file_stem=f"pandas_{backend}",
        fn=lambda: exp.assign(series),
    )


def profile_polars(exp: Experiment, ids: list[str], backend: str) -> None:
    """Profile pl.Series assignment; skip if polars is not installed.

    Parameters
    ----------
    exp : Experiment
        Pre-constructed experiment for the target backend.
    ids : list[str]
        Identifiers to assign.
    backend : str
        Backend label used in output and file stem.
    """
    try:
        import polars as pl
    except ImportError:
        print(f"\npolars not installed — skipping pl.Series ({backend})")
        return
    series = pl.Series(ids)
    _profile(
        label=f"pl.Series  n={N:,}  {backend}  cache=False",
        file_stem=f"polars_{backend}",
        fn=lambda: exp.assign(series),
    )


def _run_backend(backend: str, exp: Experiment, ids: list[str]) -> None:
    """Profile all input types for one backend.

    Parameters
    ----------
    backend : str
        Backend label (``"hashlib"`` or ``"xxhash"``).
    exp : Experiment
        Pre-constructed experiment for this backend.
    ids : list[str]
        Identifiers to assign.
    """
    profile_list(exp, ids, backend)
    profile_numpy(exp, ids, backend)
    profile_pandas(exp, ids, backend)
    profile_polars(exp, ids, backend)


# --- Entry point -------------------------------------------------------------


def run() -> None:
    """Run all profiling scenarios (all input types) and save results."""
    print(f"Profiling cohorting.Experiment.assign()  N={N:,}  SORT_BY={SORT_BY!r}")
    print(f"Results: {RESULTS_DIR.resolve()}")

    ids = _make_ids(N)

    print("\n\n--- SipHash backend ---")
    _run_backend("siphash", _make_exp(), ids)

    stems = [
        f"{input_type}_siphash" for input_type in ("list", "numpy", "pandas", "polars")
    ]
    print("\n\nTo explore visually:")
    print("  snakeviz <file.prof>")
    for stem in stems:
        path = RESULTS_DIR / f"{stem}.prof"
        if path.exists():
            print(f"  snakeviz {path}")


if __name__ == "__main__":
    run()
