"""Type stubs for the cohorting._core Rust extension module."""

from __future__ import annotations

import numpy as np

def hash_single(id: str, sep_salt: bytes) -> float: ...
def hash_strings(ids: list[str], sep_salt: bytes) -> list[float]: ...
def hash_numpy(ids: np.ndarray, sep_salt: bytes) -> np.ndarray: ...
def random_float() -> float: ...
def random_floats(n: int) -> list[float]: ...
def random_floats_numpy(n: int) -> np.ndarray: ...
def assign_single(
    id: str,
    sep_salt: bytes,
    cohort_names: list[str],
    lower_bounds: list[float],
) -> str: ...
def assign_strings(
    ids: list[str],
    sep_salt: bytes,
    cohort_names: list[str],
    lower_bounds: list[float],
) -> list[str]: ...
def assign_numpy(
    ids: np.ndarray,
    sep_salt: bytes,
    cohort_names: list[str],
    lower_bounds: list[float],
) -> list[str]: ...
