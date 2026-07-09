"""Type stubs for the cohorting._core Rust extension module."""

from __future__ import annotations

from typing import Any

import numpy as np

def hash_single(id: str, sep_salt: bytes, use_xxhash: bool) -> float: ...
def hash_strings(
    ids: list[str], sep_salt: bytes, use_xxhash: bool
) -> list[float]: ...
def random_float() -> float: ...
def random_floats(n: int) -> list[float]: ...
def assign_single(
    id: str,
    sep_salt: bytes,
    use_xxhash: bool,
    cohort_names: list[str],
    lower_bounds: list[float],
) -> str: ...
def assign_strings(
    ids: list[str],
    sep_salt: bytes,
    use_xxhash: bool,
    cohort_names: list[str],
    lower_bounds: list[float],
) -> list[str]: ...
