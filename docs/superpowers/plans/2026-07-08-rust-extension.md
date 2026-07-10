# Rust Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the performance-critical hashing and cohort-assignment hot paths from Python into a Rust native extension built with PyO3 and Maturin.

**Architecture:** A Rust crate (`rust-core/`) compiled as a PyO3 extension module (`cohorting._core`). The Python layer (`_hash.py`, `_cohort.py`) becomes a thin delegation layer that normalizes input types and calls Rust functions for all hash computation and cohort assignment. Split validation, public API signatures, and the `Experiment` class stay in Python unchanged.

**Tech Stack:** Rust (edition 2021), PyO3 0.23, Maturin 1.x, blake2 0.10, twox-hash 1.6, numpy 0.23 (Rust crate)

---

## File Structure

```
cohorting/
├── rust-core/                     # NEW: Cargo workspace
│   ├── Cargo.toml                 # NEW
│   └── src/
│       ├── lib.rs                 # NEW: PyO3 module definition
│       ├── hash.rs                # NEW: Batch hashing functions
│       ├── cohort.rs              # NEW: Batch cohort assignment
│       └── utils.rs               # NEW: Shared constants (INV_2_64)
├── src/
│   └── cohorting/
│       ├── _core.pyi              # NEW: Type stubs for Rust extension
│       ├── _hash.py               # MODIFY: Thin delegation to Rust
│       ├── _cohort.py             # MODIFY: Thin delegation to Rust
│       ├── _config.py             # MODIFY: Remove _ensure_xxhash
│       └── ...
├── pyproject.toml                 # MODIFY: Maturin build system
├── Taskfile.yaml                  # MODIFY: Add build-rust tasks
└── .pre-commit-config.yaml        # MODIFY: Add Rust checks
```

## Hash Output Compatibility

**Rust is the canonical hash implementation.** Since the library is zerover (0.1.0), breaking hash-output changes are acceptable. The Rust implementation uses:

- **blake2b**: `Blake2b512` (full 64-byte) → take first 8 bytes as little-endian u64 → multiply by `INV_2_64`. This differs from Python's `blake2b(digest_size=8)` which produces different output due to the digest-size parameter affecting the hash internals.
- **xxh3_64**: one-shot `twox_hash::xxh3::hash64` over concatenated `id_bytes ++ sep_salt`. Matches Python's streaming xxh3_64 result.

Golden value tests in `test_hash.py` and `test_cohort.py` **will be updated** with new Rust-produced values.

---

### Task 1: Rust Crate Scaffold

**Files:**

- Create: `rust-core/Cargo.toml`
- Create: `rust-core/src/lib.rs`
- Create: `rust-core/src/utils.rs`

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "cohorting-core"
version = "0.1.0"
edition = "2021"

[lib]
name = "_core"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.23", features = ["extension-module"] }
blake2 = "0.10"
twox-hash = "1.6"
numpy = "0.23"
getrandom = "0.2"
```

- [ ] **Step 2: Create utils.rs**

```rust
/// Reciprocal of 2^64; multiply instead of divide to produce float in [0, 1).
pub const INV_2_64: f64 = 1.0 / ((1u128 << 64) as f64);
```

- [ ] **Step 3: Create lib.rs (skeleton module declaration)**

```rust
use pyo3::prelude::*;

mod cohort;
mod hash;
mod utils;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hash::hash_single, m)?)?;
    m.add_function(wrap_pyfunction!(hash::hash_strings, m)?)?;
    m.add_function(wrap_pyfunction!(hash::random_float, m)?)?;
    m.add_function(wrap_pyfunction!(hash::random_floats, m)?)?;
    m.add_function(wrap_pyfunction!(cohort::assign_single, m)?)?;
    m.add_function(wrap_pyfunction!(cohort::assign_strings, m)?)?;
    Ok(())
}
```

- [ ] **Step 4: Verify Rust compiles**

```bash
cargo build --manifest-path rust-core/Cargo.toml
```

Expected: Compiles successfully (warnings OK, no errors).

- [ ] **Step 5: Commit**

```bash
git add rust-core/Cargo.toml rust-core/src/lib.rs rust-core/src/utils.rs
git commit -m "➕ Add Rust crate scaffold for cohorting-core"
```

---

### Task 2: Core Hashing — hash_single and hash_strings

**Files:**

- Create: `rust-core/src/hash.rs`
- Modify: `rust-core/src/lib.rs` (register new functions — already done in skeleton)

- [ ] **Step 1: Write hash.rs with hash_single and hash_strings**

```rust
use pyo3::prelude::*;
use blake2::{Blake2b512, Digest};
use twox_hash::xxh3;

use crate::utils::INV_2_64;

/// Hash a single identifier to a float in [0, 1).
///
/// sep_salt is the pre-encoded b"\x00" + salt bytes from Python.
#[inline]
pub fn hash_single_inner(id_bytes: &[u8], sep_salt: &[u8], use_xxhash: bool) -> f64 {
    if use_xxhash {
        let mut buf = Vec::with_capacity(id_bytes.len() + sep_salt.len());
        buf.extend_from_slice(id_bytes);
        buf.extend_from_slice(sep_salt);
        xxh3::hash64(&buf) as f64 * INV_2_64
    } else {
        let mut hasher = Blake2b512::new();
        hasher.update(id_bytes);
        hasher.update(sep_salt);
        let result = hasher.finalize();
        u64::from_le_bytes(result[..8].try_into().unwrap()) as f64 * INV_2_64
    }
}

/// Hash a single string identifier to a float in [0, 1).
#[pyfunction]
pub fn hash_single(id: String, sep_salt: Vec<u8>, use_xxhash: bool) -> f64 {
    hash_single_inner(id.as_bytes(), &sep_salt, use_xxhash)
}

/// Hash a list of string identifiers to floats in [0, 1).
#[pyfunction]
pub fn hash_strings(
    ids: Vec<String>,
    sep_salt: Vec<u8>,
    use_xxhash: bool,
) -> Vec<f64> {
    ids.iter()
        .map(|id| hash_single_inner(id.as_bytes(), &sep_salt, use_xxhash))
        .collect()
}

/// Return a single random float in [0, 1) sourced from OS entropy.
#[pyfunction]
pub fn random_float() -> f64 {
    let mut buf = [0u8; 8];
    getrandom::getrandom(&mut buf).unwrap();
    u64::from_le_bytes(buf) as f64 * INV_2_64
}

/// Return n random floats in [0, 1) sourced from OS entropy.
#[pyfunction]
pub fn random_floats(n: usize) -> Vec<f64> {
    let mut buf = vec![0u8; n * 8];
    getrandom::getrandom(&mut buf).unwrap();
    buf.chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()) as f64 * INV_2_64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_single_deterministic_blake2b() {
        let a = hash_single_inner(b"user_1", b"\x00exp", false);
        let b = hash_single_inner(b"user_1", b"\x00exp", false);
        assert_eq!(a, b);
        assert!((0.0..1.0).contains(&a));
    }

    #[test]
    fn test_hash_single_deterministic_xxhash() {
        let a = hash_single_inner(b"user_1", b"\x00exp", true);
        let b = hash_single_inner(b"user_1", b"\x00exp", true);
        assert_eq!(a, b);
        assert!((0.0..1.0).contains(&a));
    }

    #[test]
    fn test_hash_different_salts() {
        let a = hash_single_inner(b"user_1", b"\x00exp1", false);
        let b = hash_single_inner(b"user_1", b"\x00exp2", false);
        assert_ne!(a, b);
    }

    #[test]
    fn test_hash_different_ids() {
        let a = hash_single_inner(b"user_1", b"\x00exp", false);
        let b = hash_single_inner(b"user_2", b"\x00exp", false);
        assert_ne!(a, b);
    }

    #[test]
    fn test_hash_strings_matches_single() {
        let sep_salt = b"\x00exp".to_vec();
        let ids: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let batch = hash_strings(ids.clone(), sep_salt.clone(), false);
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(batch[i], hash_single(id.clone(), sep_salt.clone(), false));
        }
    }

    #[test]
    fn test_random_float_in_range() {
        let r = random_float();
        assert!((0.0..1.0).contains(&r));
    }

    #[test]
    fn test_random_floats_count() {
        let r = random_floats(10);
        assert_eq!(r.len(), 10);
        for v in &r {
            assert!((0.0..1.0).contains(v));
        }
    }
}
```

Add to Cargo.toml dependencies if not already present — `getrandom` is needed. Update Cargo.toml if not already done:

```toml
getrandom = "0.2"
```

- [ ] **Step 2: Run Rust tests**

```bash
cargo test --manifest-path rust-core/Cargo.toml
```

Expected: All 7 tests pass.

- [ ] **Step 3: Commit**

```bash
git add rust-core/src/hash.rs rust-core/Cargo.toml
git commit -m "✨ Add core hashing: hash_single, hash_strings, random_float, random_floats"
```

---

### Task 3: Core Assignment — assign_single and assign_strings

**Files:**

- Create: `rust-core/src/cohort.rs`

- [ ] **Step 1: Write cohort.rs**

```rust
use pyo3::prelude::*;

use crate::hash::hash_single_inner;

/// Assign a single string identifier to a cohort.
///
/// lower_bounds must be sorted ascending. cohort_names[i] corresponds to the
/// bucket whose lower bound is lower_bounds[i].
#[pyfunction]
pub fn assign_single(
    id: String,
    sep_salt: Vec<u8>,
    use_xxhash: bool,
    cohort_names: Vec<String>,
    lower_bounds: Vec<f64>,
) -> String {
    let hash_val = hash_single_inner(id.as_bytes(), &sep_salt, use_xxhash);
    let idx = lower_bounds.partition_point(|&lb| lb <= hash_val) - 1;
    cohort_names[idx].clone()
}

/// Assign a list of string identifiers to cohorts.
#[pyfunction]
pub fn assign_strings(
    ids: Vec<String>,
    sep_salt: Vec<u8>,
    use_xxhash: bool,
    cohort_names: Vec<String>,
    lower_bounds: Vec<f64>,
) -> Vec<String> {
    ids.iter()
        .map(|id| {
            let hash_val = hash_single_inner(id.as_bytes(), &sep_salt, use_xxhash);
            let idx = lower_bounds.partition_point(|&lb| lb <= hash_val) - 1;
            cohort_names[idx].clone()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assign_single_control() {
        let names = vec!["control".into(), "treatment".into()];
        let lowers = vec![0.0, 0.5];
        let sep_salt = b"\x00exp".to_vec();

        // hash_single for "user_1" with xxhash should be deterministic
        let result1 = assign_single("user_1".into(), sep_salt.clone(), false, names.clone(), lowers.clone());
        let result2 = assign_single("user_1".into(), sep_salt.clone(), false, names.clone(), lowers.clone());
        assert_eq!(result1, result2);
        assert!(names.contains(&result1));
    }

    #[test]
    fn test_assign_strings_matches_single() {
        let names = vec!["control".into(), "treatment".into()];
        let lowers = vec![0.0, 0.5];
        let sep_salt = b"\x00exp".to_vec();
        let ids: Vec<String> = vec!["a".into(), "b".into(), "c".into()];

        let batch = assign_strings(ids.clone(), sep_salt.clone(), false, names.clone(), lowers.clone());
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(
                batch[i],
                assign_single(id.clone(), sep_salt.clone(), false, names.clone(), lowers.clone())
            );
        }
    }

    #[test]
    fn test_assign_three_way_split() {
        let names = vec!["a".into(), "b".into(), "c".into()];
        let lowers = vec![0.0, 0.3333333333333333, 0.6666666666666666];
        let sep_salt = b"\x00exp".to_vec();

        let result = assign_single("user_1".into(), sep_salt, false, names.clone(), lowers);
        assert!(names.contains(&result));
    }
}
```

- [ ] **Step 2: Run Rust tests**

```bash
cargo test --manifest-path rust-core/Cargo.toml
```

Expected: All 10 tests pass (7 hash + 3 cohort).

- [ ] **Step 3: Commit**

```bash
git add rust-core/src/cohort.rs
git commit -m "✨ Add core cohort assignment: assign_single, assign_strings"
```

---

### Task 4: Maturin Build Integration

**Files:**

- Modify: `pyproject.toml`
- Modify: `Taskfile.yaml`

- [ ] **Step 1: Update pyproject.toml build system**

Replace the `[build-system]` section:

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[tool.maturin]
module-name = "cohorting._core"
python-source = "src"
manifest-path = "rust-core/Cargo.toml"
```

- [ ] **Step 2: Update Taskfile.yaml with Rust build tasks**

Add these tasks to `Taskfile.yaml` (before the `sync:` task):

```yaml
#######################################
build-rust:
    desc: Build the Rust extension (release)
    cmds:
        - uv run maturin develop --release --manifest-path rust-core/Cargo.toml
#######################################
build-rust-dev:
    desc: Build the Rust extension (debug, fast compile)
    cmds:
        - uv run maturin develop --manifest-path rust-core/Cargo.toml
#######################################
install-dev:
    desc: Install with debug Rust build for development
    cmds:
        - task: sync
        - task: build-rust-dev
```

Update the `sync` task to include `maturin`:

```yaml
sync:
    desc: Install Project Dependencies
    internal: true
    cmds:
        - uv sync --all-extras {{.CLI_ARGS}}
        - task: build-rust-dev
    env:
        UV_PYTHON: "{{.UV_PYTHON}}"
```

Also add `maturin` to dev dependencies in `pyproject.toml`:

```toml
[dependency-groups]
dev = [
  "pytest",
  "pytest-cov",
  "maturin>=1.0,<2.0",
  ...
]
```

- [ ] **Step 3: Build the Rust extension**

```bash
task build-rust-dev
```

Expected: Extension builds and installs successfully. Verify with:

```bash
uv run python -c "import cohorting._core; print(dir(cohorting._core))"
```

Expected: Shows `hash_single`, `hash_strings`, `random_float`, `random_floats`, `assign_single`, `assign_strings`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml Taskfile.yaml uv.lock
git commit -m "🔧 Add Maturin build integration for Rust extension"
```

---

### Task 5: Python Delegation — \_hash.py

**Files:**

- Modify: `src/cohorting/_hash.py`
- Create: `src/cohorting/_core.pyi`

- [ ] **Step 1: Create \_core.pyi type stubs**

```python
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
def hash_numpy(
    ids: np.ndarray, sep_salt: bytes, use_xxhash: bool
) -> np.ndarray: ...
def random_floats_numpy(n: int) -> np.ndarray: ...
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
def assign_numpy(
    ids: np.ndarray,
    sep_salt: bytes,
    use_xxhash: bool,
    cohort_names: list[str],
    lower_bounds: list[float],
) -> np.ndarray: ...
```

- [ ] **Step 2: Rewrite \_hash.py to delegate to Rust**

Replace `_hash.py` with the thin delegation layer. Keep the public API (`hash_values`, `hash_orm`) and module-level flags unchanged. Delete all deleted symbols listed in the design doc:

```python
"""Hash function for producing deterministic floats from string identifiers."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast, overload

from cohorting._core import (
    hash_single as _rust_hash_single,
    hash_strings as _rust_hash_strings,
    random_float as _rust_random_float,
    random_floats as _rust_random_floats,
)
from cohorting._models import _is_numpy_array, _is_pandas_series, _is_polars_series

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    import polars as pl

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
    """Apply the selected hash backend to all supported data types."""
    if not use_deterministic:
        if isinstance(data, (str, int)):
            norm: str | int = int(data) if isinstance(data, bool) else data
            return _rust_random_float()
        if isinstance(data, list):
            return _rust_random_floats(len(data))
        if _is_numpy_array(data):
            import numpy as _np
            raw_arr = _np.frombuffer(os.urandom(data.size * 8), dtype=_np.uint64)
            return (raw_arr.astype(_np.float64) * (1.0 / (2**64))).reshape(data.shape)
        if _is_pandas_series(data):
            import numpy as _np
            import pandas as _pd
            raw_arr = _np.frombuffer(os.urandom(len(data) * 8), dtype=_np.uint64)
            return _pd.Series(
                data=raw_arr.astype(_np.float64) * (1.0 / (2**64)),
                index=data.index,
                name=data.name,
            )
        if _is_polars_series(data):
            n = len(data)
            raw_bytes = os.urandom(n * 8)
            import polars as _pl
            return _pl.Series(
                name=data.name,
                values=[
                    int.from_bytes(raw_bytes[i * 8 : (i + 1) * 8], "little")
                    * (1.0 / (2**64))
                    for i in range(n)
                ],
            )
        raise TypeError(
            f"hash_values expected str, list, np.ndarray, pd.Series, or pl.Series; "
            f"got {type(data).__name__}"
        )

    if isinstance(data, (str, int)):
        norm: str | int = int(data) if isinstance(data, bool) else data
        norm_id = str(norm)
        if use_cache:
            return _cached_hash_single(norm_id, sep_salt, use_xxhash)
        return _rust_hash_single(norm_id, sep_salt, use_xxhash)

    if isinstance(data, list):
        norm_list = [str(int(x) if isinstance(x, bool) else x) for x in data]
        result = _rust_hash_strings(norm_list, sep_salt, use_xxhash)
        if use_cache:
            return [_cached_hash_single(x, sep_salt, use_xxhash) for x in norm_list]
        return result

    if _is_numpy_array(data):
        import numpy as _np
        flat = data.flatten()
        str_ids = [str(int(x) if isinstance(x, bool) else x) for x in flat]
        result_flat = _rust_hash_strings(str_ids, sep_salt, use_xxhash)
        return _np.array(result_flat, dtype=_np.float64).reshape(data.shape)

    if _is_pandas_series(data):
        import pandas as _pd
        str_ids = [str(int(x) if isinstance(x, bool) else x) for x in data]
        result = _rust_hash_strings(str_ids, sep_salt, use_xxhash)
        return _pd.Series(result, index=data.index, name=data.name)

    if _is_polars_series(data):
        import polars as _pl
        str_ids = [str(int(x) if isinstance(x, bool) else x) for x in data]
        result = _rust_hash_strings(str_ids, sep_salt, use_xxhash)
        return _pl.Series(name=data.name, values=result)

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

    The default backend is hashlib blake2b (always available, compiled in Rust).
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
    """
    return _dispatch_hash(
        data,
        sep_salt=b"\x00" + salt.encode(),
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
    """
    if isinstance(obj, list):
        return hash_values([cast(str, getattr(o, id_field)) for o in obj], salt=salt)
    return hash_values(cast(str, getattr(obj, id_field)), salt=salt)
```

- [ ] **Step 3: Update \_config.py — remove \_ensure_xxhash**

In `_config.py`, replace the `xxhash.setter` method to remove the call to `_ensure_xxhash()`:

```python
@xxhash.setter
def xxhash(self, enable: bool) -> None:
    import cohorting._cohort as _cohort_mod
    import cohorting._hash as _hash_mod

    _hash_mod._USE_XXHASH = enable
    _hash_mod._cached_hash_single.cache_clear()
    _cohort_mod._cached_assign_single.cache_clear()
```

- [ ] **Step 4: Run existing tests to see which pass/fail**

```bash
task test -- tests/test_hash.py -x
```

Expected: Tests that check golden hash values will fail (expected — Rust produces different blake2b output). Tests that check range, shape, types should pass.

- [ ] **Step 5: Update golden value tests**

Update `test_hash_hashlib_known_value` in `tests/test_hash.py`. The Rust blake2b implementation uses full `Blake2b512` (taking the first 8 bytes) which produces a different value than Python's `blake2b(digest_size=8)`. Run this command to get the new golden value, then replace in the test:

```bash
uv run python -c "from cohorting._core import hash_single; print(hash_single('user_123', b'\x00exp', False))"
```

Take the printed float value and update the test:

```python
def test_hash_hashlib_known_value() -> None:
    """Default hashlib backend produces the expected deterministic float.

    Golden value: Blake2b512 (first 8 bytes as little-endian u64 / 2^64)
    over "user_123" + b"\\x00exp".
    """
    expected = <value from command above>
    assert hash_values("user_123", salt="exp") == expected
```

Also update `test_hash_xxhash_known_value`. The Rust xxh3_64 one-shot hash over concatenated bytes should match Python's streaming xxh3_64 result. Verify and update:

```bash
uv run python -c "from cohorting._core import hash_single; print(hash_single('user_123', b'\x00exp', True))"
```

Replace the expected value in the test and remove the `pytest.importorskip("xxhash")` call since xxhash is now compiled into Rust.

- [ ] **Step 6: Run tests to verify**

```bash
task test -- tests/test_hash.py -v
```

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/cohorting/_hash.py src/cohorting/_config.py src/cohorting/_core.pyi tests/test_hash.py
git commit -m "✨ Delegate hash computation to Rust _core extension"
```

---

### Task 6: Python Delegation — \_cohort.py

**Files:**

- Modify: `src/cohorting/_cohort.py`

- [ ] **Step 1: Rewrite \_cohort.py to delegate to Rust**

Replace `_cohort.py` with a thin delegation layer. The key changes:

- Import `assign_single` and `assign_strings` from `_core`
- Replace `_assign_hashlib` and `_assign_xxhash` with a single `_cached_assign_single`
- Replace hot-path `_dispatch_assign` branches with Rust calls
- Keep `_splits_to_sorted_bounds`, `_get_lower_bounds` in Python

```python
"""Cohort assignment functions."""

from __future__ import annotations

import bisect
from functools import cache, lru_cache
from typing import TYPE_CHECKING, Any, cast, overload

import cohorting._hash as _hash_mod
from cohorting._core import (
    assign_single as _rust_assign_single,
    assign_strings as _rust_assign_strings,
    random_float as _rust_random_float,
    random_floats as _rust_random_floats,
    random_floats_numpy as _rust_random_floats_numpy,
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
    """Convert a SplitMap to a sorted tuple for bisect-based assignment."""
    return tuple(
        sorted(
            ((name, b["lower"], b["upper"]) for name, b in splits.items()),
            key=lambda t: t[1],
        )
    )


@cache
def _get_lower_bounds(sorted_bounds: _SortedBounds) -> tuple[float, ...]:
    """Extract lower bounds for bisect lookup, cached per unique experiment config."""
    return tuple(b[1] for b in sorted_bounds)


@cache
def _get_cohort_names(sorted_bounds: _SortedBounds) -> tuple[str, ...]:
    """Extract cohort names in sorted order, cached per unique experiment config."""
    return tuple(b[0] for b in sorted_bounds)


@lru_cache(maxsize=65_536)
def _cached_assign_single(
    x: str, sep_salt: bytes, use_xxhash: bool, sorted_bounds: _SortedBounds
) -> str:
    """Cached hash + assign via Rust."""
    lowers = _get_lower_bounds(sorted_bounds)
    names = _get_cohort_names(sorted_bounds)
    return _rust_assign_single(
        x, sep_salt, use_xxhash, list(names), list(lowers)
    )


def _dispatch_assign(
    data: Any,
    *,
    sep_salt: bytes,
    sorted_bounds: _SortedBounds,
    use_xxhash: bool,
    use_deterministic: bool,
    use_cache: bool,
) -> Any:
    """Apply cohort assignment to all supported data types."""
    def _random_assign(hash_val: float) -> str:
        """Map a random float to a cohort name via bisect."""
        lowers = _get_lower_bounds(sorted_bounds)
        names_list = list(_get_cohort_names(sorted_bounds))
        idx = bisect.bisect_right(lowers, hash_val) - 1
        return names_list[idx]

    if not use_deterministic:
        if isinstance(data, (str, int)):
            return _random_assign(_rust_random_float())
        if isinstance(data, list):
            return [_random_assign(_rust_random_float()) for _ in data]
        if _is_numpy_array(data):
            import numpy as _np
            floats = _rust_random_floats_numpy(data.size)
            lowers = _get_lower_bounds(sorted_bounds)
            names_arr = _np.array(list(_get_cohort_names(sorted_bounds)))
            indices = _np.searchsorted(lowers, floats, side="right") - 1
            return names_arr[indices].reshape(data.shape)
        if _is_pandas_series(data):
            import numpy as _np
            import pandas as _pd
            floats = _rust_random_floats_numpy(len(data))
            lowers = _get_lower_bounds(sorted_bounds)
            names_arr = _np.array(list(_get_cohort_names(sorted_bounds)))
            indices = _np.searchsorted(lowers, floats, side="right") - 1
            return _pd.Series(names_arr[indices], index=data.index, name=data.name)
        if _is_polars_series(data):
            import polars as _pl
            floats = _rust_random_floats(len(data))
            lowers = _get_lower_bounds(sorted_bounds)
            names_list = list(_get_cohort_names(sorted_bounds))
            result = [
                names_list[bisect.bisect_right(lowers, f) - 1] for f in floats
            ]
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
    """
    split_map = _normalize_splits(splits)
    validate_splits(split_map)
    sep_salt = b"\x00" + salt.encode()
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
    """
    if isinstance(obj, list):
        return assign_cohorts(
            [cast(str, getattr(o, id_field)) for o in obj], splits=splits, salt=salt
        )
    return assign_cohorts(cast(str, getattr(obj, id_field)), splits=splits, salt=salt)
```

- [ ] **Step 2: Run cohort tests**

```bash
task test -- tests/test_cohort.py tests/test_experiment.py -x
```

Expected: Golden value tests will fail. Fix them by getting new Rust-produced values and updating test assertions.

- [ ] **Step 3: Update golden values in test_cohort.py**

Determine which tests have hardcoded expected cohort names based on deterministic hashing. Those need updating with Rust-produced outputs:

```bash
uv run python -c "
from cohorting._core import assign_single
result = assign_single('user_1', b'\x00exp', False, ['control', 'treatment'], [0.0, 0.5])
print(result)
"
```

Update the affected test assertions in `test_cohort.py`.

- [ ] **Step 4: Run the full test suite**

```bash
task test
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cohorting/_cohort.py tests/test_cohort.py tests/test_experiment.py
git commit -m "✨ Delegate cohort assignment to Rust _core extension"
```

---

### Task 7: NumPy Interop

**Files:**

- Modify: `rust-core/src/hash.rs` (add `hash_numpy`, `random_floats_numpy`)
- Modify: `rust-core/src/cohort.rs` (add `assign_numpy`)
- Modify: `rust-core/src/lib.rs` (register new functions)
- Modify: `src/cohorting/_hash.py` (use `hash_numpy`)
- Modify: `src/cohorting/_cohort.py` (use `assign_numpy`)

- [ ] **Step 1: Add hash_numpy and random_floats_numpy to hash.rs**

Append to `rust-core/src/hash.rs`:

```rust
use numpy::{PyArray1, PyReadonlyArray1};
use numpy::types::PyUnicode;

/// Hash a numpy array of strings to floats in [0, 1).
#[pyfunction]
pub fn hash_numpy<'py>(
    py: Python<'py>,
    ids: PyReadonlyArray1<'py, PyUnicode>,
    sep_salt: Vec<u8>,
    use_xxhash: bool,
) -> Bound<'py, PyArray1<f64>> {
    let n = ids.len();
    let out = PyArray1::<f64>::zeros(py, n, false);
    unsafe {
        let out_slice = out.as_slice_mut().unwrap();
        for (i, id) in ids.as_slice().unwrap().iter().enumerate() {
            let id_str: &str = id.to_str().unwrap();
            out_slice[i] = hash_single_inner(id_str.as_bytes(), &sep_salt, use_xxhash);
        }
    }
    out
}

/// Return n random floats as a numpy array.
#[pyfunction]
pub fn random_floats_numpy<'py>(
    py: Python<'py>,
    n: usize,
) -> Bound<'py, PyArray1<f64>> {
    let mut buf = vec![0u8; n * 8];
    getrandom::getrandom(&mut buf).unwrap();
    let out = PyArray1::<f64>::zeros(py, n, false);
    unsafe {
        let out_slice = out.as_slice_mut().unwrap();
        for (i, chunk) in buf.chunks_exact(8).enumerate() {
            let val = u64::from_le_bytes(chunk.try_into().unwrap());
            out_slice[i] = val as f64 * INV_2_64;
        }
    }
    out
}
```

- [ ] **Step 2: Add assign_numpy to cohort.rs**

Append to `rust-core/src/cohort.rs`:

```rust
use numpy::{PyArray1, PyReadonlyArray1};
use numpy::types::PyUnicode;

/// Assign cohorts to a numpy array of string identifiers.
#[pyfunction]
pub fn assign_numpy<'py>(
    py: Python<'py>,
    ids: PyReadonlyArray1<'py, PyUnicode>,
    sep_salt: Vec<u8>,
    use_xxhash: bool,
    cohort_names: Vec<String>,
    lower_bounds: Vec<f64>,
) -> Bound<'py, PyArray1<PyUnicode>> {
    let n = ids.len();
    let out = PyArray1::<PyUnicode>::zeros(py, n, false);
    unsafe {
        let out_slice = out.as_slice_mut().unwrap();
        for (i, id) in ids.as_slice().unwrap().iter().enumerate() {
            let id_str: &str = id.to_str().unwrap();
            let hash_val = hash_single_inner(id_str.as_bytes(), &sep_salt, use_xxhash);
            let idx = lower_bounds.partition_point(|&lb| lb <= hash_val) - 1;
            out_slice[i] = cohort_names[idx].clone();
        }
    }
    out
}
```

Note: `PyArray1<PyUnicode>` for string output requires creating Python string objects. The exact API may need adjustment — if `PyUnicode` array writing is complex, fall back to returning `Vec<String>` from a helper that iterates the numpy array.

- [ ] **Step 3: Register new functions in lib.rs**

Update the `#[pymodule]` in `lib.rs`:

```rust
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hash::hash_single, m)?)?;
    m.add_function(wrap_pyfunction!(hash::hash_strings, m)?)?;
    m.add_function(wrap_pyfunction!(hash::hash_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(hash::random_float, m)?)?;
    m.add_function(wrap_pyfunction!(hash::random_floats, m)?)?;
    m.add_function(wrap_pyfunction!(hash::random_floats_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(cohort::assign_single, m)?)?;
    m.add_function(wrap_pyfunction!(cohort::assign_strings, m)?)?;
    m.add_function(wrap_pyfunction!(cohort::assign_numpy, m)?)?;
    Ok(())
}
```

- [ ] **Step 4: Update Python delegation to use numpy functions**

In `_hash.py`, update the numpy path:

```python
if _is_numpy_array(data):
    import numpy as _np
    # Convert to string array for the Rust call
    str_arr = data.astype(str)
    result = _rust_hash_numpy(str_arr, sep_salt, use_xxhash)
    return result.reshape(data.shape)
```

In `_cohort.py`, update the numpy path similarly:

```python
if _is_numpy_array(data):
    str_arr = data.astype(str)
    result = _rust_assign_numpy(
        str_arr, sep_salt, use_xxhash, list(names), list(lowers)
    )
    return result.reshape(data.shape)
```

Add the new imports:

```python
from cohorting._core import (
    hash_single as _rust_hash_single,
    hash_strings as _rust_hash_strings,
    hash_numpy as _rust_hash_numpy,
    random_float as _rust_random_float,
    random_floats as _rust_random_floats,
    random_floats_numpy as _rust_random_floats_numpy,
)
```

- [ ] **Step 5: Build and test**

```bash
task build-rust-dev
task test -- tests/test_hash.py tests/test_cohort.py -v
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add rust-core/src/hash.rs rust-core/src/cohort.rs rust-core/src/lib.rs src/cohorting/_hash.py src/cohorting/_cohort.py
git commit -m "✨ Add NumPy interop: hash_numpy, assign_numpy, random_floats_numpy"
```

---

### Task 8: Polars Interop

**Files:**

- Modify: `rust-core/src/hash.rs` (add `hash_polars`)
- Modify: `rust-core/src/cohort.rs` (add `assign_polars`)
- Modify: `rust-core/src/lib.rs` (register new functions)
- Modify: `src/cohorting/_hash.py` (use `hash_polars`)
- Modify: `src/cohorting/_cohort.py` (use `assign_polars`)

- [ ] **Step 1: Research Polars Series construction from Rust**

Polars Python Series can be constructed from Rust by accepting a `Bound<PyAny>` and extracting string chunks using `PySeriesMethods` or by calling into Polars' Python C API. The simplest approach: accept the Polars Series as a `Bound<PyAny>`, call `.to_list()` on it to get a Python list, then hash in Rust and return a new Polars Series constructed via `PyModule::import("polars")`.

However, this is slow. The fast approach: use the `polars` crate's Python bindings. For simplicity and correctness, the initial implementation can use the list-based approach and a note can be added for future optimization.

Given the complexity, the pragmatic approach: handle Polars in Python by converting to a list, hashing in Rust via `hash_strings`/`assign_strings`, and constructing a new Polars Series in Python. This avoids the `polars` Rust crate dependency and is simpler to implement.

For this task, **Polars interop stays in Python as a thin wrapper around the Rust list-based functions.**

- [ ] **Step 1 (revised): Verify Polars path already works via list conversion**

The `_hash.py` and `_cohort.py` already handle Polars by converting values to a Python list, calling `_rust_hash_strings`/`_rust_assign_strings`, and wrapping the result in a `pl.Series`. No Rust changes needed for Polars.

- [ ] **Step 2: Run polars tests**

```bash
task test -- tests/test_hash.py::test_hash_polars_series_returns_series tests/test_hash.py::test_hash_polars_series_values_in_range -v
```

Expected: Pass.

- [ ] **Step 3: Commit** (skip if no changes needed, or document decision)

If no Rust Polars interop needed, skip this commit. Otherwise:

```bash
git commit -m "✨ Polars interop via Python list bridge to Rust"
```

---

### Task 9: Cleanup and Final Integration

**Files:**

- Modify: `src/cohorting/_config.py` (remove \_ensure_xxhash call, update cache clearing)
- Modify: `src/cohorting/_hash.py` (remove unused imports)
- Modify: `src/cohorting/_cohort.py` (remove unused imports)
- Modify: `src/cohorting/_core.pyi` (finalize type stubs)
- Modify: `pyproject.toml` (remove xxhash optional dependency, deprecate extra)

- [ ] **Step 1: Deprecate xxhash extra in pyproject.toml**

Change `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
xxhash = []  # No-op stub; xxhash is now compiled into the Rust extension
numpy = ["numpy>=1.24"]
pandas = ["pandas>=2.0"]
polars = ["polars>=0.20"]
```

Remove `xxhash>=3.0` from `[dependency-groups].dev`.

- [ ] **Step 2: Clean up \_config.py**

Remove any remaining references to `_ensure_xxhash`. The `xxhash.setter` no longer needs to import-check xxhash since both backends are compiled in Rust:

```python
@xxhash.setter
def xxhash(self, enable: bool) -> None:
    import cohorting._cohort as _cohort_mod
    import cohorting._hash as _hash_mod

    _hash_mod._USE_XXHASH = enable
    _hash_mod._cached_hash_single.cache_clear()
    _cohort_mod._cached_assign_single.cache_clear()
```

- [ ] **Step 3: Clean up \_hash.py**

Remove the `_ensure_xxhash` import if it was still referenced. Remove any remaining `_compute_hashlib`, `_compute_xxhash`, `_hash_hashlib`, `_hash_xxhash` references. Remove `_INV_2_64` (now in Rust). Remove `_VEC_HASH_*` globals (replaced by Rust numpy functions). Remove `_get_vec_hash`. Remove `_xxhash_mod`.

- [ ] **Step 4: Clean up \_cohort.py**

Remove `_assign_hashlib` and `_assign_xxhash` if still present. Remove `bisect` import (assignment logic moved to Rust). Remove `_lookup_cohort`.

- [ ] **Step 5: Run full test suite**

```bash
task test
```

Expected: All tests pass.

- [ ] **Step 6: Run linting and type checking**

```bash
task fix
task lint
task check
```

Expected: No errors.

- [ ] **Step 7: Final commit**

```bash
git add -A
git commit -m "🔥 Remove Python hash implementations, delegate fully to Rust"
```

---

### Task 10: Pre-commit and CI Updates

**Files:**

- Modify: `.pre-commit-config.yaml`

- [ ] **Step 1: Add Rust hooks to pre-commit config**

Read the current `.pre-commit-config.yaml` and add Rust checks:

```yaml
- repo: local
  hooks:
      - id: cargo-fmt
        name: Rust format
        entry: cargo fmt --check
        language: system
        files: ^rust-core/
        pass_filenames: false
      - id: cargo-clippy
        name: Rust lint
        entry: cargo clippy --manifest-path rust-core/Cargo.toml -- -D warnings
        language: system
        files: ^rust-core/
        pass_filenames: false
```

- [ ] **Step 2: Verify pre-commit passes**

```bash
uv tool run --isolated pre-commit run --all-files
```

- [ ] **Step 3: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "🔧 Add Rust formatting and linting to pre-commit hooks"
```

---

## Verification Checklist

Before declaring the implementation complete, verify:

- [ ] `task test` — all Python tests pass
- [ ] `task lint` — Ruff checks pass
- [ ] `task fix` — Ruff formatting passes
- [ ] `task check` — Mypy type checking passes
- [ ] `cargo test --manifest-path rust-core/Cargo.toml` — all Rust tests pass
- [ ] `cargo clippy --manifest-path rust-core/Cargo.toml -- -D warnings` — no Rust warnings
- [ ] `uv run python -c "import cohorting; print(cohorting.hash_values('test', salt='s'))"` — basic smoke test
- [ ] `uv run python -c "import cohorting; from cohorting import Experiment, even_split; exp = Experiment(name='test', splits=even_split(['a','b'])); print(exp.assign('user_1'))"` — Experiment smoke test
