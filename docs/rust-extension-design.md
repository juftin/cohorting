# Rust Extension Design: `cohorting` Performance Core

## Summary

Move the performance-critical hashing and cohort-assignment hot paths from Python into a Rust
native extension built with PyO3 and Maturin. The Python layer becomes a thin delegation
layer — all hash computation and cohort assignment happens in Rust.

## Motivation

The current Python implementation has three bottlenecks that a native extension can eliminate:

| Bottleneck                                       | Root Cause                                                                                                                     | Rust Fix                                                                                                 |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| Per-element Python call overhead                 | `_compute_hashlib` / `_compute_xxhash` called once per identifier via list comprehension, `np.vectorize`, or `pl.map_elements` | Batch-hash N identifiers in a single Rust call with a tight loop                                         |
| `np.vectorize` is a Python for-loop              | `np.vectorize` does not compile or JIT — it runs a Python callable per element                                                 | Accept `numpy` arrays via `numpy` crate, iterate in Rust, return `PyArray1`                              |
| `pl.map_elements` is a Python for-loop           | Polars calls a Python lambda per element via the GIL                                                                           | Accept Polars Series via PyO3, call into Polars Rust internals or fall back to chunked iteration in Rust |
| String encoding + bool normalization per element | `x.encode()`, `str(x)`, `isinstance(x, bool)` in the hot loop                                                                  | Normalize and encode in Rust; `bool` → `"True"`/`"False"` is a single branch                             |

Expected throughput improvement: **5–20×** for batched inputs (arrays/series), **2–5×** for
single-element and small-list inputs (where Python call overhead dominates less).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Python (thin delegation layer)                         │
│  src/cohorting/_hash.py    _cohort.py    experiment.py  │
│       │                        │                        │
│       ▼                        ▼                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Delegates directly to Rust _core module         │   │
│  │  - Type normalization (bool→int, etc.)           │   │
│  │  - Split validation & sorting (stays in Python)  │   │
│  │  - Calls rust _core functions for hot paths      │   │
│  └────────────────────┬─────────────────────────────┘   │
│                       │                                  │
└───────────────────────┼──────────────────────────────────┘
                        │
┌───────────────────────┼──────────────────────────────────┐
│  Rust (cohorting._core)                                  │
│                       ▼                                  │
│  ┌──────────────────────────────────────────────────┐   │
│  │  hash.rs: batch hashing (blake2b + xxhash)       │   │
│  │  - hash_single(str, salt) -> float               │   │
│  │  - hash_strings(list[str], salt) -> list[float]  │   │
│  │  - hash_numpy(array, salt) -> array[float64]     │   │
│  │  - hash_polars(series, salt) -> series[float64]  │   │
│  ├──────────────────────────────────────────────────┤   │
│  │  cohort.rs: batch assignment                     │   │
│  │  - assign_single(str, bounds, salt) -> str       │   │
│  │  - assign_strings(list[str], bounds, salt)       │   │
│  │  - assign_numpy(array, bounds, salt)             │   │
│  │  - assign_polars(series, bounds, salt)           │   │
│  ├──────────────────────────────────────────────────┤   │
│  │  types.rs: Python↔Rust type bridging             │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

## Detailed Design

### 1. Project Structure

```
cohorting/
├── rust-core/                    # Cargo workspace
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                # PyO3 module definition (pymodule)
│       ├── hash.rs               # Batch hashing
│       ├── cohort.rs             # Batch cohort assignment
│       ├── types.rs              # Python type extraction & conversion
│       └── utils.rs              # Shared constants, _INV_2_64, etc.
├── src/
│   └── cohorting/
│       ├── _core.pyi             # Type stubs for the Rust extension
│       ├── _hash.py              # (modified) thin delegation to Rust
│       ├── _cohort.py            # (modified) thin delegation to Rust
│       └── ...
├── pyproject.toml                # (modified) add maturin build deps
└── Taskfile.yaml                 # (modified) add build-rust task
```

### 2. Rust Crate Setup

**`rust-core/Cargo.toml`**:

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
twox-hash = "1.6"        # xxh3_64 implementation
numpy = "0.23"            # np.ndarray interop — required at build time
```

### 3. Core Rust Functions

#### 3.1 Batch Hashing (`hash.rs`)

```rust
use pyo3::prelude::*;
use blake2::{Blake2b512, Digest};  // use digest_size=8 via wrapper
use twox_hash::xxh3::hash64;

const INV_2_64: f64 = 1.0 / ((1u128 << 64) as f64);

/// Hash a list of strings to floats in [0, 1).
/// sep_salt is pre-encoded b"\x00" + salt.
#[pyfunction]
fn hash_strings(
    ids: Vec<String>,
    sep_salt: Vec<u8>,
    use_xxhash: bool,
) -> Vec<f64> {
    ids.iter().map(|id| hash_single(id.as_bytes(), &sep_salt, use_xxhash)).collect()
}

/// Inlineable core: no alloc per element.
#[inline]
fn hash_single(id_bytes: &[u8], sep_salt: &[u8], use_xxhash: bool) -> f64 {
    if use_xxhash {
        // xxh3_64 with seed from salt (or concatenate, matching Python behavior)
        // Python: h.update(x.encode()) then h.update(sep_salt) then h.intdigest()
        // For xxh3_64 streaming, we must hash the concatenation. For one-shot:
        let mut buf = Vec::with_capacity(id_bytes.len() + sep_salt.len());
        buf.extend_from_slice(id_bytes);
        buf.extend_from_slice(sep_salt);
        hash64(&buf) as u64 as f64 * INV_2_64
    } else {
        let mut hasher = Blake2b512::new_with_salt_and_personal(
            &[], &[], b"",  // match Python blake2b defaults
        )
        .unwrap();
        // Use digest_size=8 equivalent
        hasher.update(id_bytes);
        hasher.update(sep_salt);
        let result = hasher.finalize();
        u64::from_le_bytes(result[..8].try_into().unwrap()) as f64 * INV_2_64
    }
}
```

**Key design decision: Rust is the canonical hash implementation.** Since the library is
zerover (0.1.0), breaking hash-output changes are acceptable across versions. The Rust
implementation defines the hash function — there is no legacy Python output to match. We
can use simpler one-shot hashing and choose the fastest available primitives without
worrying about cross-implementation compatibility.

#### 3.2 Batch Cohort Assignment (`cohort.rs`)

```rust
/// Pre-sorted bounds as Vec<(name, lower, upper)>.
/// The bisect is done in Rust via `partition_point`.
#[pyfunction]
fn assign_strings(
    ids: Vec<String>,
    sep_salt: Vec<u8>,
    use_xxhash: bool,
    sorted_names: Vec<String>,
    lower_bounds: Vec<f64>,
) -> Vec<String> {
    ids.iter()
        .map(|id| {
            let hash_val = hash_single(id.as_bytes(), &sep_salt, use_xxhash);
            // bisect_right on lower_bounds
            let idx = lower_bounds.partition_point(|&lb| lb <= hash_val) - 1;
            sorted_names[idx].clone()
        })
        .collect()
}
```

#### 3.3 NumPy Integration (`hash.rs`)

```rust
use numpy::{PyArray1, PyReadonlyArray1};
use numpy::types::PyUnicode;

#[pyfunction]
fn hash_numpy<'py>(
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
            out_slice[i] = hash_single(id_str.as_bytes(), &sep_salt, use_xxhash);
        }
    }
    out
}
```

For the non-deterministic path (OS entropy), use `getrandom` crate:

```rust
use getrandom::getrandom;

#[pyfunction]
fn random_floats(py: Python<'_>, n: usize) -> Bound<'_, PyArray1<f64>> {
    let mut buf = vec![0u8; n * 8];
    getrandom(&mut buf).unwrap();
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

### 4. Python Delegation Layer

`_hash.py` and `_cohort.py` become thin wrappers. All hash computation code
(`_compute_hashlib`, `_compute_xxhash`, `_hash_hashlib`, `_hash_xxhash`, the `np.vectorize`
wrappers, the list-comprehension paths, the `pl.map_elements` paths) is deleted and replaced
with direct calls into `_core`.

Split validation, split normalization, and the public API signatures stay in Python
unchanged.

```python
# _hash.py — after migration
from cohorting._core import (
    hash_single,
    hash_strings,
    hash_numpy,
    hash_polars,
    random_floats,
    random_floats_numpy,
)

def _dispatch_hash(data, *, sep_salt, use_xxhash, use_deterministic, use_cache):
    if not use_deterministic:
        return _dispatch_random(data)

    if use_cache:
        return _dispatch_cached(data, sep_salt, use_xxhash)

    # Hot path: delegate to Rust
    if isinstance(data, (str, int)):
        norm = int(data) if isinstance(data, bool) else data
        return hash_single(str(norm).encode(), sep_salt, use_xxhash)

    if isinstance(data, list):
        return hash_strings([str(int(x) if isinstance(x, bool) else x) for x in data],
                           sep_salt, use_xxhash)

    if _is_numpy_array(data):
        return hash_numpy(data, sep_salt, use_xxhash)

    if _is_pandas_series(data):
        import pandas as pd
        result = hash_numpy(data.to_numpy(), sep_salt, use_xxhash)
        return pd.Series(result, index=data.index, name=data.name)

    if _is_polars_series(data):
        return hash_polars(data, sep_salt, use_xxhash)
    ...
```

The `use_cache=True` path can initially use a Python-side LRU wrapper around the Rust
single-element call (`@lru_cache` wrapping `hash_single`), or defer caching to phase 5
of the rollout.

### 5. What Stays vs. What Gets Deleted

**Deleted from `_hash.py`:**

| Symbol                                                                                              | Reason                                                     |
| --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `_compute_hashlib`                                                                                  | Replaced by `_core.hash_single`                            |
| `_compute_xxhash`                                                                                   | Replaced by `_core.hash_single`                            |
| `_hash_hashlib`                                                                                     | Replaced by `_core.hash_single`                            |
| `_hash_xxhash`                                                                                      | Replaced by `_core.hash_single`                            |
| `_INV_2_64`                                                                                         | Moves to Rust `utils.rs`                                   |
| `_VEC_HASH_HASHLIB` / `_VEC_HASH_XXHASH` / `_VEC_HASH_HASHLIB_NOCACHE` / `_VEC_HASH_XXHASH_NOCACHE` | `np.vectorize` wrappers replaced by `_core.hash_numpy`     |
| `_xxhash_mod`                                                                                       | No longer needed; Rust ships both backends unconditionally |
| `_ensure_xxhash`                                                                                    | No longer needed                                           |
| `_get_vec_hash`                                                                                     | Replaced by `_core.hash_numpy`                             |
| List-comprehension paths in `_dispatch_hash`                                                        | Replaced by `_core.hash_strings`                           |
| `pl.map_elements` path in `_dispatch_hash`                                                          | Replaced by `_core.hash_polars`                            |

**Deleted from `_cohort.py`:**

| Symbol                                       | Reason                                                                                                                                                                 |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `_assign_hashlib`                            | Replaced by `_core.assign_single`                                                                                                                                      |
| `_assign_xxhash`                             | Replaced by `_core.assign_single`                                                                                                                                      |
| `_splits_to_sorted_bounds`                   | Moves to Rust (pre-computed bounds stay in Python, but sorting logic is dead once Python no longer hashes) — actually stays; bounds are passed to Rust as sorted lists |
| `_get_lower_bounds`                          | Stays (used to extract lower bounds for the Rust call)                                                                                                                 |
| `_lookup_cohort`                             | Moves to Rust `cohort.rs`                                                                                                                                              |
| `np.vectorize` paths in `_dispatch_assign`   | Replaced by `_core.assign_numpy`                                                                                                                                       |
| `pl.map_elements` path in `_dispatch_assign` | Replaced by `_core.assign_polars`                                                                                                                                      |

**Stays in Python:**

| Symbol                                                    | Reason                                                                               |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `hash_values`, `hash_orm`                                 | Public API                                                                           |
| `assign_cohorts`, `assign_cohorts_to_frame`, `assign_orm` | Public API                                                                           |
| `Experiment` class                                        | Public API; delegates to `_core` internally                                          |
| `_Config` / `config`                                      | Controls backend/determinism/cache flags                                             |
| `_models.py` (entire module)                              | Split definitions, type guards — no hashing logic                                    |
| `_validate.py`                                            | Split validation — no hashing logic                                                  |
| `_USE_XXHASH`, `_USE_DETERMINISTIC`, `_USE_CACHE`         | Module-level flags; read by the Python delegation layer to choose Rust function args |

### 6. `xxhash` Optional Dependency

Currently `xxhash` is a Python package installed via `cohorting[xxhash]`. Once both
blake2b and xxh3_64 are compiled into the Rust crate unconditionally, the Python
`xxhash` package is no longer used.

**Plan:**

- Remove `xxhash` from `[project.optional-dependencies]` and `[dependency-groups].dev`.
- The `xxhash` extra becomes a no-op stub (empty dependency list) for one release cycle
  so existing `pip install cohorting[xxhash]` invocations don't break.
- The `cohorting.config.xxhash = True` setter no longer calls `_ensure_xxhash()` — the
  import check is gone because Rust ships both backends.
- Document that the `[xxhash]` extra is deprecated and will be removed in 0.3.0.

### 7. Error Handling

Rust panics from PyO3 functions surface as `pyo3_runtime.PanicException` in Python. We
want to translate expected failure modes into standard Python exception types.

| Scenario                                             | Rust behavior                                                                                           | Python exception                                                                                                                                          |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Invalid salt (non-UTF8 bytes after `\x00` separator) | `hash_single` accepts raw `&[u8]` — no failure possible                                                 | N/A (salt is pre-validated in Python)                                                                                                                     |
| Empty splits list                                    | `assign_strings` receives empty `lower_bounds` → `partition_point` returns 0 → index underflow on `- 1` | Caught in Python before the Rust call (validation stays in Python)                                                                                        |
| Allocating output array fails (OOM)                  | `PyArray1::zeros` returns an Err                                                                        | Propagated as `MemoryError` by PyO3                                                                                                                       |
| Identifier list too large for `Vec` allocation       | `Vec::with_capacity` panics on OOM                                                                      | `pyo3_runtime.PanicException` — acceptable; OOM is unrecoverable                                                                                          |
| Unicode decode failure in numpy array                | `id.to_str()` fails                                                                                     | `unwrap()` panics → `pyo3_runtime.PanicException`. Replace `unwrap()` with `.map_err(\|e\| PyErr::new::<pyo3::exceptions::PyUnicodeDecodeError, _>(...))` |

**Guideline:** `unwrap()` is acceptable for conditions that can't happen (pre-validated
inputs, fixed-size slices). For conditions that depend on external data (numpy array
contents, allocation), use proper error propagation with `PyResult<T>`.

### 8. Build System Integration

**`pyproject.toml` changes:**

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[tool.maturin]
module-name = "cohorting._core"
python-source = "src"
manifest-path = "rust-core/Cargo.toml"
```

**`Taskfile.yaml` additions:**

```yaml
build-rust:
    desc: Build the Rust extension in release mode
    cmds:
        - uv run maturin develop --release --manifest-path rust-core/Cargo.toml
```

The Rust extension becomes the build artifact. `maturin build` produces a
platform-specific wheel with the compiled `.so`/`.dylib`/`.pyd` included.

### 9. Release Workflow

The publish CI workflow (`.github/workflows/publish.yaml`) changes from `uv build` to
`maturin build` with a matrix of platform targets:

```yaml
strategy:
    matrix:
        target:
            - linux-x86_64 # manylinux
            - linux-aarch64 # ARM64 server
            - macos-x86_64 # Intel Mac
            - macos-aarch64 # Apple Silicon
            - windows-x86_64 # win-amd64
```

Each matrix job runs:

```bash
maturin build --release --target $TARGET --out dist/
```

Wheels are uploaded to PyPI via `maturin upload` (or `twine` if already in use). The sdist
is produced by `maturin build --sdist` and includes the Rust source so users with a Rust
toolchain can compile from source on unsupported platforms.

The existing `publish.yaml` uses a single `uv build` → `twine upload` flow. The new
workflow keeps `twine upload` (or `gh release upload`) but replaces `uv build` with the
maturin matrix.

### 10. Development Workflow

**Taskfile changes:**

```yaml
build-rust:
    desc: Build the Rust extension (release)
    cmds:
        - uv run maturin develop --release --manifest-path rust-core/Cargo.toml

build-rust-dev:
    desc: Build the Rust extension (debug, fast compile)
    cmds:
        - uv run maturin develop --manifest-path rust-core/Cargo.toml
```

The existing `task install` and `task sync` call `task build-rust` as a dependency so that
a fresh checkout gets `_core` built automatically. A new `task install-dev` target builds
the debug profile for the edit-compile-test loop:

```yaml
install-dev:
    desc: Install with debug Rust build for development
    cmds:
        - task: sync
        - task: build-rust-dev
```

`task test`, `task lint`, `task fix`, `task check` all depend on `build-rust-dev` so they
work without manual setup.

**Pre-commit hook:**

Add a `maturin` pass to `.pre-commit-config.yaml`:

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
        entry: cargo clippy -- -D warnings
        language: system
        files: ^rust-core/
        pass_filenames: false
```

**Dev loop:**

```
# One-time setup
task install-dev

# Edit → rebuild → test loop
cargo build --manifest-path rust-core/Cargo.toml  # compile
task test -- tests/test_hash.py -x                 # run affected tests
```

`maturin develop` copies the built `.so` into the virtualenv, so `task test` picks it up
immediately after a `cargo build` without re-running maturin each time.

### 11. Hash Function Design

Since zerover permits breaking hash-output changes, the Rust implementation defines the
hash function going forward. The design goals:

- **Deterministic**: same (identifier, salt) always produces the same float in [0, 1)
- **Uniform**: hash outputs are uniformly distributed across [0, 1)
- **Fast**: one-shot hashing with no per-element allocation in the hot loop

The hash algorithm is:

```
hash(id, salt) = truncate_64bit(hash_fn(id.encode() ++ b"\x00" ++ salt.encode())) * 1/2^64
```

Where `hash_fn` is blake2b (default) or xxh3_64 (opt-in), and `++` is concatenation.
The `\x00` separator prevents collisions between `id="foo" + salt="bar"` and `id="foob" + salt="ar"`.

Rust uses the `blake2` and `twox-hash` crates directly — no need to mimic Python's streaming
API since there's no legacy output to match.

### 12. Platform Support

| Platform                 | Rust Tier | Notes                                                                  |
| ------------------------ | --------- | ---------------------------------------------------------------------- |
| Linux x86_64 (manylinux) | 1         | Primary target; `maturin build` produces `manylinux_2_17` wheels       |
| macOS arm64 + x86_64     | 1         | Universal2 wheels via `maturin build --target universal2-apple-darwin` |
| Windows x86_64           | 2         | `maturin build` produces win-amd64 wheels                              |
| PyPy                     | 3         | PyO3 supports PyPy via `abi3`; test in CI                              |
| musl (Alpine)            | 2         | manylinux wheels include musl via `manylinux_2_17`                     |

For unsupported platforms, users must install from source with a Rust toolchain.

### 13. Testing Strategy

#### Unit Tests (Rust)

```rust
// rust-core/src/hash.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_single_deterministic() {
        let a = hash_single(b"user_1", b"\x00exp", false);
        let b = hash_single(b"user_1", b"\x00exp", false);
        assert_eq!(a, b);
        assert!((0.0..1.0).contains(&a));
    }

    #[test]
    fn test_hash_different_salts_produce_different_results() {
        let a = hash_single(b"user_1", b"\x00exp1", false);
        let b = hash_single(b"user_1", b"\x00exp2", false);
        assert_ne!(a, b);
    }
}
```

#### Python Integration Tests

- Test that `Experiment.assign` and `hash_values` return correct types for all supported
  input types (str, int, list, np.ndarray, pd.Series, pl.Series)
- Test that deterministic mode produces consistent results for same inputs
- Test that non-deterministic mode produces values in [0, 1)
- Test that split validation still raises correct errors (validation stays in Python)

#### Performance Regression Tests (CodSpeed)

The `.github/workflows/codspeed.yaml` workflow runs `tests/test_benchmarks.py` on every
push to `main` and every PR. CodSpeed detects performance regressions by comparing wall-clock
times against the baseline.

The existing benchmark suite covers all the hot paths that the Rust migration touches:
single and batch hashing, single and batch assignment, numpy/pandas/polars backends,
xxhash mode, ORM assignment, and `Experiment` construction.

Post-migration, CodSpeed becomes the enforcement mechanism — any PR that slows down a
hot path relative to the Rust baseline is flagged automatically.

### 14. Rollout Plan

#### Phase 0: Baselining (pre-Rust)

| Step                         | What                                                                                                                                                                                                                     | Why                                                                                                                                           |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **0a. Test coverage audit**  | Ensure the existing test suite covers all public API paths: every input type, every backend, deterministic + non-deterministic, cache on/off, edge cases (empty strings, bools, ints, unicode, large inputs)             | The existing test suite becomes the acceptance suite for the Rust implementation — if tests pass against Rust, the migration is correct       |
| **0b. Performance baseline** | Run `tests/test_benchmarks.py` via CodSpeed (`task benchmarks`) to capture per-benchmark wall-clock times in the CodSpeed dashboard. These become the pre-Rust baseline that every subsequent commit is compared against | CodSpeed detects regressions automatically on PRs — the Rust migration should show speedups across every benchmark, any slowdown is a blocker |
| **0c. Hash output snapshot** | For a fixed set of (id, salt) pairs, record the Python hash outputs as `scripts/profiling/results/hash_snapshot_python.json`                                                                                             | These values WILL change in Rust (zerover allows breaking hashing). The snapshot is documentation of what changed, not a constraint           |

#### Phase 1–5: Implementation

| Phase                  | Scope                                                                                                         | Risk                                          |
| ---------------------- | ------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| **1. Core hashing**    | `hash_single`, `hash_strings` in Rust; `_dispatch_hash` delegates list/str/int paths                          | Low — single-element hash is simplest         |
| **2. Core assignment** | `assign_single`, `assign_strings` in Rust; `_dispatch_assign` delegates                                       | Low — builds on phase 1                       |
| **3. NumPy interop**   | `hash_numpy`, `assign_numpy`, `random_floats_numpy`                                                           | Medium — numpy crate API surface              |
| **4. Polars interop**  | `hash_polars`, `assign_polars` — accept PySeries, extract string chunks, hash in Rust, return new Series      | Medium — Polars Series construction from Rust |
| **5. Cached path**     | Move LRU caching to Rust (dashmap or moka) for thread-safe caching, or keep in Python as `@lru_cache` wrapper | Low if kept in Python                         |

#### Verification at each phase

1. Existing Python test suite must pass (the Phase 0 audit ensures it's comprehensive)
2. CodSpeed must show no regressions vs the Phase 0 Python baseline (and speedups on all hot-path benchmarks)
3. New Rust unit tests for edge cases discovered during Phase 0a

### 15. Open Questions

1. **Cache semantics across Rust↔Python boundary**: The existing `@lru_cache(maxsize=65536)`
   decorators live in Python. If the hot loop moves to Rust, the cache becomes unreachable.
   Options:
    - Keep cache in Python (wrap the Rust call, not the inner loop)
    - Move cache to Rust with `moka` or `dashmap` for thread safety
    - Skip cache in Rust entirely (batch workloads don't benefit from it)

    Recommendation: skip cache in Rust for the initial implementation; batch workloads are
    the primary use case for the native extension.

2. **GIL strategy**: The Rust hot loops don't call back into Python, so they can release
   the GIL. Use `py.allow_threads(|| { ... })` around the batch loop to enable true
   parallelism when called from multiple threads.

3. **Wheel size**: A compiled Rust extension adds ~1-3 MB per platform wheel. Acceptable
   for the target use case (server-side data processing). If size is a concern, strip
   symbols and use `lto = true` in the release profile.

4. **Integer identifier handling**: The current Python code normalizes integers via `str(x)`
   and bools via `int(x)` before hashing. In Rust, this becomes a simple branch or an enum
   `Identifier::Str(String) | Identifier::Int(i64)`. The Rust functions should accept both
   to avoid forcing Python callers to pre-normalize.

### 16. Rejected Alternatives

- **Cython**: Requires a compile step, tricky to support across Python versions, and
  produces harder-to-debug C code. PyO3+Rust provides stronger safety guarantees and
  easier cross-compilation via `maturin`.
- **Numba JIT**: Only works on CPython, doesn't support all Python features, and adds
  a heavy dependency for comparatively modest speedups vs a native extension.
- **Rewrite everything in Rust**: Unnecessary. The Python API surface is well-designed
  and tested; only the inner hash loops need acceleration.
