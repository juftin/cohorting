<h1 align="center">cohorting</h1>

<p align="center">
    <strong>Deterministic experiment cohort assignment in Python.</strong>
</p>

<p align="center">
  <a href="https://github.com/juftin/cohorting"><img src="https://img.shields.io/github/v/release/juftin/cohorting?color=blue&label=cohorting&logo=github" alt="GitHub"></a>
  <a href="https://pypi.python.org/pypi/cohorting/"><img src="https://img.shields.io/pypi/pyversions/cohorting?label=PyPI&logo=python" alt="PyPI"></a>
  <a href="https://github.com/juftin/cohorting/blob/main/LICENSE"><img src="https://img.shields.io/github/license/juftin/cohorting?color=blue&label=License" alt="GitHub License"></a>
  <a href="https://github.com/juftin/cohorting/actions/workflows/test.yaml?query=branch%3Amain"><img src="https://github.com/juftin/cohorting/actions/workflows/test.yaml/badge.svg?branch=main" alt="Testing Status"></a>
  <a href="https://app.codspeed.io/juftin/cohorting?utm_source=badge"><img src="https://img.shields.io/endpoint?url=https://codspeed.io/badge.json" alt="CodSpeed"/></a>
  <a href="https://github.com/go-task/task"><img src="https://img.shields.io/badge/task---?message=task&logo=task&color=teal&labelColor=grey" alt="task"></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
  <a href="https://github.com/pre-commit/pre-commit"><img src="https://img.shields.io/badge/pre--commit-enabled-lightgreen?logo=pre-commit" alt="pre-commit"></a>
  <a href="https://github.com/semantic-release/semantic-release"><img src="https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--release-e10079.svg" alt="semantic-release"></a>
  <a href="https://gitmoji.dev"><img src="https://img.shields.io/badge/gitmoji-%20😜%20😍-FFDD67.svg" alt="Gitmoji"></a>
</p>

---

## Features

- **Deterministic & Stateless:** Identifiers consistently map to the same cohort without database lookups.
- **Experiment Isolation:** Built-in salting ensures user assignments remain independent across different experiments.
- **Ecosystem Native:** Out-of-the-box vectorized support for `pandas`, `polars`, `numpy`, and Python ORMs/dataclasses.
- **High Performance:** Performance as a primary concern, highly optimized hashing and assignment techniques.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Class-Based API (Recommended)](#class-based-api-recommended)
    - [DataFrames](#dataframes)
    - [Numpy](#numpy)
    - [ORM / Model Objects](#orm--model-objects)
    - [Per-Experiment Backends](#per-experiment-backends)
- [Functional API](#functional-api)
- [Defining Splits](#defining-splits)
- [Configuration & Advanced Usage](#configuration--advanced-usage)
- [Pitfalls & Gotchas](#pitfalls--gotchas)

---

## Installation

```bash
pip install cohorting

# Install with optional high-performance backends
pip install "cohorting[xxhash]"
pip install "cohorting[numpy]"
pip install "cohorting[pandas]"
pip install "cohorting[polars]"

# Install everything
pip install "cohorting[numpy,pandas,polars,xxhash]"
```

## Quick Start

```python
from cohorting import Experiment, even_split

# 1. Define an experiment (50/50 split)
exp = Experiment(
    name="my-experiment",
    splits=even_split(names=["control", "treatment"]),
)

# 2. Assign a single user
cohort: str = exp.assign(data="user_abc")  # -> "control"

# 3. Assign a batch of users
cohorts: list[str] = exp.assign(data=[
    "user_1",
    "user_2",
    "user_3",
])  # -> ["treatment", "control", "treatment"]
```

---

## How It Works

1. **Hashing:** The user identifier and an experiment salt are separated by a null-byte (`\x00`) to prevent accidental collisions, concatenated, and fed into a cryptographic hash function (`blake2b` by default).
2. **Normalization:** The resulting 64-bit integer is scaled to a uniform float between `0.0` and `1.0`.
3. **Lookup:** The float is mapped to its corresponding cohort range using a fast binary search (`bisect`).

Because the experiment name acts as the namespace salt, assignments remain entirely uncorrelated across different experiments.

---

## Class-Based API (Recommended)

The `Experiment` class validates and sorts your configuration at initialization, making repeated calls significantly faster than the functional API.

```python
from cohorting import Experiment, even_split

exp = Experiment(
    name="my-experiment",
    splits=even_split(names=["control", "treatment"]),
)
# Fetch sorted cohort names
print(exp.cohorts)  # -> ["control", "treatment"]
```

### DataFrames

Directly transform datasets. The output column name defaults to `"cohort"` but can be customized using `output_column`.
Both `pandas` and `polars` are supported.

```python
import pandas as pd
import polars as pl
from cohorting import Experiment, even_split

exp = Experiment(
    name="my-experiment",
    splits=even_split(names=["control", "treatment"]),
)

df = pl.DataFrame({"user_id": ["user_1", "user_2"], "revenue": [10.0, 20.0]})

# Append cohort assignments using the default column name
cohort_df: pl.DataFrame = exp.assign_frame(df=df, id_column="user_id")
# ┌─────────┬─────────┬───────────┐
# │ user_id │ revenue │ cohort    │
# ╞═════════╪═════════╪═══════════╡
# │ user_1  │    10.0 │ control   │
# │ user_2  │    20.0 │ treatment │
# └─────────┴─────────┴───────────┘

# Or specify a custom output column
pandas_df: pd.DataFrame = df.to_pandas()
cohort_df: pd.DataFrame = exp.assign_frame(
    df=pandas_df,
    id_column="user_id",
    output_column="experiment_arm"
)
# ┌─────────┬─────────┬───────────────────┐
# │ user_id │ revenue │ experiment_arm    │
# ╞═════════╪═════════╪═══════════════════╡
# │ user_1  │    10.0 │ control           │
# │ user_2  │    20.0 │ treatment         │
# └─────────┴─────────┴───────────────────┘
```

### Numpy

Directly transform Numpy arrays. The output is a new
array of the same shape.

```python
import numpy as np
import numpy.typing as npt
from cohorting import Experiment, even_split

exp = Experiment(
    name="my-experiment",
    splits=even_split(names=["control", "treatment"]),
)

arr: npt.NDArray[np.str_] = np.array(["user_1", "user_2", "user_3"], dtype=np.str_)

hashes: npt.NDArray[np.float64] = exp.hash(
    data=arr
)  # -> array([0.12..., 0.87..., 0.44...], dtype=float64)

cohorts: npt.NDArray[np.str_] = exp.assign(
    data=arr
)  # -> array(["control", "treatment", "control"], dtype="<U9")
```

### ORM / Model Objects

`assign_orm` and `hash_orm` inspect attributes on dataclasses, Pydantic models, or standard class instances. They natively handle string or integer IDs.

```python
from dataclasses import dataclass
from cohorting import Experiment, even_split

exp = Experiment(
    name="my-experiment",
    splits=even_split(names=["control", "treatment"]),
)


@dataclass
class User:
    user_id: str
    email: str


user = User(user_id="user_123", email="alice@example.com")

# Assign a single object
cohort: str = exp.assign_orm(obj=user, id_field="user_id")  # -> "treatment"
h: float = exp.hash_orm(obj=user, id_field="user_id")  # -> 0.6968...

# Assign a list of objects
users: list[User] = [
    User(user_id="user_1", email="a@example.com"),
    User(user_id="user_2", email="b@example.com"),
]
cohorts: list[str] = exp.assign_orm(obj=users, id_field="user_id")  # -> ["control", "treatment"]
```

### Per-Experiment Backends

You can opt into the faster `xxhash` backend for specific performance-critical experiments without modifying global configurations.

```python
from cohorting import Experiment, even_split

exp = Experiment(
    name="high-throughput-test",
    splits=even_split(names=["control", "treatment"]),
    xxhash=True,  # Requires cohorting[xxhash]
)
```

> [!WARNING]
> `xxhash` generates entirely different hash spaces than `hashlib`. Do not toggle this setting while an experiment is actively running in production.

---

## Functional API

Stateless, standalone functions for quick scripting or one-off calculations. Note that splits are re-validated on every execution; prefer the `Experiment` class for production paths.

```python
from cohorting import assign_cohorts, hash_values, SplitMap

splits: SplitMap = {
    "control": {"lower": 0.0, "upper": 0.65},
    "treatment": {"lower": 0.65, "upper": 1.0},
}

# String & Integer IDs (int IDs match their string representation exactly)
cohort: str = assign_cohorts(data="user_123", splits=splits, salt="exp")
raw_hash: float = hash_values(data=123, salt="exp")

# Native Python Lists
cohort_list: list[str] = assign_cohorts(data=["user_1", "user_2"], splits=splits, salt="exp")

# NumPy Arrays
import numpy as np
import numpy.typing as npt
arr: npt.NDArray[np.str_] = np.array(["user_1", "user_2"])
cohort_arr: npt.NDArray[np.str_] = assign_cohorts(data=arr, splits=splits, salt="exp")

# Pandas Series
import pandas as pd
series = pd.Series(["user_1", "user_2"])
cohort_series: pd.Series = assign_cohorts(data=series, splits=splits, salt="exp")

# Polars Series
import polars as pl
pl_series = pl.Series(["user_1", "user_2"])
cohort_pl: pl.Series = assign_cohorts(data=pl_series, splits=splits, salt="exp")
```

---

## Defining Splits

`splits` define the boundaries of an experiment's arms. They are expressed as a
dictionary of lower and upper thresholds (`SplitMap`), or as a list of `CohortSplit` objects
(`list[CohortSplit]`). The type `SplitInput` is a union of both.

Splits must:

- Start at exactly `0.0` (`>=0`)
- End at exactly `1.0` (`<1`)
- Have no gaps or overlaps between cohorts

### Equal Split

Equally distributes the `[0.0, 1.0)` space among provided variants.

```python
from cohorting import even_split, SplitInput

splits: SplitInput = even_split(names=["control", "treatment_a", "treatment_b"])  # ~33.3% each
```

### Weighted Split

Provides precise proportional assignments. Weights must sum exactly to `1.0`.

```python
from cohorting import weighted_split, SplitMap

splits: SplitMap = weighted_split(weights={"control": 0.9, "treatment": 0.1})  # 90/10
splits: SplitMap = weighted_split(
    weights={"control": 0.6, "treatment_a": 0.2, "treatment_b": 0.2}
)  # 60/20/20
```

### Manual Definitions

For precise control over exact boundary thresholds.

```python
from cohorting import SplitMap

splits: SplitMap = {
    "control": {"lower": 0.0, "upper": 0.4},
    "treatment": {"lower": 0.4, "upper": 1.0},
}
```

You can also define splits as Pythonic dataclasses if you prefer:

```python
from cohorting import CohortSplit, SplitInput

splits: SplitInput = [
    CohortSplit(name="control", lower=0.0, upper=0.5),
    CohortSplit(name="treatment", lower=0.5, upper=1.0),
]
```

### Validation Guardrails

Splits are automatically verified at runtime. You can run manual checks with `validate_splits`.

```python
from cohorting import validate_splits, SplitMap

splits: SplitMap = {
    "control": {"lower": 0.0, "upper": 0.5},
    "treatment": {"lower": 0.5, "upper": 1.0},
}

# Raises ValueError if gaps, overlaps, or invalid boundaries ([0.0, 1.0]) are found
validate_splits(splits=splits)
```

---

## Configuration & Advanced Usage

### Global Engine Overrides

Modify the default execution behaviors globally via `cohorting.config`:

```python
import cohorting

# Switch the global engine to xxhash (~20-30% faster)
cohorting.config.xxhash = True

# Turn on LRU caching (Recommended for web apps with recurrent traffic)
# Holds up to 65,536 entries per backend; drops overhead to a dictionary lookup
cohorting.config.cache = True
```

Alternatively, you can configure the backend engine using environment variables before execution:

```bash
COHORTING_HASH_BACKEND=xxhash python main.py
```

### Non-Deterministic Simulation Mode

To run load-testing simulations without setting up mock tracking IDs, you can bypass deterministic hashing in favor of OS entropy (`os.urandom`).

```python
from cohorting import Experiment, even_split

exp = Experiment(
    name="simulation-run",
    splits=even_split(names=["control", "treatment"]),
    deterministic=False,  # Bypasses internal hash algorithms
)

# Every subsequent execution returns a pseudo-random variant
variant = exp.assign(data="user_1")
```

> [!CAUTION]
> Non-deterministic mode is immune to `random.seed` and `numpy.random.seed`. Never toggle this on in production environments.

---

## Pitfalls & Gotchas

Modifying these parameters on an active experiment will silently reassign existing users to new variants, invalidating your experiment's data integrity.

- **Switching Engines Post-Launch:** Never flip between `hashlib` and `xxhash` while an experiment is live.
- **Modifying Boundary Coordinates:** Adjusting weights, adding variant arms, or removing cohorts entirely redistributes users across the entire space. If changes are necessary, deploy a new experiment under a fresh `name`.
- **Altering the Namespace Salt:** Changing an experiment's `name` attribute shifts the underlying salt string. Treat the experiment name as an immutable asset.
- **Reusing Salts Across Features:** If Experiment A and Experiment B share an identical salt/name, user distributions will correlate perfectly (e.g., users landing in `control` for A will always land in `control` for B). Maintain distinct experiment names.

### Thread Safety Warning

`cohorting.config` acts as a **process-wide singleton and is not thread-safe**. Mutating config settings concurrently inside standard application threads will lead to race conditions.

**Best Practice:** Define your global configurations once during process initialization. If threads require localized parameters, pass explicit settings directly to individual `Experiment` instances instead.

---

## Performance Notes

- **Algorithmic Speed:** Cohort lookups use `bisect` for O(log n) performance regardless of the number of cohorts.
- **Data Extraction:** Pandas paths extract to NumPy before vectorizing to reduce per-element Python overhead.
- **Lazy Imports:** `numpy`, `pandas`, and `polars` are imported lazily — only when those types are actually passed in.
