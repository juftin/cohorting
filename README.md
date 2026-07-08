<h1 align="center">cohorting</h1>

<p align="center">
    Deterministic experiment cohort assignment
</p>

<p align="center">
  <a href="https://github.com/juftin/cohorting"><img src="https://img.shields.io/github/v/release/juftin/cohorting?color=blue&label=cohorting&logo=github" alt="GitHub"></a>
  <a href="https://pypi.python.org/pypi/cohorting/"><img src="https://img.shields.io/pypi/pyversions/cohorting?label=PyPI&logo=python" alt="PyPI"></a>
  <a href="https://github.com/juftin/cohorting/blob/main/LICENSE"><img src="https://img.shields.io/github/license/juftin/cohorting?color=blue&label=License" alt="GitHub License"></a>
  <a href="https://github.com/juftin/cohorting/actions/workflows/test.yaml?query=branch%3Amain"><img src="https://github.com/juftin/cohorting/actions/workflows/test.yaml/badge.svg?branch=main" alt="Testing Status"></a>
  <a href="https://github.com/go-task/task"><img src="https://img.shields.io/badge/task---?message=task&logo=task&color=teal&labelColor=grey" alt="task"></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
  <a href="https://github.com/pre-commit/pre-commit"><img src="https://img.shields.io/badge/pre--commit-enabled-lightgreen?logo=pre-commit" alt="pre-commit"></a>
  <a href="https://github.com/semantic-release/semantic-release"><img src="https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--release-e10079.svg" alt="semantic-release"></a>
  <a href="https://gitmoji.dev"><img src="https://img.shields.io/badge/gitmoji-%20😜%20😍-FFDD67.svg" alt="Gitmoji"></a>
</p>

## Installation

```bash
pip install cohorting
# with optional dependencies
pip install "cohorting[xxhash]"   # faster hash backend (see below)
pip install "cohorting[numpy]"
pip install "cohorting[pandas]"
pip install "cohorting[polars]"
pip install "cohorting[numpy,pandas,polars]"
```

## Quick Start

```python
from cohorting import Experiment, even_split

exp = Experiment(
    name="my-experiment",
    splits=even_split(names=["control", "treatment"]),  # 50/50 A/B split
)

cohort: str = exp.assign(data="user_abc")  # e.g. "control"
cohorts: list[str] = exp.assign(data=[
    "user_1",
    "user_2",
    "user_3",
])  # e.g. ["treatment", "control", "treatment"]
```

## How It Works

Each identifier is deterministically hashed into a float between 0 and 1. That float is then compared against each cohort's defined range — whichever range it falls into determines the cohort. Because the same identifier always produces the same float, the same user always lands in the same cohort for a given experiment.

The experiment name acts as a namespace: two experiments hashing the same user will produce different floats, so cohort assignments are independent across experiments.

### Technical Details

- The identifier and experiment salt are concatenated and fed into a hash function (`hashlib` `blake2b` by default), producing a deterministic 64-bit integer that is normalized to a float in `[0, 1)`.
- A null-byte separator (`\x00`) is inserted between the identifier and salt to prevent collisions between inputs like `id="ab", salt="c"` and `id="a", salt="bc"`.
- Cohort lookup is done with `bisect` — O(log n) regardless of the number of cohorts.
- Integer identifiers are converted to their decimal string representation before hashing, so `123` and `"123"` always land in the same cohort.
- An optional `xxhash` backend is ~20–30% faster for high-throughput workloads. See [Switching to the xxhash backend](#switching-to-the-xxhash-backend).

## Class-based API

`Experiment` is the recommended way to assign cohorts. It validates and sorts splits once at construction, so repeated `assign` and `hash` calls are faster than the functional API.

`salt` defaults to `name`, so you only need to set it explicitly when you want a different hashing namespace than the experiment name.

```python
import pandas as pd
from cohorting import Experiment, even_split

exp = Experiment(
    name="checkout-redesign",
    splits=even_split(names=["control", "treatment"]),  # 50/50 A/B split
)

cohort: str = exp.assign(data="user_123")  # e.g. "treatment"
cohorts: list[str] = exp.assign(data=[
    "user_1",
    "user_2",
    "user_3",
])  # e.g. ["control", "treatment", "control"]
hashes: list[float] = exp.hash(data=["user_1", "user_2"])  # e.g. [0.12..., 0.87...]

# List cohort names in bucket order
names: list[str] = exp.cohorts  # ["control", "treatment"]

# Add a cohort column to a DataFrame
df = pd.DataFrame(data={"user_id": ["user_1", "user_2"], "revenue": [10.0, 20.0]})
result: pd.DataFrame = exp.assign_frame(df=df, id_column="user_id")
# original columns + "cohort":
# ┌─────────┬─────────┬───────────┐
# │ user_id │ revenue │ cohort    │
# ╞═════════╪═════════╪═══════════╡
# │ user_1  │    10.0 │ control   │
# │ user_2  │    20.0 │ treatment │
# └─────────┴─────────┴───────────┘

result: pd.DataFrame = exp.assign_frame(df=df, id_column="user_id", output_column="experiment_arm")
# original columns + "experiment_arm":
# ┌─────────┬─────────┬────────────────┐
# │ user_id │ revenue │ experiment_arm  │
# ╞═════════╪═════════╪════════════════╡
# │ user_1  │    10.0 │ control        │
# │ user_2  │    20.0 │ treatment      │
# └─────────┴─────────┴────────────────┘
```

### ORM / Model Objects

`assign_orm` and `hash_orm` read a named attribute from a dataclass, Pydantic model, or any plain Python object.

```python
from dataclasses import dataclass
from cohorting import Experiment, even_split


@dataclass
class User:
    user_id: str
    email: str


@dataclass
class Account:
    account_id: int  # integer IDs are also accepted
    name: str


exp = Experiment(
    name="checkout-redesign",
    splits=even_split(names=["control", "treatment"]),  # 50/50 A/B split
)
user = User(user_id="user_123", email="alice@example.com")

cohort: str = exp.assign_orm(obj=user, id_field="user_id")  # e.g. "treatment"
h: float = exp.hash_orm(obj=user, id_field="user_id")  # e.g. 0.6968...

# Works with integer id fields too
account = Account(account_id=9001, name="Acme Corp")
cohort: str = exp.assign_orm(obj=account, id_field="account_id")  # e.g. "control"

# Also works on lists
users: list[User] = [User(user_id="user_1", email="a@x.com"), User(user_id="user_2", email="b@x.com")]
cohorts: list[str] = exp.assign_orm(obj=users, id_field="user_id")  # e.g. ["control", "treatment"]
```

### xxhash Per Experiment

Use the faster `xxhash` backend for a specific experiment without affecting global config:

```python
from cohorting import Experiment, even_split

exp = Experiment(
    name="checkout-redesign",
    splits=even_split(names=["control", "treatment"]),  # 50/50 A/B split
    xxhash=True,  # requires cohorting[xxhash]
)
```

> **Important**: `xxhash=True` produces different hash values than the default hashlib backend. Don't mix backends for the same experiment across deployments.

## Functional API

The functional API provides standalone functions for one-off use or when you prefer not to manage an `Experiment` instance. It re-validates and re-encodes splits on every call — prefer `Experiment` when making repeated calls against the same experiment configuration.

These examples share a common split definition:

```python
from cohorting import SplitMap

splits: SplitMap = {
    "control": {"lower": 0.0, "upper": 0.5},
    "treatment": {"lower": 0.5, "upper": 1.0},
}
```

### String

```python
from cohorting import assign_cohorts, hash_values

h: float = hash_values(data="user_123", salt="exp")  # e.g. 0.6968...  (always >= 0, < 1)
cohort: str = assign_cohorts(data="user_123", splits=splits, salt="exp")  # e.g. "treatment"
```

### Integer

Integer identifiers are accepted directly. They are converted to their decimal string
representation before hashing, so `123` and `"123"` always produce the same cohort.

```python
from cohorting import assign_cohorts, hash_values

h: float = hash_values(data=123, salt="exp")  # e.g. 0.4312...  (always >= 0, < 1)
cohort: str = assign_cohorts(data=123, splits=splits, salt="exp")  # e.g. "control"

# 123 (int) and "123" (str) hash identically
assert hash_values(data=123, salt="exp") == hash_values(data="123", salt="exp")
```

### List

```python
from cohorting import assign_cohorts, hash_values

users: list[str] = ["user_1", "user_2", "user_3", "user_4"]

hashes: list[float] = hash_values(data=users, salt="exp")
# e.g. [0.12..., 0.87..., 0.44..., 0.61...]  — one float per input, same order

cohorts: list[str] = assign_cohorts(data=users, splits=splits, salt="exp")
# e.g. ["control", "treatment", "control", "treatment"]  — one cohort name per input

# Lists of ints also work
user_ids: list[int] = [1001, 1002, 1003]
cohorts_by_id: list[str] = assign_cohorts(data=user_ids, splits=splits, salt="exp")
# e.g. ["treatment", "control", "treatment"]
```

### NumPy Array

```python
import numpy as np
import numpy.typing as npt
from cohorting import assign_cohorts, hash_values

arr: npt.NDArray[np.str_] = np.array(["user_1", "user_2", "user_3"], dtype=np.str_)

hashes: npt.NDArray[np.float64] = hash_values(data=arr, salt="exp")
# e.g. array([0.12..., 0.87..., 0.44...], dtype=float64)

cohorts: npt.NDArray[np.str_] = assign_cohorts(data=arr, splits=splits, salt="exp")
# e.g. array(["control", "treatment", "control"], dtype="<U9")
```

### Pandas Series

```python
import pandas as pd
from cohorting import assign_cohorts, assign_cohorts_to_frame, hash_values

series: pd.Series = pd.Series(data=["user_1", "user_2", "user_3"], name="user_id")

hashes: pd.Series = hash_values(data=series, salt="exp")
# float64 Series, original index and name preserved
# e.g. 0    0.12...
#      1    0.87...
#      2    0.44...
#      Name: user_id, dtype: float64

cohorts: pd.Series = assign_cohorts(data=series, splits=splits, salt="exp")
# object Series of cohort names, original index preserved
# e.g. 0     control
#      1    treatment
#      2     control
#      Name: user_id, dtype: object

# Directly on a DataFrame — adds a "cohort" column
df: pd.DataFrame = pd.DataFrame(data={"user_id": ["user_1", "user_2"], "revenue": [10.0, 20.0]})
result: pd.DataFrame = assign_cohorts_to_frame(df=df, id_column="user_id", splits=splits, salt="exp")
# original columns + "cohort":
# ┌─────────┬─────────┬───────────┐
# │ user_id │ revenue │ cohort    │
# ╞═════════╪═════════╪═══════════╡
# │ user_1  │    10.0 │ control   │
# │ user_2  │    20.0 │ treatment │
# └─────────┴─────────┴───────────┘
```

### Polars Series

```python
import polars as pl
from cohorting import assign_cohorts, assign_cohorts_to_frame, hash_values

series: pl.Series = pl.Series(name="user_id", values=["user_1", "user_2", "user_3"])

hashes: pl.Series = hash_values(data=series, salt="exp")
# Float64 Series, same name as input
# e.g. shape: (3,) [f64]
#      [0.12..., 0.87..., 0.44...]

cohorts: pl.Series = assign_cohorts(data=series, splits=splits, salt="exp")
# String Series, same name as input
# e.g. shape: (3,) [str]
#      ["control", "treatment", "control"]

# Directly on a DataFrame
df: pl.DataFrame = pl.DataFrame(data={"user_id": ["user_1", "user_2"], "revenue": [10.0, 20.0]})
result: pl.DataFrame = assign_cohorts_to_frame(df=df, id_column="user_id", splits=splits, salt="exp")
# original columns + "cohort":
# ┌─────────┬─────────┬───────────┐
# │ user_id ┆ revenue ┆ cohort    │
# ╞═════════╪═════════╪═══════════╡
# │ user_1  ┆ 10.0    ┆ control   │
# │ user_2  ┆ 20.0    ┆ treatment │
# └─────────┴─────────┴───────────┘
```

### ORM / Model Objects

`assign_orm` and `hash_orm` read a named attribute from a dataclass, Pydantic model, or any plain Python object.

```python
from dataclasses import dataclass
from cohorting import assign_orm, even_split, hash_orm


@dataclass
class User:
    user_id: str
    email: str


splits = even_split(names=["control", "treatment"])  # 50/50 A/B split
user = User(user_id="user_123", email="alice@example.com")

cohort: str = assign_orm(obj=user, id_field="user_id", splits=splits, salt="checkout-redesign")
# e.g. "treatment"

h: float = hash_orm(obj=user, id_field="user_id", salt="checkout-redesign")
# e.g. 0.6968...

# Also works on lists
users: list[User] = [User(user_id="user_1", email="a@x.com"), User(user_id="user_2", email="b@x.com")]
cohorts: list[str] = assign_orm(obj=users, id_field="user_id", splits=splits, salt="checkout-redesign")
# e.g. ["control", "treatment"]
```

## Defining Splits

### Equal Split

Divides `[0, 1)` into equal-width buckets:

```python
from cohorting import even_split, SplitMap

splits: SplitMap = even_split(names=["control", "treatment"])  # 50/50
splits: SplitMap = even_split(names=["control", "treatment_a", "treatment_b"])  # 33/33/33
```

### Weighted Split

Divides `[0, 1)` by proportional weights. Weights must sum to `1.0`:

```python
from cohorting import weighted_split, SplitMap

splits: SplitMap = weighted_split(weights={"control": 0.9, "treatment": 0.1})  # 90/10
splits: SplitMap = weighted_split(weights={"control": 0.6, "treatment_a": 0.2, "treatment_b": 0.2})  # 60/20/20
```

### Manual Dict

Specify exact `[lower, upper)` boundaries directly:

```python
from cohorting import SplitMap

splits: SplitMap = {
    "control": {"lower": 0.0, "upper": 0.5},
    "treatment": {"lower": 0.5, "upper": 1.0},
}
```

### Dataclass-Based

Pass a list of `CohortSplit` instances anywhere a split is accepted:

```python
from cohorting import CohortSplit, Experiment, SplitInput, assign_cohorts

splits: SplitInput = [
    CohortSplit(name="control", lower=0.0, upper=0.5),
    CohortSplit(name="treatment", lower=0.5, upper=1.0),
]

exp = Experiment(name="my-experiment", splits=splits)  # 50/50 A/B split
cohorts: list[str] = assign_cohorts(data=["user_1", "user_2"], splits=splits, salt="my-experiment")
```

### Validation

`validate_splits` is called automatically by `Experiment` and `assign_cohorts`. Call it manually when building splits by hand:

```python
from cohorting import validate_splits, SplitMap

splits: SplitMap = {"control": {"lower": 0.0, "upper": 1.0}}
validate_splits(splits=splits)  # raises ValueError on gaps, overlaps, or wrong bounds
```

Splits must:

- Start at exactly `0.0`
- End at exactly `1.0`
- Have no gaps or overlaps between cohorts

## Experiment Isolation

Each experiment uses its salt as a hash namespace. Since `salt` defaults to `name`, two experiments with different names automatically produce independent assignments for the same user:

```python
from cohorting import Experiment, even_split

splits = even_split(names=["control", "treatment"])  # 50/50 A/B split

exp_a = Experiment(name="exp-a", splits=splits)
exp_b = Experiment(name="exp-b", splits=splits)

cohort_a: str = exp_a.assign(data="user_123")  # e.g. "treatment"
cohort_b: str = exp_b.assign(data="user_123")  # e.g. "control"  (independent)
```

## Custom Output Column

```python
import pandas as pd
from cohorting import SplitMap, assign_cohorts_to_frame

splits: SplitMap = {
    "control": {"lower": 0.0, "upper": 0.5},
    "treatment": {"lower": 0.5, "upper": 1.0},
}
df = pd.DataFrame(data={"user_id": ["user_1", "user_2"]})

result: pd.DataFrame = assign_cohorts_to_frame(
    df=df,
    id_column="user_id",
    splits=splits,
    salt="exp",
    output_column="experiment_arm",  # default is "cohort"
)
# original columns + "experiment_arm":
# ┌─────────┬────────────────┐
# │ user_id │ experiment_arm │
# ╞═════════╪════════════════╡
# │ user_1  │ control        │
# │ user_2  │ treatment      │
# └─────────┴────────────────┘
```

## Switching to the xxhash Backend

The default backend is `hashlib` `blake2b` (stdlib, always available). `xxhash` is SIMD-accelerated and ~20–30% faster for high-throughput workloads — but it must be explicitly opted in to prevent surprise hash changes if `xxhash` is already installed as a dependency of something else.

Install the extra:

```bash
pip install "cohorting[xxhash]"
```

Opt in at runtime before any hashes are computed:

```python
import cohorting

cohorting.config.xxhash = True
```

Or via environment variable at process start:

```bash
COHORTING_HASH_BACKEND=xxhash python your_script.py
```

> **Important**: switching backends changes all hash outputs. Keep the setting consistent across every process in a deployment. Switching clears the internal hash cache automatically. If `xxhash` is unavailable, the env-var path (`COHORTING_HASH_BACKEND=xxhash`) warns and falls back to `hashlib`; the code-based path (`config.xxhash = True`, `Experiment(xxhash=True)`) raises `ImportError`.

### Thread safety

`cohorting.config` is a **process-wide singleton and is not thread-safe**. Mutating it from one thread (e.g. `config.xxhash = True`) affects all threads immediately, and calling `cache_clear()` during a concurrent request can cause another thread to get an un-cached result mid-flight.

**Rule of thumb**: set config once at process startup before any requests are served, then leave it alone.

For code where different coroutines or threads need different settings, use `Experiment` instead — it captures its configuration at construction time and never reads from `cohorting.config` again:

```python
from cohorting import Experiment, even_split

# Each experiment is self-contained; safe to use concurrently.
exp = Experiment(
    name="checkout-redesign",
    splits=even_split(names=["control", "treatment"]),  # 50/50 A/B split
    xxhash=True,
    cache=True,
)
```

## Non-Deterministic Assignment

By default, cohorting is fully deterministic. Set `deterministic=False` to have each assignment draw from OS entropy instead of hashing — useful for simulation, load testing, or any scenario where you want random cohort allocation that is immune to Python- or NumPy-level random seeds.

```python
import cohorting

# Global: affects all functional API calls
cohorting.config.deterministic = False

from cohorting import assign_cohorts, even_split

splits = even_split(names=["control", "treatment"])  # 50/50 A/B split
cohort: str = assign_cohorts(data="user_1", splits=splits, salt="exp")  # random each call
```

Or scoped to a single `Experiment`:

```python
from cohorting import Experiment, even_split

exp = Experiment(
    name="checkout-redesign",
    splits=even_split(names=["control", "treatment"]),  # 50/50 A/B split
    deterministic=False,
)

cohort: str = exp.assign(data="user_1")  # random each call
```

> **Important**: assignments are drawn from `os.urandom`, which is immune to `random.seed` and `numpy.random.seed`. The same identifier will produce a different cohort on every call. Do not use non-deterministic mode in production experiments — there is no way to look up a user's previously assigned cohort.

## Gotchas

These changes will silently reassign users to different cohorts, breaking the reproducibility guarantee.

### Switching the Hash Backend Mid-Experiment

`hashlib` and `xxhash` produce completely different hash values. If you start an experiment with one backend and switch to the other, every user's assignment changes. Keep the backend setting identical across all processes and deployments for the lifetime of the experiment.

### Changing Cohort Boundaries

The cohort a user lands in depends on where their hash float falls within the split ranges. Changing any boundary — even slightly — will move users near that boundary into a different cohort. This includes:

- **Adding a new cohort** to a live experiment (e.g. adding a second treatment arm redistributes a portion of users from every existing cohort)
- **Removing a cohort** (the freed range is absorbed by adjacent cohorts, moving users who were in it)
- **Reweighting cohorts** via `weighted_split` or by editing the split map directly

If you need to add a cohort, start a new experiment with a new name rather than modifying an existing one.

### Changing the Salt

The salt is the hash namespace. Changing it — including changing the experiment `name` when `salt` is left as the default — produces entirely different hash values for every user. Treat the salt as immutable once an experiment is running.

### Reusing a Salt Across Experiments

Two experiments with the same salt are not independent: a user's hash float is identical in both, so their relative cohort position is correlated. If both experiments use a 50/50 split, a user in `control` for one will always be in `control` for the other. Use a unique name (and therefore a unique salt) per experiment.

## Enabling the LRU Cache

By default, caching is **off**. Each call hashes the identifier directly with no dict overhead. This is the right default for single-pass batch jobs where each identifier is unique — caching would only add overhead and churn through evictions.

Enable caching when the **same identifiers recur** across many calls (e.g. a web server assigning cohorts to repeat visitors):

```python
import cohorting

# Global: applies to all functional API calls
cohorting.config.cache = True
```

Or scoped to a single `Experiment`:

```python
from cohorting import Experiment, even_split

exp = Experiment(
    name="checkout-redesign",
    splits=even_split(names=["control", "treatment"]),  # 50/50 A/B split
    cache=True,
)
```

With caching on, repeated calls for the same `(identifier, salt)` pair cost only a dict lookup after the first call. The cache holds up to 65 536 entries per backend; once full, least-recently-used entries are evicted automatically.

## Performance Notes

- **bisect** for O(log n) cohort lookup regardless of the number of cohorts
- Pandas paths extract to NumPy before vectorizing to reduce per-element Python overhead
- `numpy`, `pandas`, and `polars` are imported lazily — only when those types are actually passed in
