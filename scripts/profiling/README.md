# Profiling Scripts

Two scripts for analyzing `cohorting` performance. Run them from the repo root after `task install`.

---

## Scripts

### `benchmark.py` — throughput timing

Measures wall-clock time and items/sec for every combination of input type, backend (hashlib / xxhash), and cache setting across four sizes (1k → 1M).

```
uv run python scripts/profiling/benchmark.py
```

Output is a table of `ms` and `M items/sec` per scenario. Use this to compare backends and cache effects at a glance.

---

### `hotspots.py` — call-graph profiling with cProfile

Profiles `Experiment.assign()` at N=100,000 for every input type (list, ndarray, pd.Series, pl.Series) for both hashlib and xxhash backends. Prints the top 20 functions by cumulative time and saves `.prof` files to `scripts/profiling/results/`.

```
uv run python scripts/profiling/hotspots.py
```

Top-level knobs (edit at the top of the file):

| Variable  | Default      | Meaning                                    |
| --------- | ------------ | ------------------------------------------ |
| `N`       | `100_000`    | Number of identifiers per run              |
| `TOP_N`   | `20`         | Functions printed per scenario             |
| `SORT_BY` | `cumulative` | Sort key (`cumulative`/`tottime`/`ncalls`) |

---

## Reading cProfile output

```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
```

- **`tottime`** — time spent _inside_ this function only (excludes callees). Use to find the expensive leaf.
- **`cumtime`** — total time including all callees. Use to find the expensive call tree.
- **`ncalls`** — call count. High count × low `tottime` = tight loop; high `tottime` = slow function body.

Start with `cumtime` to navigate the call tree, switch to `tottime` once you find the hot subtree.

---

## Tool guide

### snakeviz — interactive cProfile visualizer

Renders `.prof` files as a flame-chart in the browser.

```bash
# Inspect a single profile
snakeviz scripts/profiling/results/list_hashlib.prof

# Compare backends side by side (open two terminals)
snakeviz scripts/profiling/results/list_hashlib.prof
snakeviz scripts/profiling/results/list_xxhash.prof
```

Click any wedge to zoom in; click the header bar to zoom back out.

---

### pyinstrument — sampling profiler

Lower overhead than cProfile; great for production-like runs or when cProfile distorts timing.

```bash
uv run pyinstrument -r html -o /tmp/cohorting.html \
    scripts/profiling/hotspots.py
open /tmp/cohorting.html
```

Or profile interactively:

```python
from pyinstrument import Profiler
with Profiler() as p:
    exp.assign(ids)
p.print()
```

---

### line-profiler — line-by-line CPU timing

Pinpoints _which line_ inside a function is slow. Decorate the function you care about and run via `kernprof`.

```bash
# 1. Add @profile decorator to the function of interest (no import needed — kernprof injects it)
# 2. Run
uv run kernprof -l -v scripts/profiling/hotspots.py

# Output shows time per line in _compute_hashlib, _assign_hashlib, etc.
```

To profile `_compute_hashlib` specifically, add `@profile` above its `def` line in `src/cohorting/_hash.py`, run `kernprof`, then remove the decorator when done.

---

### memray — memory profiler

Tracks allocations frame-by-frame. Useful when you suspect GC pressure or large intermediate arrays.

```bash
uv run memray run -o /tmp/cohorting.bin \
    scripts/profiling/benchmark.py

# Flamegraph
uv run memray flamegraph /tmp/cohorting.bin -o /tmp/cohorting-mem.html
open /tmp/cohorting-mem.html

# Or live TUI
uv run memray run --live scripts/profiling/benchmark.py
```

> memray requires Linux or macOS. It is not supported on Windows.

---

## Typical workflow

1. Run `benchmark.py` to find the slow input type / backend combination.
2. Run `hotspots.py` and open the matching `.prof` in snakeviz to find the hot call tree.
3. Add `@profile` to the hottest function and run `kernprof` to find the hot line.
4. Use `memray` if allocation count looks suspicious (many short-lived objects in snakeviz).
