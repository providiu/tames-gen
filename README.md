# tame_phase

High-performance generator for **scored distinguished point (DP) databases** used by the wild phase of a parallel Kangaroo ECDLP solver.

---

## What it does

`tame_phase.c` generates the **scored distinguished point (DP) database** used by the wild phase (`kangaroo_wild.c by arulbero`) https://github.com/arulbero/kangaroo-wild

It runs massively parallel **"tame kangaroo" random walks** on the **secp256k1 elliptic curve**, collecting distinguished points and their distances from a known starting point.

The resulting database enables the wild phase to solve the **Elliptic Curve Discrete Logarithm Problem (ECDLP)** by detecting collisions between tame and wild walks.

The scored DP technique is based on:

> D. J. Bernstein and T. Lange
> *Computing small discrete logarithms faster*
> INDOCRYPT 2012

---

# Features

* **Multi-threaded**
  Auto-detects CPU count and scales up to **512 worker threads**

* **Batch Montgomery inversion**
  Amortizes expensive modular inversions across configurable batch sizes

* **Scored DP selection**
  Collects ~3× the target number of DPs, then selects the **most frequently visited points**

* **Hugepage support**
  Automatically tries:

  * 1GB pages
  * 2MB pages
  * regular pages with THP hints

  to reduce **TLB pressure**

* **Disk-backed mode**
  Memory-maps the hash table to a file for machines with limited RAM

* **Checkpoint / resume**

  * periodic checkpointing (default: **30 minutes**)
  * full resume capability

* **Cycle detection**

  * per-kangaroo history buffer
  * escape table mechanism

  used to break out of short cycles

---

# Output Files

The program produces **4 binary files**

Prefix format:

```
{RANGE_HIGH}_scored_{GLOBAL_BITS}
```

| File                   | Contents                                                                       |
| ---------------------- | ------------------------------------------------------------------------------ |
| `*_params.bin`         | V2 configuration header (global bits, range, q_hat, R_factor, walk parameters) |
| `*_tame_db.bin`        | Packed distance values (variable-length, little-endian signed)                 |
| `*_fingerprints.bin`   | 32-bit fingerprints for fast rejection during wild phase lookups               |
| `*_bucket_offsets.bin` | Bucket index for **O(log N)** lookups by hash prefix                           |

---

# Compilation

```bash
gcc -O3 -march=native -mavx2 -flto -funroll-loops \
    -o tame_phase tame_phase.c G_stub.c \
    -lpthread -lm
```

### Required files

```
G.h
G_stub.c
fast_inv.h
```

Descriptions:

* **G.h / G_stub.c** – secp256k1 field arithmetic
* **fast_inv.h** – DRS62 modular inversion

---

# Usage

## Basic Examples

### Generate 16M scored DPs for range 2^69 – 2^70

Small database (~500 MB)

```bash
./tame_phase -n 16777216 -r 69 70 -g 16 -w 192 -b 60 --scored
```

Estimated time: **~40 minutes** at **~1300 M/s** (192 threads)

---

### Generate 400M scored DPs for range 2^74 – 2^75

Large database (~24 GB)

```bash
./tame_phase -n 400000000 -r 74 75 -g 16 -w 192 -b 60 --scored
```

Estimated time: **~17 hours** at **~1300 M/s** (192 threads)

---

# Estimating Generation Time

In **scored mode**, the program collects **~3× the target distinguished points**, then selects the top **N**.

Each distinguished point requires on average:

```
2^global_bits
```

steps to find.

### Formulas

```
Total steps = 3 × N × 2^global_bits        (scored mode)
Total steps = N × 2^global_bits            (unscored mode)
Time (seconds) = Total steps / speed
```

---

## Example Calculation

| Parameter                   | Example        |
| --------------------------- | -------------- |
| N (target DPs)              | 400,000,000    |
| global_bits                 | 16             |
| Collect count (scored, 3×N) | 1,200,000,000  |
| Steps per DP (2^16)         | 65,536         |
| Total steps                 | ~78.6 trillion |
| Speed (192 threads)         | ~1,300 M/s     |
| Estimated time              | ~16.8 hours    |

---

**Note**

Actual speed depends on:

* CPU architecture
* number of threads
* batch size

Use the **M/s value printed during the first seconds of a run** to estimate total runtime for your hardware.


---

## Custom workers and batch size

```bash
./tame_phase -n 8500000000 -g 14 -r 89 90 -w 192 -b 20 --prefix 90_scored_14 --checkpoint 30
```

---

## Disk-backed mode

Useful for **large databases on machines with limited RAM**

```bash
./tame_phase -n 536870912 -r 74 75 -g 16 --scored --disk /mnt/fast_ssd/tame_ht
```

---

## Resume from checkpoint

```bash
./tame_phase -n 536870912 -r 74 75 -g 16 --scored --resume
```

---

# CLI Flags

| Flag             | Description                                  | Default   |
| ---------------- | -------------------------------------------- | --------- |
| `-n N`           | Target number of distinguished points        | 400000000 |
| `-g BITS`        | Global bits (DP detection mask)              | 16        |
| `-r LOW HIGH`    | Range in bits (example: `74 75` → 2^74–2^75) | 74 75     |
| `-w N`           | Number of worker threads                     | Auto      |
| `-b N`           | Batch size per worker (1–128)                | 20        |
| `-t BITS`        | Distance truncation bits                     | 0         |
| `-H BITS`        | Hash index bits for bucket offsets           | Auto      |
| `--scored`       | Enable scored DP selection                   | Off       |
| `--seed S`       | Jump table RNG seed                          | 42        |
| `--prefix P`     | Output file prefix                           | Auto      |
| `--vita N`       | Max steps per kangaroo before respawn        | 8388608   |
| `--checkpoint M` | Checkpoint interval (minutes)                | 30        |
| `--resume`       | Resume from checkpoint                       | Off       |
| `--disk PATH`    | Disk-backed hash table                       | Off       |

---

# How It Works

## Jump Table

Precomputes **2⁹ (512) random elliptic curve points** with distances in the optimal range around:

```
√(range / √(2^global_bits))
```

---

## Tame Walks

Each kangaroo starts at a **random point within the search range** and performs pseudo-random steps.

Steps are determined by hashing the **current x-coordinate** into the jump table.

---

## Distinguished Point Collection

When a kangaroo lands on a point whose **x-coordinate has `global_bits` trailing zeros**, the program records:

* fingerprint of the point
* accumulated distance

---

## Scoring (optional)

The program collects **~3× the requested DP count**, then selects the **most frequently visited points**.

It also computes:

```
q_hat
R_factor
```

These parameters are used by the **wild phase**.

---

## Export

Distinguished points are:

1. sorted by bucket
2. written into packed database files

These files are then loaded by the **wild phase solver**.

---

# Memory Requirements

The **hash table dominates memory usage**.

Assuming **32 bytes per slot** with **~50% load factor**:

| Target DPs | Approx RAM        |
| ---------- | ----------------- |
| 100M       | ~6 GB             |
| 400M       | ~24 GB            |
| 536M       | ~32 GB            |
| 1B+        | Use `--disk` mode |

---

# License

MIT License
