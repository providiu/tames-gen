Tame Walk Phase for Pollard's Kangaroo ECDLP Solver

What it does
tame_phase.c generates the scored distinguished point (DP) database used by the wild phase (kangaroo_wild.c). It runs massively parallel "tame kangaroo" random walks on the secp256k1 elliptic curve, collecting distinguished points and their distances from a known starting point. The resulting database enables the wild phase to solve the Elliptic Curve Discrete Logarithm Problem (ECDLP) by detecting collisions between tame and wild walks.

The scored DP technique is based on: D.J. Bernstein and T. Lange, "Computing small discrete logarithms faster", INDOCRYPT 2012.

Features
Multi-threaded — auto-detects CPU count, up to 512 worker threads
Batch Montgomery inversion — amortizes expensive modular inversions across configurable batch sizes
Scored DP selection — collects ~3x target DPs, then selects the most frequently visited (highest-quality) points
Hugepage support — automatically tries 1GB, 2MB, then regular pages with THP hints to reduce TLB pressure
Disk-backed mode — memory-maps the hash table from a file for machines with limited RAM
Checkpoint / resume — periodic checkpointing (default: every 30 min) with full resume capability
Cycle detection — per-kangaroo history buffer + escape table mechanism to break out of short cycles
Output Files
The program produces 4 binary files (prefix auto-generated as {RANGE_HIGH}_scored_{GLOBAL_BITS}):

File	Contents
*_params.bin	V2 configuration header (global bits, range, q_hat, R_factor, all walk parameters)
*_tame_db.bin	Packed distance values (variable-length, little-endian signed)
*_fingerprints.bin	32-bit fingerprints for fast rejection during wild phase lookups
*_bucket_offsets.bin	Bucket index for O(log N) lookups by hash prefix
Compilation

gcc -O3 -march=native -mavx2 -flto -funroll-loops \
    -o tame_phase tame_phase.c G_stub.c \
    -lpthread -lm
Required files: G.h, G_stub.c (secp256k1 field arithmetic), fast_inv.h (DRS62 modular inversion)

Usage

# Basic: generate 400M scored DPs for range 2^74 - 2^75
./tame_phase -n 400000000 -r 74 75 -g 16 --scored

# Custom workers and batch size
./tame_phase -n 536870912 -r 74 75 -g 16 -w 192 -b 60 --scored

# Disk-backed mode (for large databases)
./tame_phase -n 536870912 -r 74 75 -g 16 --scored --disk /mnt/fast_ssd/tame_ht

# Resume from checkpoint
./tame_phase -n 536870912 -r 74 75 -g 16 --scored --resume
CLI Flags
Flag	Description	Default
-n N	Target number of distinguished points	400,000,000
-g BITS	Global bits (DP detection mask)	16
-r LOW HIGH	Range in bits (e.g., 74 75 for 2^74–2^75)	74 75
-w N	Number of worker threads	Auto (CPU count)
-b N	Batch size per worker (1–128)	20
-t BITS	Distance truncation bits	0
-H BITS	Hash index bits for bucket offsets	Auto
--scored	Enable scored DP selection (top N by visit count)	Off
--seed S	Jump table RNG seed	42
--prefix P	Output file prefix	Auto
--vita N	Max steps per kangaroo before respawn	8,388,608
--checkpoint M	Checkpoint interval in minutes	30
--resume	Resume from previous checkpoint	Off
--disk PATH	Use disk-backed hash table	Off (RAM)
How It Works
Jump table — Precomputes 2^9 (512) random EC points with distances in the optimal range around √(range / √(2^global_bits))
Tame walks — Each kangaroo starts at a random point within the range and takes pseudo-random steps determined by hashing the current x-coordinate into the jump table
DP collection — When a kangaroo lands on a point whose x-coordinate has global_bits trailing zeros, it records the point's fingerprint and accumulated distance
Scoring (optional) — Collects ~3x the target count, then selects the most frequently visited DPs, computing quality factors q_hat and R_factor for the wild phase
Export — Sorts DPs by bucket, writes packed database files for the wild phase to load
Memory Requirements
The hash table dominates memory usage. At 32 bytes per slot with ~50% load factor:

Target DPs	Approx. RAM
100M	~6 GB
400M	~24 GB
536M	~32 GB
1B+	Use --disk mode
